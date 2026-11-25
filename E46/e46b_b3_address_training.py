#!/usr/bin/env python3
"""E46-B3: collect frozen-host activations and train a supervised address space.

The Qwen host is never trained. Only address_writer and query_encoder are updated.
The script has three stages:
  collect: capture pooled activations for memory facts and train/held-out queries
  train:   contrastively align matching memory/query activations
  build:   encode stored facts with the trained writer into a persistent memory bank

Run from the repository root. Requires torch and transformers for collect; torch for
train/build. This is a small controlled retrieval experiment, not a claim of semantic
generalization from a tiny synthetic fact set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor

from E46.e46b_memory_adapter import NeuralMemoryAdapter


def load_host(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path).expanduser()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Expected a Hugging Face Transformers model directory containing config.json: {path}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float32, local_files_only=True, trust_remote_code=False
    ).to(device).eval()
    model.config.use_cache = False
    return tokenizer, model


def candidate_layers(model):
    pattern = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.\d+$")
    return [name for name, _ in model.named_modules() if pattern.search(name)]


def first_hidden(output):
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, Tensor):
        return output
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


def collect(args):
    device = torch.device("cpu" if args.device == "cpu" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, model = load_host(args.model_path, device)
    layers = candidate_layers(model)
    if not layers:
        raise RuntimeError("No decoder block candidates found in model.named_modules().")
    layer_path = args.layer_path or layers[len(layers) // 2]
    layer = model.get_submodule(layer_path)
    spec = json.loads(Path(args.spec).read_text())
    facts = spec["facts"]
    captured = {}

    def capture_hook(_module, _inputs, output):
        # Collection runs one prompt at a time, so mean pooling has no padding tokens.
        captured["vector"] = first_hidden(output).detach().float().mean(dim=1).squeeze(0).cpu()

    handle = layer.register_forward_hook(capture_hook)

    def encode(prompt):
        captured.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if "vector" not in captured:
            raise RuntimeError(f"Layer {layer_path} did not execute.")
        return captured["vector"].clone()

    records = []
    for fact in facts:
        item = {
            "capture_id": fact["id"],
            "capture_prompt": fact["capture_prompt"],
            "capture": encode(fact["capture_prompt"]),
            "train_queries": [
                {"text": q, "vector": encode(q)} for q in fact["train_queries"]
            ],
            "heldout_queries": [
                {"text": q, "vector": encode(q)} for q in fact["heldout_queries"]
            ],
        }
        records.append(item)
        print(f"captured {fact['id']}: {1 + len(item['train_queries']) + len(item['heldout_queries'])} prompts")
    handle.remove()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "model_path": str(Path(args.model_path).expanduser()),
        "model_type": getattr(model.config, "model_type", "unknown"),
        "hidden_dim": int(model.config.hidden_size),
        "layer_path": layer_path,
        "facts": records,
    }, output)
    print(json.dumps({
        "phase": "collect", "facts": len(records),
        "training_queries": sum(len(x["train_queries"]) for x in records),
        "heldout_queries": sum(len(x["heldout_queries"]) for x in records),
        "hidden_dim": int(model.config.hidden_size), "layer_path": layer_path,
        "dataset": str(output), "device": str(device),
    }, indent=2))


def load_dataset(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported E46-B3 activation dataset format.")
    if len(data["facts"]) < 3:
        raise ValueError("At least three distinct facts are required for contrastive negatives.")
    return data


def make_training_tensors(data):
    captures = torch.stack([f["capture"].float() for f in data["facts"]])
    queries, labels = [], []
    for index, fact in enumerate(data["facts"]):
        for item in fact["train_queries"]:
            queries.append(item["vector"].float())
            labels.append(index)
    if not queries:
        raise ValueError("Dataset contains no training queries.")
    return captures, torch.stack(queries), torch.tensor(labels, dtype=torch.long)


def evaluate(adapter, facts, split):
    captures = torch.stack([f["capture"].float() for f in facts])
    addresses = F.normalize(adapter.address_writer(captures), dim=-1)
    query_vectors, labels = [], []
    for index, fact in enumerate(facts):
        for item in fact[split]:
            query_vectors.append(item["vector"].float())
            labels.append(index)
    if not query_vectors:
        return {"queries": 0, "top1_accuracy": None}
    queries = F.normalize(adapter.query_encoder(torch.stack(query_vectors)), dim=-1)
    ranking = queries @ addresses.T
    target = torch.tensor(labels, dtype=torch.long)
    prediction = ranking.argmax(dim=-1)
    return {
        "queries": len(labels),
        "top1_accuracy": float((prediction == target).float().mean().item()),
        "mean_positive_rank": float(
            (ranking.argsort(dim=-1, descending=True) == target[:, None])
            .float().argmax(dim=-1).float().mean().item() + 1
        ),
    }


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = load_dataset(args.dataset)
    captures, queries, labels = make_training_tensors(data)
    hidden_dim = int(data["hidden_dim"])
    adapter = NeuralMemoryAdapter(hidden_dim, args.address_dim, args.payload_dim, top_k=1)
    # Payload projection is deliberately not part of this addressing-only experiment.
    parameters = list(adapter.address_writer.parameters()) + list(adapter.query_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    facts = data["facts"]
    before = evaluate(adapter, facts, "heldout_queries")
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        memory_addresses = F.normalize(adapter.address_writer(captures), dim=-1)
        query_addresses = F.normalize(adapter.query_encoder(queries), dim=-1)
        logits = (query_addresses @ memory_addresses.T) / args.temperature
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            train_metrics = evaluate(adapter, facts, "train_queries")
            heldout_metrics = evaluate(adapter, facts, "heldout_queries")
        score = heldout_metrics["top1_accuracy"] or 0.0
        history.append({
            "epoch": epoch, "loss": float(loss.item()),
            "train_top1": train_metrics["top1_accuracy"],
            "heldout_top1": heldout_metrics["top1_accuracy"],
        })
        if score > best_accuracy:
            best_accuracy, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} loss={loss.item():.4f} "
                  f"train_top1={train_metrics['top1_accuracy']:.3f} "
                  f"heldout_top1={heldout_metrics['top1_accuracy']:.3f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    adapter.load_state_dict(best_state)
    adapter.eval()
    out = Path(args.adapter_dir)
    adapter.save(out)
    report = {
        "experiment": "E46-B3",
        "model_path": data["model_path"],
        "layer_path": data["layer_path"],
        "facts": len(facts),
        "training_queries": int(len(queries)),
        "heldout_queries": sum(len(f["heldout_queries"]) for f in facts),
        "trainable_components": ["address_writer", "query_encoder"],
        "frozen_components": ["Qwen3 host", "payload_writer", "payload_reader"],
        "epochs": args.epochs, "best_epoch": best_epoch,
        "heldout_before_training": before,
        "heldout_after_training": evaluate(adapter, facts, "heldout_queries"),
        "best_heldout_accuracy": best_accuracy,
        "history": history,
        "adapter_dir": str(out),
        "note": "Small controlled dataset; held-out paraphrases test wording variation, not broad semantic generalization.",
    }
    (out / "address_training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


def build_bank(args):
    data = load_dataset(args.dataset)
    adapter = NeuralMemoryAdapter.load(args.adapter_dir, map_location="cpu")
    if adapter.hidden_dim != int(data["hidden_dim"]):
        raise ValueError("Dataset hidden dimension does not match trained adapter.")
    if len(adapter.bank):
        raise ValueError("Refusing to append to a non-empty bank. Use a fresh adapter directory.")
    for fact in data["facts"]:
        adapter.write(
            fact["capture"].float().unsqueeze(0),
            {"capture_id": fact["capture_id"], "layer_path": data["layer_path"],
             "model_path": data["model_path"], "model_type": data["model_type"]},
        )
    adapter.save(args.adapter_dir)
    print(json.dumps({
        "phase": "build-bank", "memory_records": len(adapter.bank),
        "adapter_dir": args.adapter_dir,
        "capture_ids": [m["capture_id"] for m in adapter.bank.metadata],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Capture frozen Qwen activations for the dataset.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--spec", default="E46/e46b_b3_dataset.json")
    p.add_argument("--output", default="runs/E46B/B3/activations.pt")
    p.add_argument("--layer-path", default=None)
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.set_defaults(func=collect)

    p = sub.add_parser("train", help="Train address writer and query encoder.")
    p.add_argument("--dataset", default="runs/E46B/B3/activations.pt")
    p.add_argument("--adapter-dir", default="runs/E46B/B3/adapter")
    p.add_argument("--address-dim", type=int, default=128)
    p.add_argument("--payload-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--log-every", type=int, default=25)
    p.set_defaults(func=train)

    p = sub.add_parser("build-bank", help="Encode facts into the persistent bank.")
    p.add_argument("--dataset", default="runs/E46B/B3/activations.pt")
    p.add_argument("--adapter-dir", default="runs/E46B/B3/adapter")
    p.set_defaults(func=build_bank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
