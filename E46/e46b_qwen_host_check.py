#!/usr/bin/env python3
"""E46-B1: inspect a local Hugging Face checkpoint and test activation persistence.

Run capture and inject as separate commands/processes:
  python E46/e46b_qwen_host_check.py capture --model-path /path/to/Qwen3-1.7B-Base
  python E46/e46b_qwen_host_check.py inject --model-path /path/to/Qwen3-1.7B-Base

Requires torch and transformers. The checkpoint must already exist locally.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

# Allow both `python E46/script.py` and `python -m E46.script` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from E46.e46b_memory_adapter import ActivationHook, NeuralMemoryAdapter


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_host(model_path: str, device: torch.device):
    model_path = Path(model_path).expanduser()
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        raise ValueError(
            "This runner expects a Hugging Face Transformers model directory, not a GGUF file. "
            "The supplied file appears to be a GGUF checkpoint. Also verify that the model "
            "is a text causal language model; Qwen3-TTS is a speech-generation model, not "
            "the Qwen3-1.7B-Base text host used by this experiment."
        )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path is not a local directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise ValueError(
            f"No config.json found in {model_path}. This runner requires a local "
            "Hugging Face Transformers checkpoint directory, not an LM Studio GGUF file."
        )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True,
        trust_remote_code=False,
    )
    model.to(device).eval()
    model.config.use_cache = False
    return tokenizer, model


def candidate_layers(model):
    pattern = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.\d+$")
    return [name for name, _ in model.named_modules() if pattern.search(name)]


def hidden_from_output(output):
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("Selected decoder module did not return a hidden-state tensor.")


def run_capture(args):
    device = choose_device()
    tokenizer, model = load_host(args.model_path, device)
    layers = candidate_layers(model)
    if not layers:
        raise RuntimeError("No likely decoder block paths found. Inspect model.named_modules().")
    layer_path = args.layer_path or layers[len(layers) // 2]
    layer = model.get_submodule(layer_path)
    hidden_dim = int(getattr(model.config, "hidden_size"))
    adapter_dir = Path(args.adapter_dir)
    adapter_path = adapter_dir / "adapter.pt"
    if adapter_path.exists():
        adapter = NeuralMemoryAdapter.load(adapter_dir, map_location=device)
        if adapter.hidden_dim != hidden_dim:
            raise ValueError("Saved adapter hidden width does not match this host model.")
    else:
        adapter = NeuralMemoryAdapter(hidden_dim, args.address_dim, args.payload_dim,
                                      args.top_k)
    if adapter.bank.metadata:
        prior_layer = adapter.bank.metadata[-1].get("layer_path")
        prior_model = adapter.bank.metadata[-1].get("model_path")
        if prior_layer and prior_layer != layer_path:
            raise ValueError(f"Existing memories were captured at {prior_layer!r}, not {layer_path!r}.")
        if prior_model and prior_model != args.model_path:
            raise ValueError("Existing memory bank belongs to a different model path.")
    adapter.to(device=device, dtype=next(model.parameters()).dtype)

    captured = {}
    def capture_hook(_module, _inputs, output):
        captured["hidden"] = hidden_from_output(output).detach()
    handle = layer.register_forward_hook(capture_hook)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("The selected decoder layer did not execute.")
    hidden = captured["hidden"]
    if hidden.shape[-1] != hidden_dim:
        raise ValueError(f"Captured width {hidden.shape[-1]} != config hidden_size {hidden_dim}.")
    adapter.write(hidden, {"capture_id": args.capture_id, "layer_path": layer_path,
                           "model_path": args.model_path,
                           "model_type": getattr(model.config, "model_type", "unknown")})
    adapter.save(adapter_dir)
    print(json.dumps({
        "phase": "capture", "model_path": args.model_path, "device": str(device),
        "dtype": str(next(model.parameters()).dtype), "layer_path": layer_path,
        "candidate_layers": layers, "hidden_shape": list(hidden.shape),
        "hidden_dim": hidden_dim, "memory_records": len(adapter.bank),
        "adapter_dir": str(adapter_dir),
        "note": "Run inject in a separate process to verify persisted reload.",
    }, indent=2))


def run_inject(args):
    device = choose_device()
    tokenizer, model = load_host(args.model_path, device)
    adapter = NeuralMemoryAdapter.load(args.adapter_dir, map_location=device)
    layers = candidate_layers(model)
    saved_layer = adapter.bank.metadata[-1].get("layer_path") if adapter.bank.metadata else None
    layer_path = args.layer_path or saved_layer
    if not layer_path:
        raise RuntimeError("No saved layer path; pass --layer-path explicitly.")
    if saved_layer and layer_path != saved_layer:
        raise ValueError(f"Memory was captured at {saved_layer!r}, not {layer_path!r}.")
    saved_model = adapter.bank.metadata[-1].get("model_path") if adapter.bank.metadata else None
    if saved_model and saved_model != args.model_path:
        raise ValueError("Memory bank was captured from a different model path.")
    if layer_path not in layers:
        raise ValueError(f"Saved layer path {layer_path!r} is not present in this model.")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    parameter_versions = [parameter._version for parameter in model.parameters()]
    query_activation = {}
    query_layer = model.get_submodule(layer_path)
    def capture_query(_module, _inputs, output):
        query_activation["hidden"] = hidden_from_output(output).detach()
    query_handle = query_layer.register_forward_hook(capture_query)
    with torch.no_grad():
        baseline = model(**inputs, use_cache=False).logits
    query_handle.remove()
    if "hidden" not in query_activation:
        raise RuntimeError("The selected decoder layer did not execute for the query.")
    with torch.no_grad():
        _recalled, selected_scores, selected_indices = adapter.recall(
            query_activation["hidden"], return_indices=True
        )
        query_vector = torch.nn.functional.normalize(
            adapter.query_encoder(adapter.pool(query_activation["hidden"])).float(),
            dim=-1,
        )
        _all_payloads, all_scores, all_indices = adapter.bank.retrieve(
            query_vector, top_k=len(adapter.bank), return_indices=True
        )
    selected_set = set(selected_indices[0].tolist())
    retrieval_report = []
    for score, index in zip(all_scores[0].tolist(), all_indices[0].tolist()):
        metadata = adapter.bank.metadata[index] if index < len(adapter.bank.metadata) else {}
        retrieval_report.append({
            "memory_index": int(index),
            "capture_id": metadata.get("capture_id", f"record-{index}"),
            "metadata": metadata,
            "cosine_score": float(score),
            "selected_for_injection": int(index) in selected_set,
        })
    adapter.residual_gate.data.fill_(args.gate)
    with ActivationHook(model, adapter, layer_path, freeze_base=True):
        with torch.no_grad():
            adapted = model(**inputs, use_cache=False).logits
    base_weights_unchanged = all(
        parameter._version == version
        for parameter, version in zip(model.parameters(), parameter_versions)
    )
    if not base_weights_unchanged:
        raise RuntimeError("A base-model parameter changed during the adapter smoke test.")
    delta = (adapted.float() - baseline.float()).abs()
    if not torch.isfinite(adapted).all():
        raise RuntimeError("Adapted model produced non-finite logits.")
    print(json.dumps({
        "phase": "inject", "model_path": args.model_path, "device": str(device),
        "layer_path": layer_path, "memory_records": len(adapter.bank),
        "retrieval_ranking_all_memories": retrieval_report,
        "gate": args.gate, "logits_shape": list(adapted.shape),
        "max_abs_logit_delta": float(delta.max().item()),
        "mean_abs_logit_delta": float(delta.mean().item()),
        "base_trainable_parameters": sum(p.numel() for p in model.parameters()
                                         if p.requires_grad),
        "base_parameter_versions_unchanged": base_weights_unchanged,
        "finite_logits": True,
        "interpretation": (
            "A finite nonzero delta verifies that the hook changes this forward pass; "
            "it does not establish that the recalled memory is useful."
            if args.gate != 0 else
            "A zero gate should preserve the baseline logits up to numerical tolerance."
        ),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["capture", "inject"])
    parser.add_argument("--model-path", required=True,
                        help="Local directory containing the downloaded checkpoint.")
    parser.add_argument("--adapter-dir", default="runs/E46B/qwen_memory")
    parser.add_argument("--layer-path", default=None,
                        help="Exact decoder block path; capture auto-selects the middle layer.")
    parser.add_argument("--prompt", default="A short neutral sentence for activation inspection.")
    parser.add_argument("--capture-id", default="capture-001")
    parser.add_argument("--address-dim", type=int, default=128)
    parser.add_argument("--payload-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--gate", type=float, default=0.0)
    args = parser.parse_args()
    if args.phase == "capture": run_capture(args)
    else: run_inject(args)


if __name__ == "__main__":
    main()
