#!/usr/bin/env python3
"""Single-GPU E49 pilot runner for CUDA or ROCm PyTorch."""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from e49_model import E49Config, E49LM

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["dense", "lowrank", "latent"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    for name, default in [("layers",4),("d-model",256),("heads",4),("kv-heads",4),("d-ff",704),("context",256),("rank",64),("latent-dim",16),("constructor-hidden",256),("constructor-type-dim",32),("constructor-fourier-bands",8)]:
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-tokens", type=int, default=10_000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--checkpoint-every-steps", type=int, default=100)
    p.add_argument("--validation-stride", type=int, default=32)
    p.add_argument("--validation-batch-size", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=4901)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sampling-mode", choices=["random", "unique_blocks"], default="random",
                    help="unique_blocks visits non-overlapping context blocks once per epoch")
    return p.parse_args()

def load_tokens(path, context, vocab_size):
    if not path.exists():
        raise SystemExit(f"Missing token file: {path}")
    data = np.memmap(path, dtype=np.uint32, mode="r")
    if data.size <= context + 1:
        raise SystemExit(f"Not enough tokens in {path}: {data.size}")
    if int(data.max()) >= vocab_size:
        raise SystemExit(f"Token ID in {path} exceeds --vocab-size {vocab_size}")
    return data

class UniqueBlockSampler:
    """Shuffle non-overlapping next-token blocks without replacement per epoch."""

    def __init__(self, token_count, context, seed):
        self.context = int(context)
        self.num_blocks = (int(token_count) - 1) // self.context
        if self.num_blocks < 1:
            raise ValueError("Training data is too short for unique block sampling")
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = self.rng.permutation(self.num_blocks)

    def next_starts(self, batch_size):
        batch_size = int(batch_size)
        if batch_size < 1 or batch_size > self.num_blocks:
            raise ValueError("Batch size must fit within one unique-block epoch")
        # Drop an incomplete tail rather than mixing epochs in one batch.
        if self.cursor + batch_size > self.num_blocks:
            self.epoch += 1
            self.order = self.rng.permutation(self.num_blocks)
            self.cursor = 0
        selected = self.order[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        if self.cursor == self.num_blocks:
            self.epoch += 1
            self.order = self.rng.permutation(self.num_blocks)
            self.cursor = 0
        return (selected * self.context).tolist()

    def state_dict(self):
        return {
            "context": self.context,
            "num_blocks": self.num_blocks,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["context"] != self.context or state["num_blocks"] != self.num_blocks:
            raise ValueError("Saved unique-block sampler does not match current training data")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.order = np.asarray(state["order"], dtype=np.int64)
        if self.order.size != self.num_blocks or not 0 <= self.cursor < self.num_blocks:
            raise ValueError("Invalid saved unique-block sampler state")
        self.rng.bit_generator.state = state["rng_state"]


def make_batch(data, batch_size, context, device, rng, block_sampler=None):
    if block_sampler is None:
        starts = torch.randint(0, len(data)-context-1, (batch_size,), generator=rng).numpy()
    else:
        starts = block_sampler.next_starts(batch_size)
    arr = np.stack([data[int(i):int(i)+context+1].astype(np.int64) for i in starts])
    tokens = torch.from_numpy(arr).to(device=device, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]

def evaluate(model, data, context, stride, batch_size, device):
    """Score every validation target once using deterministic sliding context."""
    import torch.nn.functional as F

    model.eval()
    loss_sum = 0.0
    token_count = 0
    pending_x, pending_y = [], []

    def score_batch(xs, ys):
        nonlocal loss_sum, token_count
        inputs = torch.from_numpy(np.stack(xs)).to(device=device, dtype=torch.long)
        labels = torch.from_numpy(np.stack(ys)).to(device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        selected_logits = logits[:, context-stride:, :].contiguous()
        selected_labels = labels[:, context-stride:].contiguous()
        loss = F.cross_entropy(selected_logits.view(-1, selected_logits.size(-1)),
                               selected_labels.view(-1), reduction="sum")
        loss_sum += float(loss.float())
        token_count += selected_labels.numel()

    def score_single(x, y, score_from):
        nonlocal loss_sum, token_count
        inputs = torch.from_numpy(x[None, :]).to(device=device, dtype=torch.long)
        labels = torch.from_numpy(y[None, :]).to(device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        selected_logits = logits[:, score_from:, :].contiguous()
        selected_labels = labels[:, score_from:].contiguous()
        loss = F.cross_entropy(selected_logits.view(-1, selected_logits.size(-1)),
                               selected_labels.view(-1), reduction="sum")
        loss_sum += float(loss.float())
        token_count += selected_labels.numel()

    with torch.inference_mode():
        target_start = 0
        total_targets = len(data) - 1
        while target_start < total_targets:
            target_end = min(target_start + stride, total_targets)
            begin = max(0, target_end - context)
            tokens = np.asarray(data[begin:target_end+1], dtype=np.int64)
            x, y = tokens[:-1], tokens[1:]
            score_count = target_end - target_start
            score_from = len(y) - score_count
            if len(x) == context and score_count == stride:
                pending_x.append(x)
                pending_y.append(y)
                if len(pending_x) == batch_size:
                    score_batch(pending_x, pending_y)
                    pending_x.clear()
                    pending_y.clear()
            else:
                if pending_x:
                    score_batch(pending_x, pending_y)
                    pending_x.clear()
                    pending_y.clear()
                score_single(x, y, score_from)
            target_start = target_end
        if pending_x:
            score_batch(pending_x, pending_y)

    torch.cuda.synchronize(device)
    model.train()
    mean_loss = loss_sum / max(1, token_count)
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(20.0, mean_loss)),
        "scored_tokens": token_count,
    }

def main():
    a = parse_args()
    positive = [a.tokens,a.micro_batch,a.grad_accum,a.eval_every_steps,a.checkpoint_every_steps,
                a.validation_stride,a.validation_batch_size,a.log_every]
    if min(positive) <= 0:
        raise SystemExit("Token, batch, and interval values must be positive")
    if a.validation_stride > a.context:
        raise SystemExit("--validation-stride cannot exceed --context")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/ROCm GPU visible to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This runner requires GPU BF16 support")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg = E49Config(
        vocab_size=a.vocab_size,n_layers=a.layers,d_model=a.d_model,n_heads=a.heads,
        n_kv_heads=a.kv_heads,d_ff=a.d_ff,context=a.context,mode=a.mode,
        rank=a.rank,latent_dim=a.latent_dim,constructor_hidden=a.constructor_hidden,
        constructor_type_dim=a.constructor_type_dim,constructor_fourier_bands=a.constructor_fourier_bands)
    cfg.validate()
    manifest = a.data_dir / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if int(meta.get("vocab_size",-1)) != a.vocab_size:
            raise SystemExit(f"Manifest vocab_size {meta.get('vocab_size')} != requested {a.vocab_size}")
    train = load_tokens(a.data_dir/"train.bin",a.context,a.vocab_size)
    valid = load_tokens(a.data_dir/"validation.bin",a.context,a.vocab_size)
    model = E49LM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=a.weight_decay)
    tokens_per_step = a.micro_batch*a.context*a.grad_accum
    planned_steps = math.ceil(a.tokens/tokens_per_step)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    ckpt = a.output_dir/"latest.pt"
    metrics = a.output_dir/"metrics.jsonl"
    step0 = tokens_seen = 0
    best_val_loss = float("inf")
    if a.resume:
        if not ckpt.exists(): raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["config"] != cfg.to_dict() or state["mode"] != a.mode:
            raise SystemExit("Checkpoint config/mode mismatch")
        saved_sampling_mode = state.get("sampling_mode", "random")
        if saved_sampling_mode != a.sampling_mode:
            raise SystemExit("Checkpoint sampling mode mismatch")
        if block_sampler is not None:
            if "block_sampler" not in state or state["block_sampler"] is None:
                raise SystemExit("Checkpoint has no unique-block sampler state; cannot resume safely")
            block_sampler.load_state_dict(state["block_sampler"])
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step0 = int(state["step"]); tokens_seen = int(state["tokens_seen"])
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        best_path = a.output_dir/"best.pt"
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            best_val_loss = min(best_val_loss, float(best_state.get("validation_loss", float("inf"))))
    data_rng = torch.Generator(device="cpu").manual_seed(a.seed+101)
    eval_rng = torch.Generator(device="cpu").manual_seed(a.seed+202)
    block_sampler = (UniqueBlockSampler(len(train), a.context, a.seed+303)
                     if a.sampling_mode == "unique_blocks" else None)
    if a.resume and "data_rng" in state:
        data_rng.set_state(state["data_rng"]); eval_rng.set_state(state["eval_rng"])
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["gpu_rng"])
    info = {
        "experiment":"E49-local","mode":a.mode,"config":cfg.to_dict(),
        "parameters":model.parameter_breakdown(),"tokens_per_step":tokens_per_step,
        "planned_steps":planned_steps,"seed":a.seed,"sampling_mode":a.sampling_mode,
        "device":torch.cuda.get_device_name(0),
        "torch":torch.__version__,"hip":torch.version.hip,
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        "validation_protocol":{"method":"full deterministic sliding-window validation",
            "stride":a.validation_stride,"batch_size":a.validation_batch_size,
            "each_next_token_target_scored_once":True}}
    if step0 == 0: (a.output_dir/"run_config.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2),flush=True)
    model.train(); torch.cuda.reset_peak_memory_stats(device)
    started = time.time(); initial_tokens = tokens_seen
    for step in range(step0,planned_steps):
        if tokens_seen < a.warmup_tokens:
            lr = a.lr*(tokens_seen+1)/max(1,a.warmup_tokens)
        else:
            progress = (tokens_seen-a.warmup_tokens)/max(1,a.tokens-a.warmup_tokens)
            lr = a.lr*(a.min_lr_ratio+(1-a.min_lr_ratio)*0.5*(1+math.cos(math.pi*min(1.0,progress))))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True); train_loss = 0.0
        for _ in range(a.grad_accum):
            x,y = make_batch(train,a.micro_batch,a.context,device,data_rng,block_sampler)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,loss = model(x,y); loss = loss/a.grad_accum
            loss.backward(); train_loss += float(loss.detach().float())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); tokens_seen += tokens_per_step
        if (step+1)%a.log_every == 0 or step+1 == planned_steps:
            elapsed = max(time.time()-started,1e-9)
            row = {"type":"train","step":step+1,"tokens_seen":tokens_seen,"loss":train_loss,
                   "lr":lr,"tokens_per_second":(tokens_seen-initial_tokens)/elapsed,
                   "elapsed_seconds":elapsed,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes":torch.cuda.max_memory_reserved(device)}
            with metrics.open("a") as f: f.write(json.dumps(row)+"\n")
            print(f"step={step+1}/{planned_steps} tokens={tokens_seen:,} loss={train_loss:.5f} tok/s={row['tokens_per_second']:.1f}",flush=True)
        if (step+1)%a.eval_every_steps == 0 or step+1 == planned_steps:
            val_metrics = evaluate(model,valid,a.context,a.validation_stride,
                                   a.validation_batch_size,device)
            val_loss = val_metrics["validation_loss"]
            row = {"type":"validation","step":step+1,"tokens_seen":tokens_seen,
                   "val_loss":val_loss,"perplexity":val_metrics["validation_perplexity"],
                   "scored_tokens":val_metrics["scored_tokens"],
                   "validation_stride":a.validation_stride}
            with metrics.open("a") as f: f.write(json.dumps(row)+"\n")
            print(f"EVAL step={step+1} val_loss={val_loss:.6f} "
                  f"ppl={row['perplexity']:.4f} tokens={row['scored_tokens']:,}",flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {"mode":a.mode,"config":cfg.to_dict(),"step":step+1,
                    "tokens_seen":tokens_seen,
                    "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "validation_loss":val_loss,"validation_perplexity":row["perplexity"],
                    "validation_scored_tokens":row["scored_tokens"],
                    "validation_stride":a.validation_stride}
                best_path = a.output_dir/"best.pt"
                temp_best = best_path.with_suffix(".tmp")
                torch.save(best_state,temp_best); temp_best.replace(best_path)
                print(f"NEW BEST step={step+1} val_loss={val_loss:.6f} checkpoint={best_path}",flush=True)
        if (step+1)%a.checkpoint_every_steps == 0 or step+1 == planned_steps:
            state = {"mode":a.mode,"config":cfg.to_dict(),"step":step+1,"tokens_seen":tokens_seen,
                     "model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "cpu_rng":torch.get_rng_state(),"gpu_rng":torch.cuda.get_rng_state(),
                     "data_rng":data_rng.get_state(),"eval_rng":eval_rng.get_state(),
                     "sampling_mode":a.sampling_mode,
                     "block_sampler":block_sampler.state_dict() if block_sampler is not None else None,
                     "best_val_loss":best_val_loss}
            temp = ckpt.with_suffix(".tmp"); torch.save(state,temp); temp.replace(ckpt)
    print(f"Finished {a.mode}: {tokens_seen:,} tokens; latest={ckpt}; best_val_loss={best_val_loss:.6f}",flush=True)

if __name__ == "__main__":
    main()
