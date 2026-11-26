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
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=4901)
    p.add_argument("--resume", action="store_true")
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

def make_batch(data, batch_size, context, device, rng):
    starts = torch.randint(0, len(data)-context-1, (batch_size,), generator=rng).numpy()
    arr = np.stack([data[int(i):int(i)+context+1].astype(np.int64) for i in starts])
    tokens = torch.from_numpy(arr).to(device=device, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]

def evaluate(model, data, args, device, rng):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = make_batch(data, args.micro_batch, args.context, device, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss_sum += float(loss.float()) * y.numel()
            token_count += y.numel()
    model.train()
    return loss_sum / max(1, token_count)

def main():
    a = parse_args()
    positive = [a.tokens,a.micro_batch,a.grad_accum,a.eval_every_steps,a.checkpoint_every_steps,a.eval_batches,a.log_every]
    if min(positive) <= 0:
        raise SystemExit("Token, batch, and interval values must be positive")
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
    if a.resume:
        if not ckpt.exists(): raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["config"] != cfg.to_dict() or state["mode"] != a.mode:
            raise SystemExit("Checkpoint config/mode mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step0 = int(state["step"]); tokens_seen = int(state["tokens_seen"])
    data_rng = torch.Generator(device="cpu").manual_seed(a.seed+101)
    eval_rng = torch.Generator(device="cpu").manual_seed(a.seed+202)
    if a.resume and "data_rng" in state:
        data_rng.set_state(state["data_rng"]); eval_rng.set_state(state["eval_rng"])
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["gpu_rng"])
    info = {
        "experiment":"E49-local","mode":a.mode,"config":cfg.to_dict(),
        "parameters":model.parameter_breakdown(),"tokens_per_step":tokens_per_step,
        "planned_steps":planned_steps,"seed":a.seed,"device":torch.cuda.get_device_name(0),
        "torch":torch.__version__,"hip":torch.version.hip,
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None}
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
            x,y = make_batch(train,a.micro_batch,a.context,device,data_rng)
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
            val_loss = evaluate(model,valid,a,device,eval_rng)
            row = {"type":"validation","step":step+1,"tokens_seen":tokens_seen,
                   "val_loss":val_loss,"perplexity":math.exp(min(20.0,val_loss))}
            with metrics.open("a") as f: f.write(json.dumps(row)+"\n")
            print(f"EVAL step={step+1} val_loss={val_loss:.6f} ppl={row['perplexity']:.4f}",flush=True)
        if (step+1)%a.checkpoint_every_steps == 0 or step+1 == planned_steps:
            state = {"mode":a.mode,"config":cfg.to_dict(),"step":step+1,"tokens_seen":tokens_seen,
                     "model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "cpu_rng":torch.get_rng_state(),"gpu_rng":torch.cuda.get_rng_state(),
                     "data_rng":data_rng.get_state(),"eval_rng":eval_rng.get_state()}
            temp = ckpt.with_suffix(".tmp"); torch.save(state,temp); temp.replace(ckpt)
    print(f"Finished {a.mode}: {tokens_seen:,} tokens; checkpoint={ckpt}",flush=True)

if __name__ == "__main__":
    main()
