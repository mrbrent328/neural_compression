#!/usr/bin/env python3
"""R5: latent diffusion over the sixteen local states produced by R3.

The R3 autoencoder is frozen. This script encodes CIFAR-10 into [16, 32]
local states, learns their distribution with a compact noise-prediction
transformer, then decodes sampled states with the original shared R3 decoder.

Example:
 python artifacts/R5_latent_diffusion.py --data-root /path/to/cifar10
 --checkpoint artifacts/real_cifar10_local_interaction/r3_local_interaction_40000.pt
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from R3_local_interaction_cifar10 import (  # noqa: E402
    CIFARParquetDataset, LocalInteractionAutoencoder, find_parquet,
    fixed_split_indices, make_subset,
)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def time_embedding(t, dim=128):
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                     * (-math.log(10000.0) / max(half - 1, 1)))
    a = t.float().unsqueeze(1) * freq.unsqueeze(0)
    e = torch.cat((a.sin(), a.cos()), dim=1)
    return F.pad(e, (0, dim - e.shape[1]))

class Block(nn.Module):
    def __init__(self, dim=32, heads=4, ff=128):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
    def forward(self, x):
        y = self.n1(x); a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))

class LatentDenoiser(nn.Module):
    def __init__(self, tokens=16, dim=32, depth=4, time_dim=128):
        super().__init__()
        self.tokens, self.dim, self.time_dim = tokens, dim, time_dim
        self.in_proj = nn.Linear(dim, dim)
        self.pos = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                  nn.Linear(time_dim, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x, t):
        h = self.in_proj(x) + self.pos + self.time(time_embedding(t, self.time_dim)).unsqueeze(1)
        for block in self.blocks: h = block(h)
        return self.out_proj(self.out_norm(h))

def alpha_bar(steps, device):
    s = 0.008
    x = torch.linspace(0, steps, steps + 1, device=device)
    a = torch.cos(((x / steps + s) / (1 + s)) * math.pi / 2).square()
    return (a / a[0]).clamp(1e-5, 1.0)

@torch.no_grad()
def encode(encoder, loader, device, cache):
    if cache.exists():
        print(f"Using latent cache: {cache}", flush=True)
        return torch.load(cache, map_location="cpu", weights_only=False).float()
    encoder.eval(); parts = []
    print("Encoding images through frozen R3 encoder...", flush=True)
    for i, (images, _) in enumerate(loader, 1):
        _, local, _ = encoder(images.to(device, non_blocking=True))
        parts.append(local.cpu().float())
        if i % 25 == 0: print(f" encoded batches: {i}", flush=True)
    z = torch.cat(parts)
    torch.save(z, cache)
    print(f"Cached latents: {tuple(z.shape)}", flush=True)
    return z

def save_grid(images, path, cols=8, scale=8):
    """Save a legible contact sheet; upscale pixels without blurring them."""
    images = images.detach().cpu().clamp(0, 1)
    n, _, h, w = images.shape
    rows = math.ceil(n / cols)
    pad = 12
    tile_w, tile_h = w * scale, h * scale
    canvas = Image.new(
        "RGB",
        (cols*tile_w+(cols+1)*pad, rows*tile_h+(rows+1)*pad),
        (28,30,36),
    )
    for i, im in enumerate(images):
        arr = (im.permute(1,2,0).numpy()*255).round().astype(np.uint8)
        tile = Image.fromarray(arr).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        canvas.paste(tile, (pad+(i%cols)*(tile_w+pad), pad+(i//cols)*(tile_h+pad)))
    canvas.save(path)

@torch.no_grad()
def sample(denoiser, decoder, mean, std, device, count, sample_steps, diffusion_steps, seed):
    denoiser.eval(); decoder.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(count, denoiser.tokens, denoiser.dim, device=device, generator=gen)
    ab = alpha_bar(diffusion_steps, device)
    ts = torch.linspace(diffusion_steps-1, 0, sample_steps, device=device).long()
    for i, ti in enumerate(ts):
        t = torch.full((count,), int(ti.item()), device=device, dtype=torch.long)
        eps = denoiser(x, t); a = ab[ti]
        x0 = (x - (1-a).sqrt()*eps) / a.sqrt()
        if i == len(ts)-1: x = x0
        else:
            ap = ab[ts[i+1]]
            x = ap.sqrt()*x0 + (1-ap).sqrt()*eps  # deterministic DDIM
    generated_latents = x * std + mean
    return decoder.decode_local(generated_latents), generated_latents, x

def distribution_summary(values):
    """Compact distribution summary over [N, tokens, dimensions]."""
    a = values.detach().float().cpu().numpy()
    flat = a.reshape(-1)
    per_dim = a.reshape(-1, a.shape[-1])
    return {
        "shape": list(a.shape),
        "overall": {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "q01": float(np.quantile(flat, 0.01)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.median(flat)),
            "q95": float(np.quantile(flat, 0.95)),
            "q99": float(np.quantile(flat, 0.99)),
        },
        "per_dimension_mean": per_dim.mean(axis=0).tolist(),
        "per_dimension_std": per_dim.std(axis=0).tolist(),
    }


def save_latent_diagnostics(real_latents, generated_latents, generated_normalized, mean, std, out):
    """Compare genuine R3 states with R5 samples in both raw and normalized space."""
    real_normalized = ((real_latents - mean.cpu()) / std.cpu()).clamp(-8, 8)
    generated_normalized = generated_normalized.detach().cpu()
    real_normalized = real_normalized.detach().cpu()
    generated_latents = generated_latents.detach().cpu()
    real_latents = real_latents.detach().cpu()

    # Fraction of generated normalized coordinates outside the central range
    # observed in genuine normalized latents.
    real_flat = real_normalized.reshape(-1)
    gen_flat = generated_normalized.reshape(-1)
    low, high = torch.quantile(real_flat, torch.tensor([0.01, 0.99]))
    outside = ((gen_flat < low) | (gen_flat > high)).float().mean()

    report = {
        "real_raw": distribution_summary(real_latents),
        "generated_raw": distribution_summary(generated_latents),
        "real_normalized": distribution_summary(real_normalized),
        "generated_normalized": distribution_summary(generated_normalized),
        "real_normalized_q01_q99": [float(low), float(high)],
        "generated_normalized_fraction_outside_real_q01_q99": float(outside),
        "note": "Generated sample count is the R5 sample-count setting; real statistics use the full saved latent set."
    }
    (out / "latent_distribution_diagnostics.json").write_text(json.dumps(report, indent=2))

    # Save arrays too, so deeper per-token/per-dimension analysis is possible.
    np.savez_compressed(
        out / "latent_distribution_diagnostics.npz",
        real_raw=real_latents.numpy(),
        generated_raw=generated_latents.numpy(),
        real_normalized=real_normalized.numpy(),
        generated_normalized=generated_normalized.numpy(),
    )
    print(" saved latent_distribution_diagnostics.json and .npz", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None,
                   help="CIFAR-10 Parquet root (required unless --latents is supplied)")
    p.add_argument("--latents", type=Path, default=None,
                   help="Optional saved R3 local-state .npy/.pt file shaped [N,16,32]; skips image encoding")
    p.add_argument("--checkpoint", type=Path, default=HERE/"real_cifar10_local_interaction"/"r3_local_interaction_40000.pt")
    p.add_argument("--out", type=Path, default=HERE/"real_cifar10_latent_diffusion")
    p.add_argument("--train-images", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--diffusion-steps", type=int, default=500)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-count", type=int, default=64)
    p.add_argument("--grid-scale", type=int, default=8,
                   help="Nearest-neighbor display enlargement for saved sample grids; does not change model resolution")
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--reencode", action="store_true")
    args = p.parse_args()
    seed_all(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"R5 latent diffusion | device={device} | torch={torch.__version__} | ROCm={getattr(torch.version,'hip',None)}", flush=True)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"R3 checkpoint missing: {args.checkpoint}. Run R3 first or pass its checkpoint path.")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", {})
    decoder = LocalInteractionAutoencoder(
        local_dim=int(arch.get("local_dim",32)),
        interaction_layers=int(arch.get("interaction_layers",1))
    ).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"]); decoder.eval()
    for q in decoder.parameters(): q.requires_grad_(False)

    latent_source = "fresh_r3_encoding"
    if args.latents is not None:
        latent_path = args.latents.expanduser()
        if not latent_path.is_file():
            raise FileNotFoundError(f"Saved latent file not found: {latent_path}")
        print(f"Loading saved R3 local states: {latent_path}", flush=True)
        if latent_path.suffix.lower() == ".npy":
            z = torch.from_numpy(np.load(latent_path, allow_pickle=False))
        elif latent_path.suffix.lower() in {".pt", ".pth"}:
            z = torch.load(latent_path, map_location="cpu", weights_only=False)
            if isinstance(z, dict):
                for key in ("local_states", "latents", "z"):
                    if key in z:
                        z = z[key]
                        break
        else:
            raise ValueError("--latents must be a .npy, .pt, or .pth file")
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        if z.ndim != 3 or tuple(z.shape[1:]) != (16, 32):
            raise ValueError(f"Expected saved local states shaped [N,16,32], got {tuple(z.shape)}")
        if z.shape[0] < args.batch_size:
            raise ValueError(f"Need at least one full batch ({args.batch_size} states); got {z.shape[0]}")
        if not torch.isfinite(z).all():
            raise ValueError("Saved latent file contains NaN or infinite values")
        z = z.float().contiguous()
        latent_source = "saved_r3_local_states"
        print(f"Loaded latent tensor: {tuple(z.shape)}", flush=True)
    else:
        if args.data_root is None:
            p.error("provide --data-root for image encoding, or --latents to use saved local states")
        ds = CIFARParquetDataset(find_parquet(args.data_root))
        pool, _ = fixed_split_indices(len(ds), 10000, args.data_seed)
        subset = make_subset(ds, pool, args.train_images, args.data_seed)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
        cache = args.out/f"r3_latents_{args.train_images}_{args.data_seed}.pt"
        if args.reencode and cache.exists(): cache.unlink()
        z = encode(decoder, loader, device, cache)
    mean = z.mean((0,1), keepdim=True).to(device)
    std = z.std((0,1), keepdim=True).clamp_min(1e-4).to(device)

    # Decoder control: feed it genuine R3 states before testing generated states.
    # This isolates decoder/checkpoint/latent compatibility from diffusion quality.
    control_count = min(args.sample_count, z.shape[0])
    real_latents = z[:control_count].to(device)
    real_reconstructions = decoder.decode_local(real_latents)
    save_grid(
        real_reconstructions,
        args.out / "real_latent_reconstructions.png",
        scale=max(1, args.grid_scale),
    )
    print(
        f" saved real_latent_reconstructions.png ({control_count} genuine R3 states)",
        flush=True,
    )

    zn = ((z-mean.cpu())/std.cpu()).clamp(-8,8)
    latent_loader = DataLoader(TensorDataset(zn), batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
    model = LatentDenoiser(z.shape[1], z.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ab = alpha_bar(args.diffusion_steps, device)
    history=[]; start=time.time()
    nparams=sum(q.numel() for q in model.parameters())
    print(f"Latents={tuple(z.shape)} | denoiser params={nparams:,} | epochs={args.epochs}", flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; seen=0
        for (clean_cpu,) in latent_loader:
            clean=clean_cpu.to(device, non_blocking=True); b=clean.shape[0]
            t=torch.randint(0,args.diffusion_steps,(b,),device=device)
            noise=torch.randn_like(clean); a=ab[t].view(b,1,1)
            loss=F.mse_loss(model(a.sqrt()*clean+(1-a).sqrt()*noise,t),noise)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += loss.detach().item()*b; seen += b
        avg=total/max(seen,1); history.append(avg)
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            elapsed=time.time()-start
            print(f"epoch {epoch}/{args.epochs} | noise MSE={avg:.6f} | {elapsed/60:.1f} min",flush=True)
            torch.save({"denoiser_state_dict":model.state_dict(),"latent_mean":mean.cpu(),
                        "latent_std":std.cpu(),"tokens":z.shape[1],"local_dim":z.shape[2],
                        "diffusion_steps":args.diffusion_steps,"sample_steps":args.sample_steps,
                        "epoch":epoch,"seed":args.seed,"r3_checkpoint":str(args.checkpoint)},
                       args.out/"r5_latent_diffusion.pt")
        if epoch%10==0 or epoch==args.epochs:
            imgs, generated_latents, generated_normalized = sample(
                model,decoder,mean,std,device,args.sample_count,args.sample_steps,
                args.diffusion_steps,args.seed+epoch
            )
            save_grid(imgs,args.out/f"samples_epoch_{epoch:03d}.png", scale=max(1,args.grid_scale))
            print(f" saved samples_epoch_{epoch:03d}.png",flush=True)
            if epoch == args.epochs:
                save_latent_diagnostics(
                    z, generated_latents, generated_normalized, mean, std, args.out
                )
    report={"experiment":"R5_R3_local_latent_diffusion","device":str(device),
            "rocm_version":getattr(torch.version,"hip",None),"latent_source":latent_source,
            "latent_examples":int(z.shape[0]),
            "training_images":(args.train_images if args.latents is None else None),
            "latent_file":(str(args.latents) if args.latents is not None else None),
            "latent_shape":list(z.shape),"denoiser_parameters":nparams,"epochs":args.epochs,
            "diffusion_steps":args.diffusion_steps,"sample_steps":args.sample_steps,
            "final_noise_mse":history[-1],"elapsed_seconds":time.time()-start,
            "r3_checkpoint":str(args.checkpoint),"history":history}
    (args.out/"r5_results.json").write_text(json.dumps(report,indent=2))
    print("R5 COMPLETE\n"+json.dumps({k:v for k,v in report.items() if k!="history"},indent=2),flush=True)

if __name__=="__main__": main()
