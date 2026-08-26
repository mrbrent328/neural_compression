#!/usr/bin/env python3
import os
import math
import time
import zlib
import zipfile
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. BATCH-AWARE NEURAL MODEL
# =============================================================================
class BatchedNeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64, context_size=16):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=256)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


# =============================================================================
# 2. FAST MINI-BATCHED ONLINE ENGINE
# =============================================================================
def compress_online_batched(data: bytes, context_size=16, batch_size=256, lr=0.002):
    n_bytes = len(data)
    if n_bytes <= context_size:
        return n_bytes

    pad_idx = 256
    model = BatchedNeuralCompressor(context_size=context_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_prefix = [pad_idx] * context_size
    raw_data_padded = torch.tensor(pad_prefix + list(data), dtype=torch.long, device=device)

    total_bits = 0.0
    model.train()

    for ptr in range(0, n_bytes, batch_size):
        current_batch_size = min(batch_size, n_bytes - ptr)

        ctx_indices = torch.arange(ptr, ptr + current_batch_size, device=device).unsqueeze(1) + torch.arange(context_size, device=device)
        contexts = raw_data_padded[ctx_indices]
        targets = raw_data_padded[context_size + ptr : context_size + ptr + current_batch_size]

        optimizer.zero_grad()
        logits = model(contexts)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        bits = -target_log_probs / math.log(2)
        total_bits += bits.sum().item()

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    return int(math.ceil(total_bits / 8.0))


# =============================================================================
# 3. REAL-WORLD DATASET DOWNLOADER
# =============================================================================
def get_realworld_datasets() -> dict[str, bytes]:
    datasets = {}
    data_dir = "./benchmark_data"
    os.makedirs(data_dir, exist_ok=True)

    # 1. Real Wikipedia Slice (1MB enwik8)
    enwik_zip = os.path.join(data_dir, "enwik8.zip")
    enwik_raw = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(enwik_raw):
        print("Downloading enwik8 dataset...")
        url = "http://mattmahoney.net/dc/enwik8.zip"
        try:
            urllib.request.urlretrieve(url, enwik_zip)
            with zipfile.ZipFile(enwik_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        except Exception as e:
            print(f"Failed to download enwik8 ({e}). Using local fallback text.")
            with open(enwik_raw, "wb") as f:
                f.write(b"Anarchism is a political philosophy and movement that is skeptical of authority...\n" * 15000)

    if os.path.exists(enwik_raw):
        with open(enwik_raw, "rb") as f:
            # Take a 1MB slice for quick evaluation
            datasets["enwik8_1mb.txt"] = f.read(1024 * 1024)

    # 2. Real Python Source Code (requests library file)
    requests_path = os.path.join(data_dir, "requests_sessions.py")
    if not os.path.exists(requests_path):
        print("Downloading real Python source code (requests/sessions.py)...")
        url = "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py"
        try:
            urllib.request.urlretrieve(url, requests_path)
        except Exception as e:
            print(f"Failed to fetch requests source ({e}).")

    if os.path.exists(requests_path):
        with open(requests_path, "rb") as f:
            datasets["requests_sessions.py"] = f.read()

    return datasets


def main():
    print("=" * 105)
    print("      REAL-WORLD DATASET BENCHMARK: ONLINE NEURAL COMPRESSION vs GZIP vs ZSTD")
    print("=" * 105)

    samples = get_realworld_datasets()

    header = f"{'File Name':<22} | {'Orig Size':<11} | {'GZIP':<8} | {'ZSTD':<8} | {'Neural (B)':<10} | {'Ratio':<8} | {'Speed'}"
    print(header)
    print("-" * 105)

    for name, data in samples.items():
        orig_sz = len(data)

        gz_sz = len(zlib.compress(data, level=9))
        zstd_sz = len(zstd.ZstdCompressor(level=10).compress(data)) if HAS_ZSTD else 0

        t0 = time.time()
        neural_sz = compress_online_batched(data, context_size=16, batch_size=256)
        elapsed = time.time() - t0

        ratio = (1.0 - (neural_sz / orig_sz)) * 100
        throughput = (orig_sz / 1024) / elapsed  # KB/s

        print(
            f"{name:<22} | {orig_sz:<11d} | {gz_sz:<8d} | {zstd_sz:<8d} | {neural_sz:<10d} | {ratio:>7.2f}% | {throughput:>8.1f} KB/s"
        )

    print("=" * 105)

if __name__ == "__main__":
    main()