#!/usr/bin/env python3
"""
series4_experiment1_v1.py - S4E1 C1 Distributed Independent States

Changes vs original:
  1. Single forward pass per symbol (detach logits for coding freqs, reuse the
     SAME graph for the training backward pass) instead of two forward passes.
  2. Plain SGD instead of Adam (Adam's per-param moment bookkeeping was ~1.1s
     of ~2.7s encode time on a 2000-byte file; SGD has near-zero optimizer overhead).
  3. Cumulative-frequency prefix sums computed with itertools.accumulate once
     per symbol instead of repeated sum(freqs[:symbol]) slicing (encoder) and
     a linear scan (decoder) -> both become O(256) once instead of O(256) *
     repeated slicing/allocation.
  4. Preallocated rolling context buffer (list, in-place slide) instead of
     rebuilding a fresh Python list each iteration.

Default lr=0.04 for SGD was chosen by a small sweep (0.005-0.1) across a
repetitive-text sample and a mixed text+noise sample -- it's a reasonable
middle ground, not exhaustively tuned. Adam (original) still wins on ratio
for highly predictable data; SGD wins on speed. Swap the optimizer back to
Adam in _make_model_and_opt() if ratio matters more than throughput for
your data -- the single-forward-pass fix helps either way.

Determinism note: still relies on identical PyTorch/BLAS build across machines.
"""

import sys
import os
import time
import struct
import hashlib
import json
from itertools import accumulate
from bisect import bisect_right
import torch
import torch.nn as nn

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)

device = torch.device("cpu")
SCALE = 1 << 16

EXPERIMENT = "S4E1"
VARIANT = "v1"
NUM_STATES = 4


class NeuralCompressor(nn.Module):
    """
    S4 Experiment 1 / C1 diagnostic v2.

    Four independent predictor states receive the same context.
    Their logits are averaged for coding exactly as in v1.

    v2 adds diagnostics only:
      - per-state prediction entropy
      - ensemble entropy
      - top-1 agreement
      - pairwise Jensen-Shannon divergence
      - pairwise cosine similarity of probability vectors

    These diagnostics do NOT participate in training or coding.
    """

    def __init__(
        self,
        vocab_size=256,
        embed_dim=32,
        hidden_dim=64,
        context_size=16,
        num_states=4,
    ):
        super().__init__()
        self.context_size = context_size
        self.num_states = num_states

        self.states = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(vocab_size + 1, embed_dim, padding_idx=256),
                nn.Flatten(start_dim=1),
                nn.Linear(context_size * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            )
            for _ in range(num_states)
        ])

    def forward_all(self, x):
        return torch.stack([state(x) for state in self.states], dim=0)

    def forward(self, x):
        return self.forward_all(x).mean(dim=0)



def diagnostic_from_logits(all_logits):
    """
    Compute diagnostics from [states, batch=1, vocab] logits.
    Returns scalar metrics. These are detached and never affect gradients.
    """
    with torch.no_grad():
        logits = all_logits[:, 0, :]
        probs = torch.softmax(torch.clamp(logits, -50.0, 50.0), dim=-1)

        eps = 1e-12
        entropy = -(probs * torch.log2(probs.clamp_min(eps))).sum(dim=1)
        ensemble = probs.mean(dim=0)
        ensemble_entropy = -(ensemble * torch.log2(ensemble.clamp_min(eps))).sum()

        top = probs.argmax(dim=1)
        agreement = (top.unsqueeze(0) == top.unsqueeze(1)).float()
        # Mean off-diagonal agreement.
        n = probs.shape[0]
        top1_agreement = (agreement.sum() - n) / (n * (n - 1))

        js_values = []
        cos_values = []
        for i in range(n):
            for j in range(i + 1, n):
                p = probs[i]
                q = probs[j]
                m = 0.5 * (p + q)
                kl_pm = (p * (torch.log2(p.clamp_min(eps)) -
                              torch.log2(m.clamp_min(eps)))).sum()
                kl_qm = (q * (torch.log2(q.clamp_min(eps)) -
                              torch.log2(m.clamp_min(eps)))).sum()
                js_values.append(0.5 * (kl_pm + kl_qm))

                cos = torch.dot(p, q) / (
                    torch.linalg.vector_norm(p) *
                    torch.linalg.vector_norm(q) + eps
                )
                cos_values.append(cos)

        return {
            "state_entropy_mean": entropy.mean().item(),
            "state_entropy_min": entropy.min().item(),
            "state_entropy_max": entropy.max().item(),
            "ensemble_entropy": ensemble_entropy.item(),
            "top1_agreement": top1_agreement.item(),
            "pairwise_js_mean": torch.stack(js_values).mean().item(),
            "pairwise_cosine_mean": torch.stack(cos_values).mean().item(),
        }


def logits_to_freqs(logits, scale=SCALE):
    """logits: 1D tensor (already detached). Returns list[int] freqs."""
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    probs = torch.softmax(logits, dim=-1)
    freqs = (probs * (scale - 256)).int() + 1
    return freqs.tolist()


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.output = bytearray()

    def encode_symbol(self, symbol, freqs, cum):
        total = cum[-1]
        cum_freq = cum[symbol]
        sym_freq = freqs[symbol]

        self.range //= total
        self.low += cum_freq * self.range
        self.range *= sym_freq

        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x00010000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = (-self.low & 0x0000FFFF) << 8
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.output.append(self.low >> 24)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.output)


class RangeDecoder:
    def __init__(self, stream):
        self.stream = stream
        self.ptr = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
            self.code = (self.code << 8) | b
            self.ptr += 1

    def decode_symbol(self, freqs, cum):
        total = cum[-1]
        count = (self.code - self.low) // (self.range // total)

        # cum[i] = sum(freqs[:i]) (exclusive prefix); find first i with cum[i] <= count < cum[i+1]
        symbol = bisect_right(cum, count) - 1

        sym_freq = freqs[symbol]
        cum_freq = cum[symbol]

        self.range //= total
        self.low += cum_freq * self.range
        self.range *= sym_freq

        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x00010000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1
            else:
                self.range = (-self.low & 0x0000FFFF) << 8
                self.low = (self.low << 8) & 0xFFFFFFFF
                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1

        return symbol


def _make_model_and_opt(context_size, lr, num_states=4):
    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size, num_states=num_states).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def new_diag_accumulator(num_states=4, chunk_size=500):
    return {
        "count": 0,
        "chunk_size": chunk_size,
        "chunk_start": 0,
        "chunk_metrics": [],
        "state_entropy_sum": [0.0] * num_states,
        "state_pair_top1_agreement": [[0] * num_states for _ in range(num_states)],
        "js_sum": 0.0,
        "cos_sum": 0.0,
        "chunk_count": 0,
    }


def _chunk_snapshot(acc, n_states):
    n = acc["chunk_count"]
    if n == 0:
        return None

    ent = [x / n for x in acc["state_entropy_sum"]]
    agree = [
        [x / n for x in row]
        for row in acc["state_pair_top1_agreement"]
    ]

    return {
        "start": acc["chunk_start"],
        "end": acc["chunk_start"] + n,
        "samples": n,
        "state_entropy": ent,
        "top1_agreement": agree,
        "pairwise_js": acc["js_sum"] / n,
        "pairwise_cosine": acc["cos_sum"] / n,
    }


def _reset_chunk(acc, next_start):
    n_states = len(acc["state_entropy_sum"])
    acc["chunk_start"] = next_start
    acc["chunk_count"] = 0
    acc["state_entropy_sum"] = [0.0] * n_states
    acc["state_pair_top1_agreement"] = [[0] * n_states for _ in range(n_states)]
    acc["js_sum"] = 0.0
    acc["cos_sum"] = 0.0


def update_diag(acc, all_logits):
    with torch.no_grad():
        logits = all_logits[:, 0, :]
        probs = torch.softmax(torch.clamp(logits, -50.0, 50.0), dim=-1)
        entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
        tops = probs.argmax(dim=1).tolist()

        acc["count"] += 1
        acc["chunk_count"] += 1

        for i, e in enumerate(entropy.tolist()):
            acc["state_entropy_sum"][i] += e

        for i in range(len(tops)):
            for j in range(len(tops)):
                if tops[i] == tops[j]:
                    acc["state_pair_top1_agreement"][i][j] += 1

        d = diagnostic_from_logits(all_logits)
        acc["js_sum"] += d["pairwise_js_mean"]
        acc["cos_sum"] += d["pairwise_cosine_mean"]

        if acc["chunk_count"] >= acc["chunk_size"]:
            snap = _chunk_snapshot(acc, len(tops))
            acc["chunk_metrics"].append(snap)
            _reset_chunk(acc, acc["count"])


def finalize_diag(acc):
    if acc["chunk_count"]:
        snap = _chunk_snapshot(acc, len(acc["state_entropy_sum"]))
        acc["chunk_metrics"].append(snap)

    return {
        "samples": acc["count"],
        "chunk_size": acc["chunk_size"],
        "chunks": acc["chunk_metrics"],
    }



def compress_file(input_path, output_bin_path, context_size=16, lr=0.04, num_states=4, diag_path=None):
    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)
    print(f"[*] Encoding '{input_path}' ({orig_len} bytes)...")
    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(context_size, lr, num_states)
    encoder = RangeEncoder()
    diag_acc = new_diag_accumulator(num_states, chunk_size=500) if diag_path else None
    pad_idx = 256
    model.train()

    ctx = [pad_idx] * context_size  # rolling buffer, slid in place

    for i in range(orig_len):
        target_byte = data[i]

        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        all_logits = model.forward_all(ctx_t)
        logits = all_logits.mean(dim=0).squeeze(0)  # exact v1 combined prediction

        if diag_acc is not None:
            update_diag(diag_acc, all_logits)

        freqs = logits_to_freqs(logits.detach())   # detach only for coding -> no extra forward
        cum = list(accumulate(freqs, initial=0))    # prefix sums, computed once

        encoder.encode_symbol(target_byte, freqs, cum)

        target_t = torch.tensor([target_byte], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = criterion(logits.unsqueeze(0), target_t)  # reuse the SAME logits/graph
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [target_byte]  # slide window

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    header = struct.pack(">I B B", orig_len, 1, num_states)  # S4E1 version=1, state count
    with open(output_bin_path, "wb") as f:
        f.write(header + bitstream)

    final_size = len(header) + len(bitstream)
    ratio = (1.0 - (final_size / orig_len)) * 100 if orig_len > 0 else 0
    print(f"[+] Compressed output saved to '{output_bin_path}'")
    print(f"    Raw Size: {orig_len} B | Binary Size: {final_size} B | Reduction: {ratio:.2f}%")
    print(f"    Elapsed: {elapsed:.2f}s ({orig_len/1024/elapsed:.2f} KB/s)")
    if diag_acc is not None:
        diag = finalize_diag(diag_acc)
        diag["graph_definition"] = {
            "nodes": [f"S{i+1}" for i in range(num_states)],
            "edge_metric": "top1_agreement",
            "edge_meaning": "fraction of samples where two states chose the same top-1 predicted byte",
            "secondary_metrics": ["pairwise_js", "pairwise_cos"],
            "chunk_size": 500,
        }
        with open(diag_path, "w") as df:
            json.dump(diag, df, indent=2)
        print(f"    Diagnostics saved to '{diag_path}'")


def decompress_file(input_bin_path, output_path, context_size=16, lr=0.04, num_states=4, diag_path=None):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    orig_len, version, num_states = struct.unpack(">I B B", raw_bin[:6])
    if version != 1:
        raise ValueError(f"Unsupported S4E1 version: {version}")
    bitstream = raw_bin[6:]

    print(f"[*] Decompressing '{input_bin_path}' ({len(raw_bin)} bytes)...")
    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(context_size, lr, num_states)
    decoder = RangeDecoder(bitstream)
    diag_acc = new_diag_accumulator(num_states, chunk_size=500) if diag_path else None
    pad_idx = 256
    model.train()
    decoded_bytes = bytearray()

    ctx = [pad_idx] * context_size

    for i in range(orig_len):
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        all_logits = model.forward_all(ctx_t)
        logits = all_logits.mean(dim=0).squeeze(0)

        if diag_acc is not None:
            update_diag(diag_acc, all_logits)

        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))

        symbol = decoder.decode_symbol(freqs, cum)
        decoded_bytes.append(symbol)

        target_t = torch.tensor([symbol], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [symbol]

    elapsed = time.time() - t0
    with open(output_path, "wb") as f:
        f.write(decoded_bytes)

    print(f"[+] Decompressed output written to '{output_path}' ({len(decoded_bytes)} bytes)")
    print(f"    Elapsed: {elapsed:.2f}s")
    if diag_acc is not None:
        diag = finalize_diag(diag_acc)
        with open(diag_path, "w") as df:
            json.dump(diag, df, indent=2)
        print(f"    Diagnostics saved to '{diag_path}'")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "compress":
        input_file = sys.argv[2]
        output_bin = sys.argv[3] if len(sys.argv) > 3 else input_file + ".bin"
        compress_file(input_file, output_bin)

    elif cmd == "decompress":
        input_bin = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "reconstructed_file"
        decompress_file(input_bin, output_file)

    elif cmd == "test":
        input_file = sys.argv[2]
        bin_file = input_file + ".tmp.bin"
        recon_file = input_file + ".tmp.recon"

        diag_file = input_file + ".s4e1_v2_diag.json"
        compress_file(input_file, bin_file, num_states=4, diag_path=diag_file)
        print("-" * 65)
        decompress_file(bin_file, recon_file, num_states=4)

        with open(input_file, "rb") as f:
            orig_hash = hashlib.sha256(f.read()).hexdigest()
        with open(recon_file, "rb") as f:
            recon_hash = hashlib.sha256(f.read()).hexdigest()

        print("-" * 65)
        print(f"Original SHA-256     : {orig_hash}")
        print(f"Reconstructed SHA-256: {recon_hash}")

        assert orig_hash == recon_hash, "CRITICAL ERROR: Reconstruction mismatch!"
        print("RESULT: EXACT BIT-EXACT MATCH (PASS)")

        if os.path.exists(bin_file):
            os.remove(bin_file)
        if os.path.exists(recon_file):
            os.remove(recon_file)

    else:
        print(f"Unknown command '{cmd}'")
        print(__doc__)


if __name__ == "__main__":
    main()
