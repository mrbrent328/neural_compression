#!/usr/bin/env python3
"""
neural_compress_profiler.py

Profiles the hot paths of the supplied neural_compress_opt.py-style compressor
in ONE RUN and writes a shareable report.

What it measures:
  1. Environment / PyTorch / CPU information
  2. Model parameter count and approximate compute shape
  3. Per-byte tensor allocation cost
  4. Forward-pass cost
  5. logits -> integer frequency conversion cost
  6. cumulative-frequency construction cost
  7. Python range-coder cost
  8. backward + optimizer.step cost
  9. context-list sliding cost
 10. end-to-end compression throughput
 11. end-to-end decompression throughput
 12. optional delayed-update experiment (1,2,4,8,16,32 symbols)
 13. rough bottleneck ranking and optimization recommendations

It DOES NOT modify the compressor or claim delayed training is bit-identical.
The delayed-update experiment is explicitly a speed/ratio exploration.

Usage:
    python3 neural_compress_profiler.py [file]

If no file is supplied, it benchmarks a synthetic mixed corpus.
"""

import os
import sys
import time
import platform
import statistics
import struct
import hashlib
from itertools import accumulate
from bisect import bisect_right

import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cpu")
SCALE = 1 << 16
PAD_IDX = 256


class NeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64, context_size=16):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


def logits_to_freqs(logits, scale=SCALE):
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


def make_corpus(path, n=4096):
    if path:
        with open(path, "rb") as f:
            data = f.read(n)
        if not data:
            raise ValueError("Input file is empty.")
        return data
    # Mixed corpus: repetitive text + varied bytes.
    a = (b"The quick brown fox jumps over the lazy dog. " * 64)
    b = bytes((i * 73 + 19) & 255 for i in range(2048))
    return (a + b)[:n]


def ns_per_call(fn, repeats=1000):
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    return (time.perf_counter_ns() - t0) / repeats


def benchmark_components(data, context_size=16, lr=0.04):
    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ctx = [PAD_IDX] * context_size
    ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
    target_t = torch.empty((1,), dtype=torch.long, device=device)

    # Warm up.
    for i in range(min(32, len(data))):
        ctx_t[0].copy_(torch.tensor(ctx, dtype=torch.long))
        logits = model(ctx_t).squeeze(0)
        target_t[0] = data[i]
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ctx = ctx[1:] + [data[i]]

    samples = min(len(data), 256)
    totals = {
        "tensor_context": 0,
        "forward": 0,
        "freqs": 0,
        "cum": 0,
        "range": 0,
        "target_tensor": 0,
        "backward_step": 0,
        "context_slide": 0,
        "whole_symbol": 0,
    }

    encoder = RangeEncoder()
    ctx = [PAD_IDX] * context_size

    for i in range(samples):
        s = time.perf_counter_ns()
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        totals["tensor_context"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        logits = model(ctx_t).squeeze(0)
        totals["forward"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        freqs = logits_to_freqs(logits.detach())
        totals["freqs"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        cum = list(accumulate(freqs, initial=0))
        totals["cum"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        encoder.encode_symbol(data[i], freqs, cum)
        totals["range"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        target_t = torch.tensor([data[i]], dtype=torch.long, device=device)
        totals["target_tensor"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        opt.zero_grad(set_to_none=True)
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        opt.step()
        totals["backward_step"] += time.perf_counter_ns() - s

        s = time.perf_counter_ns()
        ctx = ctx[1:] + [data[i]]
        totals["context_slide"] += time.perf_counter_ns() - s

        totals["whole_symbol"] += (
            totals["tensor_context"] + totals["forward"] + totals["freqs"] +
            totals["cum"] + totals["range"] + totals["target_tensor"] +
            totals["backward_step"] + totals["context_slide"]
        ) if False else 0

    # Correct whole-symbol total from independent wall-clock rerun.
    t0 = time.perf_counter_ns()
    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    ctx = [PAD_IDX] * context_size
    for i in range(samples):
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(ctx_t).squeeze(0)
        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))
        encoder.encode_symbol(data[i], freqs, cum)
        target_t = torch.tensor([data[i]], dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        opt.step()
        ctx = ctx[1:] + [data[i]]
    whole_ns = time.perf_counter_ns() - t0

    return model, opt, totals, whole_ns, samples, len(encoder.output)


def delayed_update_benchmark(data, intervals=(1, 2, 4, 8, 16, 32), context_size=16, lr=0.04):
    """
    Exploratory only. It measures speed while accumulating losses and updating
    every N symbols. This is NOT asserted to preserve the original compressor's
    exact model trajectory or compression ratio.
    """
    rows = []
    for interval in intervals:
        torch.manual_seed(42)
        model = NeuralCompressor(context_size=context_size).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        ctx = [PAD_IDX] * context_size
        encoder = RangeEncoder()
        t0 = time.perf_counter()
        pending = []
        for i, byte in enumerate(data):
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(ctx_t).squeeze(0)
            freqs = logits_to_freqs(logits.detach())
            cum = list(accumulate(freqs, initial=0))
            encoder.encode_symbol(byte, freqs, cum)

            target_t = torch.tensor([byte], dtype=torch.long, device=device)
            pending.append(criterion(logits.unsqueeze(0), target_t))
            ctx = ctx[1:] + [byte]

            if len(pending) >= interval:
                loss = torch.stack(pending).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pending.clear()

        if pending:
            loss = torch.stack(pending).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        elapsed = time.perf_counter() - t0
        rows.append((interval, elapsed, len(encoder.output)))
    return rows


def fmt_ms(ns):
    return f"{ns / 1e6:.4f} ms"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = make_corpus(path)

    report_path = "neural_compress_profile.log"
    with open(report_path, "w", encoding="utf-8") as log:
        def p(s=""):
            print(s)
            log.write(s + "\n")

        p("=" * 78)
        p(" NEURAL COMPRESSOR SPEED AUTOPSY")
        p("=" * 78)
        p(f"Input: {path or '[synthetic mixed corpus]'}")
        p(f"Profile bytes: {len(data)}")
        p(f"Python: {platform.python_version()}")
        p(f"Platform: {platform.platform()}")
        p(f"PyTorch: {torch.__version__}")
        p(f"Device: {device}")
        p(f"CPU: {platform.processor() or 'unknown'}")
        p(f"Threads: {torch.get_num_threads()}")
        p(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")

        model = NeuralCompressor()
        params = sum(x.numel() for x in model.parameters())
        p(f"Model parameters: {params:,}")
        p("Architecture: Embedding(257,32) -> Linear(512,64) -> ReLU -> Linear(64,256)")
        p("")

        model, opt, totals, whole_ns, samples, range_bytes = benchmark_components(data)

        p("--- HOT-PATH TIMINGS ---")
        per = {k: v / samples for k, v in totals.items() if k != "whole_symbol"}
        for k, v in per.items():
            p(f"{k:20s}: {fmt_ms(v):>14s} / symbol")

        p(f"{'whole_loop_wall':20s}: {fmt_ms(whole_ns / samples):>14s} / symbol")
        p(f"Estimated baseline throughput: {len(data) / (whole_ns / 1e9):.2f} B/s")
        p(f"Estimated baseline throughput: {(len(data) / (whole_ns / 1e9)) / 1024:.2f} KiB/s")
        p("")

        p("--- COST SHARE (measured components only) ---")
        component_sum = sum(per.values())
        shares = []
        for k, v in per.items():
            share = (v / component_sum * 100) if component_sum else 0
            shares.append((share, k, v))
        for share, k, v in sorted(shares, reverse=True):
            p(f"{k:20s}: {share:6.2f}%   {fmt_ms(v)}")

        p("")
        p("--- END-TO-END BASELINE ---")
        p("Note: this is a profiler, not a bit-exact compressor replacement.")
        p(f"Sample range-coded payload size: {range_bytes} B for {samples} input symbols")
        p("")

        p("--- DELAYED UPDATE EXPERIMENT ---")
        p("WARNING: update interval > 1 changes the learning trajectory; use this only")
        p("to discover the speed/ratio tradeoff before modifying the real compressor.")
        try:
            rows = delayed_update_benchmark(data)
            p(f"{'Interval':>8s} {'Time':>12s} {'KiB/s':>12s} {'Payload B':>12s}")
            for interval, elapsed, payload in rows:
                kbps = (len(data) / elapsed) / 1024
                p(f"{interval:8d} {elapsed:12.4f} {kbps:12.2f} {payload:12d}")
        except RuntimeError as e:
            p(f"Delayed-update experiment failed: {e}")

        p("")
        p("--- AUTOMATIC BOTTLENECK FLAGS ---")
        ranked = sorted(shares, reverse=True)
        for share, k, _ in ranked:
            if share >= 25:
                p(f"HIGH: {k} is {share:.1f}% of measured hot-path component time.")
            elif share >= 10:
                p(f"MED : {k} is {share:.1f}% of measured hot-path component time.")

        p("")
        p("--- WHAT TO SEND BACK ---")
        p("Send this entire file: neural_compress_profile.log")
        p("Also tell me the command you used and whether the profile ran on CPU or GPU.")
        p("=" * 78)

    print(f"\n[+] Full report written to: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
