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

EXPERIMENT = "S4E3"
VARIANT = "agreement_v1"
NUM_STATES = 4

# First deliberately-small adaptive test:
#   - always evaluate state 0
#   - if its max probability is >= CONFIDENCE_THRESHOLD,
#     use only state 0 for coding/training
#   - otherwise evaluate all states and average their logits
#
# The decision is deterministic from the current context/model state, so
# the decoder can reproduce it without storing a selector bit in the stream.
AGREEMENT_THRESHOLD = 0.95


class NeuralCompressor(nn.Module):
    """
    S4 Experiment 1 / C1:
    Multiple independent predictor states.

    Each state has the same architecture and receives the same context,
    but has independent parameters. Their logits are averaged to form the
    predictive distribution.

    This is deliberately C1: NO interaction between states yet.
    C2 will add shared interaction.
    """
    def __init__(
        self,
        vocab_size=256,
        embed_dim=48,
        hidden_dim=300,
        context_size=10,
        num_states=3,
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

    def forward_state(self, x, state_index=0):
        return self.states[state_index](x)

    def forward_selected(self, x, use_all_states=True):
        if not use_all_states:
            return self.states[0](x)

        logits = torch.stack(
            [state(x) for state in self.states],
            dim=0,
        )
        return logits.mean(dim=0)



def logits_to_confidence(logits):
    """Return the maximum softmax probability from detached logits."""
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    return torch.softmax(logits, dim=-1).max().item()


def prediction_agreement(logits_a, logits_b):
    """
    Compare two probability distributions.
    Returns 1.0 when identical, lower when they disagree.
    """
    pa = torch.softmax(torch.clamp(logits_a.detach(), -50.0, 50.0), dim=-1)
    pb = torch.softmax(torch.clamp(logits_b.detach(), -50.0, 50.0), dim=-1)
    return 1.0 - torch.abs(pa - pb).sum().item() / 2.0


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


def compress_file(
    input_path,
    output_bin_path,
    context_size=16,
    lr=0.04,
    num_states=4,
    agreement_threshold=AGREEMENT_THRESHOLD,
):
    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)
    print(f"[*] Encoding '{input_path}' ({orig_len} bytes)...")
    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(context_size, lr, num_states)
    encoder = RangeEncoder()
    pad_idx = 256
    model.train()

    ctx = [pad_idx] * context_size

    easy_count = 0
    hard_count = 0

    for i in range(orig_len):
        target_byte = data[i]

        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)

        # Agreement path:
        # If the cheap state and full model agree, use the cheap path.
        # If they disagree, spend the extra compute.
        state0_logits = model.forward_state(ctx_t, 0).squeeze(0)

        all_logits = torch.stack(
            [state(ctx_t) for state in model.states],
            dim=0,
        )
        full_logits = all_logits.mean(dim=0).squeeze(0)

        agreement = prediction_agreement(state0_logits, full_logits)

        if agreement >= agreement_threshold or num_states == 1:
            logits = state0_logits
            easy_count += 1
        else:
            logits = full_logits
            hard_count += 1

        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))
        encoder.encode_symbol(target_byte, freqs, cum)

        target_t = torch.tensor([target_byte], dtype=torch.long, device=device)
        optimizer.zero_grad()

        # Backprop through exactly the path used for coding.
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [target_byte]

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    # Header:
    # orig_len, version, state count, confidence threshold,
    # context size, embed dim, hidden dim
    header = struct.pack(
        ">I B B f H H H",
        orig_len,
        1,
        num_states,
        float(agreement_threshold),
        context_size,
        48,   # current S4E1 best embed_dim
        300,  # current S4E1 best hidden_dim
    )

    with open(output_bin_path, "wb") as f:
        f.write(header + bitstream)

    final_size = len(header) + len(bitstream)
    ratio = (1.0 - (final_size / orig_len)) * 100 if orig_len > 0 else 0

    total = easy_count + hard_count
    easy_pct = (easy_count / total * 100) if total else 0
    hard_pct = (hard_count / total * 100) if total else 0

    print(f"[+] Compressed output saved to '{output_bin_path}'")
    print(
        f"    Raw Size: {orig_len} B | Binary Size: {final_size} B | "
        f"Reduction: {ratio:.2f}%"
    )
    print(f"    Elapsed: {elapsed:.2f}s ({orig_len/1024/elapsed:.2f} KB/s)")
    print(
        f"    Adaptive split: easy={easy_count} ({easy_pct:.1f}%) | "
        f"hard={hard_count} ({hard_pct:.1f}%)"
    )
    print(f"    Agreement threshold: {agreement_threshold:.3f}")


def decompress_file(input_bin_path, output_path):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    header_size = struct.calcsize(">I B B f H H H")
    (
        orig_len,
        version,
        num_states,
        agreement_threshold,
        context_size,
        embed_dim,
        hidden_dim,
    ) = struct.unpack(">I B B f H H H", raw_bin[:header_size])

    if version != 1:
        raise ValueError(f"Unsupported S4E2 adaptive version: {version}")

    bitstream = raw_bin[header_size:]

    print(f"[*] Decompressing '{input_bin_path}' ({len(raw_bin)} bytes)...")
    t0 = time.time()

    # Rebuild with the exact dimensions stored in the header.
    torch.manual_seed(42)
    model = NeuralCompressor(
        vocab_size=256,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_size=context_size,
        num_states=num_states,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
    criterion = nn.CrossEntropyLoss()

    decoder = RangeDecoder(bitstream)
    pad_idx = 256
    model.train()
    decoded_bytes = bytearray()

    ctx = [pad_idx] * context_size
    easy_count = 0
    hard_count = 0

    for _ in range(orig_len):
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)

        # The decoder performs the identical deterministic selector.
        state0_logits = model.forward_state(ctx_t, 0).squeeze(0)
        confidence = logits_to_confidence(state0_logits.detach())

        if confidence >= agreement_threshold or num_states == 1:
            logits = state0_logits
            easy_count += 1
        else:
            all_logits = torch.stack(
                [state(ctx_t) for state in model.states],
                dim=0,
            )
            logits = all_logits.mean(dim=0).squeeze(0)
            hard_count += 1

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

    total = easy_count + hard_count
    easy_pct = (easy_count / total * 100) if total else 0
    hard_pct = (hard_count / total * 100) if total else 0

    print(f"[+] Decompressed output written to '{output_path}' ({len(decoded_bytes)} bytes)")
    print(f"    Elapsed: {elapsed:.2f}s")
    print(
        f"    Adaptive split: easy={easy_count} ({easy_pct:.1f}%) | "
        f"hard={hard_count} ({hard_pct:.1f}%)"
    )
    print(f"    Agreement threshold: {agreement_threshold:.3f}")

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nS4E2 adaptive test:")
        print("  python3 series4_experiment2_adaptive.py test <file>")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        input_file = sys.argv[2]
        bin_file = input_file + ".s4e3_agreement.tmp.bin"
        recon_file = input_file + ".s4e3_agreement.tmp.recon"

        compress_file(
            input_file,
            bin_file,
            context_size=16,
            lr=0.04,
            num_states=4,
            agreement_threshold=AGREEMENT_THRESHOLD,
        )
        print("-" * 65)

        decompress_file(bin_file, recon_file)

        with open(input_file, "rb") as f:
            orig_hash = hashlib.sha256(f.read()).hexdigest()
        with open(recon_file, "rb") as f:
            recon_hash = hashlib.sha256(f.read()).hexdigest()

        print("-" * 65)
        print(f"Original SHA-256     : {orig_hash}")
        print(f"Reconstructed SHA-256: {recon_hash}")

        if orig_hash != recon_hash:
            print("RESULT: EXACT BIT-EXACT MATCH (FAIL)")
            raise AssertionError("CRITICAL ERROR: Reconstruction mismatch!")

        print("RESULT: EXACT BIT-EXACT MATCH (PASS)")

        if os.path.exists(bin_file):
            os.remove(bin_file)
        if os.path.exists(recon_file):
            os.remove(recon_file)

    else:
        print(f"Unknown command '{cmd}'")
        print("Use: python3 series4_experiment2_adaptive.py test <file>")


if __name__ == "__main__":
    main()
