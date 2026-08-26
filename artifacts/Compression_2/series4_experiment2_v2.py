#!/usr/bin/env python3
"""
series4_experiment2_v2.py - S4E2 Local Representations + Shared Interaction

Purpose
-------
S4E1 showed four independent predictor states receiving the same context.
S4E2 keeps the four local states, but gives them a shared communication
layer between local representation and reconstruction.

Architecture
------------
same context
    |
    +--> local state S1 encoder --\
    +--> local state S2 encoder ----\
    +--> local state S3 encoder -----+--> SHARED INTERACTION --> local decoders
    +--> local state S4 encoder ----/             |
                                                   +--> logits S1..S4
                                                   |
                                              mean logits -> coder

Important:
- v2 synchronization fix: encoder and decoder always execute the same
  MultiheadAttention path by requesting attention weights on both sides.
- Local encoders and local decoders remain state-specific.
- The interaction machinery is shared.
- No manually specified graph or pairwise connections.
- No state identity/position embeddings are added.
- Diagnostics observe the structure before and after interaction.
- Diagnostics never participate in coding or optimization.
- Coding remains synchronized by using the mean of the four post-interaction
  state predictions, matching the S4E1 ensemble rule.

The interaction layer is deliberately small:
    shared Q/K/V attention across the four local states
    + shared feed-forward transform
    + residual connections
    + LayerNorm

The attention weights are also recorded as an emergent communication graph.
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

EXPERIMENT = "S4E2"
VARIANT = "local_repr_shared_interaction_v2"
NUM_STATES = 4
LATENT_DIM = 64
INTERACTION_FF_DIM = 128
INTERACTION_HEADS = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SharedInteraction(nn.Module):
    """
    Shared communication machinery.

    Input shape: [batch, states, latent]
    Output shape: [batch, states, latent]

    No state-specific parameters and no positional/state identity embeddings.
    """

    def __init__(self, dim=LATENT_DIM, ff_dim=INTERACTION_FF_DIM,
                 heads=INTERACTION_HEADS):
        super().__init__()

        if dim % heads != 0:
            raise ValueError("latent dimension must divide attention heads")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, need_weights=False):
        y = self.norm1(x)
        attn_out, weights = self.attn(
            y, y, y,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + attn_out

        y = self.norm2(x)
        x = x + self.ff(y)

        return x, weights


class NeuralCompressorS4E2(nn.Module):
    """
    Four local predictor states + one shared interaction layer.

    Each state owns its encoder and decoder. The communication machinery
    between them is shared.
    """

    def __init__(
        self,
        vocab_size=256,
        embed_dim=32,
        latent_dim=LATENT_DIM,
        interaction_ff_dim=INTERACTION_FF_DIM,
        interaction_heads=INTERACTION_HEADS,
        context_size=16,
        num_states=NUM_STATES,
    ):
        super().__init__()

        self.context_size = context_size
        self.num_states = num_states
        self.latent_dim = latent_dim

        self.local_states = nn.ModuleList([
            nn.ModuleDict({
                "embedding": nn.Embedding(
                    vocab_size + 1,
                    embed_dim,
                    padding_idx=256,
                ),
                "encoder": nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(context_size * embed_dim, latent_dim),
                    nn.ReLU(),
                ),
                "decoder": nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, vocab_size),
                ),
            })
            for _ in range(num_states)
        ])

        self.interaction = SharedInteraction(
            dim=latent_dim,
            ff_dim=interaction_ff_dim,
            heads=interaction_heads,
        )

    def encode_local(self, x):
        local = []

        for state in self.local_states:
            emb = state["embedding"](x)
            z = state["encoder"](emb)
            local.append(z)

        # [batch, states, latent]
        return torch.stack(local, dim=1)

    def decode_local(self, interacted):
        logits = []

        for i, state in enumerate(self.local_states):
            logits.append(
                state["decoder"](interacted[:, i, :])
            )

        # [states, batch, vocab] to preserve S4E1 diagnostic convention
        return torch.stack(logits, dim=0)

    def forward_all(self, x, return_latents=False, return_attention=False):
        local = self.encode_local(x)

        interacted, attention = self.interaction(
            local,
            need_weights=return_attention,
        )

        logits = self.decode_local(interacted)

        if return_latents or return_attention:
            return logits, local, interacted, attention

        return logits

    def forward(self, x):
        return self.forward_all(x).mean(dim=0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _probs_from_logits(logits):
    return torch.softmax(
        torch.clamp(logits, -50.0, 50.0),
        dim=-1,
    )


def distribution_diagnostics(all_logits):
    """
    Same prediction-space diagnostics used in S4E1.
    all_logits: [states, batch=1, vocab]
    """
    with torch.no_grad():
        probs = _probs_from_logits(all_logits[:, 0, :])
        n = probs.shape[0]

        entropy = -(
            probs * torch.log2(probs.clamp_min(1e-12))
        ).sum(dim=1)

        ensemble = probs.mean(dim=0)
        ensemble_entropy = -(
            ensemble * torch.log2(ensemble.clamp_min(1e-12))
        ).sum()

        tops = probs.argmax(dim=1)
        agreement = (
            tops.unsqueeze(0) == tops.unsqueeze(1)
        ).float()

        top1_agreement = (
            agreement.sum() - n
        ) / (n * (n - 1))

        js_values = []
        cos_values = []

        for i in range(n):
            for j in range(i + 1, n):
                p = probs[i]
                q = probs[j]
                m = 0.5 * (p + q)

                kl_pm = (
                    p * (
                        torch.log2(p.clamp_min(1e-12))
                        - torch.log2(m.clamp_min(1e-12))
                    )
                ).sum()

                kl_qm = (
                    q * (
                        torch.log2(q.clamp_min(1e-12))
                        - torch.log2(m.clamp_min(1e-12))
                    )
                ).sum()

                js_values.append(0.5 * (kl_pm + kl_qm))

                cos = torch.dot(p, q) / (
                    torch.linalg.vector_norm(p)
                    * torch.linalg.vector_norm(q)
                    + 1e-12
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


def latent_geometry(local, interacted):
    """
    Observe whether the shared layer changes the relationships between states.

    local/interacted: [batch, states, latent]
    """
    with torch.no_grad():
        before = local[0]
        after = interacted[0]

        def cosine_matrix(x):
            x = x / (
                torch.linalg.vector_norm(x, dim=1, keepdim=True)
                + 1e-12
            )
            return x @ x.T

        before_cos = cosine_matrix(before)
        after_cos = cosine_matrix(after)

        delta = after_cos - before_cos

        return {
            "pre_interaction_cosine": before_cos.tolist(),
            "post_interaction_cosine": after_cos.tolist(),
            "cosine_delta": delta.tolist(),
            "mean_offdiag_pre": (
                before_cos.sum() - before_cos.shape[0]
            ).item() / (
                before_cos.shape[0] * (before_cos.shape[0] - 1)
            ),
            "mean_offdiag_post": (
                after_cos.sum() - after_cos.shape[0]
            ).item() / (
                after_cos.shape[0] * (after_cos.shape[0] - 1)
            ),
        }


def attention_graph(attention):
    """
    Convert MultiheadAttention weights into an emergent state-to-state graph.

    Returned matrix is [source_state, target_state] after averaging heads.
    Rows are normalized attention distributions.
    """
    if attention is None:
        return None

    with torch.no_grad():
        # PyTorch MHA with average_attn_weights=False:
        # [batch, heads, target_query, source_key]
        graph = attention.mean(dim=1)[0]

        return graph.tolist()


# ---------------------------------------------------------------------------
# Quantization + range coding
# ---------------------------------------------------------------------------

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

        while (
            (self.low ^ (self.low + self.range)) < 0x01000000
            or self.range < 0x00010000
        ):
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
            b = (
                self.stream[self.ptr]
                if self.ptr < len(self.stream)
                else 0
            )
            self.code = (self.code << 8) | b
            self.ptr += 1

    def decode_symbol(self, freqs, cum):
        total = cum[-1]
        count = (self.code - self.low) // (
            self.range // total
        )

        symbol = bisect_right(cum, count) - 1

        sym_freq = freqs[symbol]
        cum_freq = cum[symbol]

        self.range //= total
        self.low += cum_freq * self.range
        self.range *= sym_freq

        while (
            (self.low ^ (self.low + self.range)) < 0x01000000
            or self.range < 0x00010000
        ):
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF

                b = (
                    self.stream[self.ptr]
                    if self.ptr < len(self.stream)
                    else 0
                )
                self.code = (
                    (self.code << 8) | b
                ) & 0xFFFFFFFF
                self.ptr += 1
            else:
                self.range = (
                    -self.low & 0x0000FFFF
                ) << 8
                self.low = (self.low << 8) & 0xFFFFFFFF

                b = (
                    self.stream[self.ptr]
                    if self.ptr < len(self.stream)
                    else 0
                )
                self.code = (
                    (self.code << 8) | b
                ) & 0xFFFFFFFF
                self.ptr += 1

        return symbol


# ---------------------------------------------------------------------------
# Diagnostics accumulator
# ---------------------------------------------------------------------------

def new_diag_accumulator(num_states=NUM_STATES, chunk_size=500):
    return {
        "count": 0,
        "chunk_size": chunk_size,
        "chunk_start": 0,
        "chunk_count": 0,
        "chunk_metrics": [],
        "state_entropy_sum": [0.0] * num_states,
        "state_pair_top1_agreement": [
            [0] * num_states for _ in range(num_states)
        ],
        "js_sum": 0.0,
        "cos_sum": 0.0,
        "pre_latent_cos_sum": 0.0,
        "post_latent_cos_sum": 0.0,
        "attention_sum": [
            [0.0] * num_states for _ in range(num_states)
        ],
    }


def _offdiag_mean(matrix):
    n = len(matrix)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                total += matrix[i][j]
                count += 1

    return total / count


def _chunk_snapshot(acc, n_states):
    n = acc["chunk_count"]

    if n == 0:
        return None

    attention = [
        [x / n for x in row]
        for row in acc["attention_sum"]
    ]

    return {
        "start": acc["chunk_start"],
        "end": acc["chunk_start"] + n,
        "samples": n,
        "state_entropy": [
            x / n for x in acc["state_entropy_sum"]
        ],
        "top1_agreement": [
            [x / n for x in row]
            for row in acc["state_pair_top1_agreement"]
        ],
        "pairwise_js": acc["js_sum"] / n,
        "pairwise_cosine": acc["cos_sum"] / n,
        "pre_interaction_latent_cosine": (
            acc["pre_latent_cos_sum"] / n
        ),
        "post_interaction_latent_cosine": (
            acc["post_latent_cos_sum"] / n
        ),
        "interaction_attention": attention,
        "interaction_attention_mean_offdiag": (
            _offdiag_mean(attention)
        ),
    }


def _reset_chunk(acc, next_start):
    n_states = len(acc["state_entropy_sum"])

    acc["chunk_start"] = next_start
    acc["chunk_count"] = 0
    acc["state_entropy_sum"] = [0.0] * n_states
    acc["state_pair_top1_agreement"] = [
        [0] * n_states for _ in range(n_states)
    ]
    acc["js_sum"] = 0.0
    acc["cos_sum"] = 0.0
    acc["pre_latent_cos_sum"] = 0.0
    acc["post_latent_cos_sum"] = 0.0
    acc["attention_sum"] = [
        [0.0] * n_states for _ in range(n_states)
    ]


def update_diag(acc, logits, local, interacted, attention):
    with torch.no_grad():
        probs = _probs_from_logits(logits[:, 0, :])

        entropy = -(
            probs * torch.log2(probs.clamp_min(1e-12))
        ).sum(dim=1)

        tops = probs.argmax(dim=1).tolist()

        acc["count"] += 1
        acc["chunk_count"] += 1

        for i, value in enumerate(entropy.tolist()):
            acc["state_entropy_sum"][i] += value

        for i in range(len(tops)):
            for j in range(len(tops)):
                if tops[i] == tops[j]:
                    acc["state_pair_top1_agreement"][i][j] += 1

        d = distribution_diagnostics(logits)
        acc["js_sum"] += d["pairwise_js_mean"]
        acc["cos_sum"] += d["pairwise_cosine_mean"]

        geometry = latent_geometry(local, interacted)
        acc["pre_latent_cos_sum"] += (
            geometry["mean_offdiag_pre"]
        )
        acc["post_latent_cos_sum"] += (
            geometry["mean_offdiag_post"]
        )

        if attention is not None:
            graph = attention_graph(attention)

            for i in range(len(graph)):
                for j in range(len(graph[i])):
                    acc["attention_sum"][i][j] += graph[i][j]

        if acc["chunk_count"] >= acc["chunk_size"]:
            snap = _chunk_snapshot(acc, len(tops))
            acc["chunk_metrics"].append(snap)
            _reset_chunk(acc, acc["count"])


def finalize_diag(acc, num_states=NUM_STATES):
    if acc["chunk_count"]:
        snap = _chunk_snapshot(acc, num_states)
        acc["chunk_metrics"].append(snap)

    return {
        "experiment": EXPERIMENT,
        "variant": VARIANT,
        "samples": acc["count"],
        "chunk_size": acc["chunk_size"],
        "nodes": [f"S{i + 1}" for i in range(num_states)],
        "graph_definition": {
            "prediction_edge_metric": (
                "fraction of samples where two states chose "
                "the same top-1 predicted byte"
            ),
            "secondary_prediction_metrics": [
                "pairwise_js",
                "pairwise_cosine",
            ],
            "interaction_edge_metric": (
                "shared multi-head attention weight from query "
                "state to source state"
            ),
            "important": (
                "The interaction graph is observed from learned "
                "attention; no graph topology is imposed."
            ),
        },
        "chunks": acc["chunk_metrics"],
    }


# ---------------------------------------------------------------------------
# Training / coding
# ---------------------------------------------------------------------------

def _make_model_and_opt(
    context_size,
    lr,
    num_states=NUM_STATES,
):
    torch.manual_seed(42)

    model = NeuralCompressorS4E2(
        context_size=context_size,
        num_states=num_states,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def compress_file(
    input_path,
    output_bin_path,
    context_size=16,
    lr=0.04,
    num_states=NUM_STATES,
    diag_path=None,
):
    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)

    print(
        f"[*] Encoding '{input_path}' ({orig_len} bytes)..."
    )

    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(
        context_size,
        lr,
        num_states,
    )

    encoder = RangeEncoder()

    diag_acc = (
        new_diag_accumulator(
            num_states,
            chunk_size=500,
        )
        if diag_path
        else None
    )

    pad_idx = 256
    model.train()

    ctx = [pad_idx] * context_size

    for i in range(orig_len):
        target_byte = data[i]

        ctx_t = torch.tensor(
            [ctx],
            dtype=torch.long,
            device=device,
        )

        if diag_acc is not None:
            all_logits, local, interacted, attention = (
                model.forward_all(
                    ctx_t,
                    return_latents=True,
                    return_attention=True,
                )
            )
        else:
            all_logits, local, interacted, attention = (
                model.forward_all(
                    ctx_t,
                    return_latents=False,
                    return_attention=True,
                )
            )

        # Exact ensemble rule: mean post-interaction prediction.
        logits = all_logits.mean(dim=0).squeeze(0)

        if diag_acc is not None:
            update_diag(
                diag_acc,
                all_logits,
                local,
                interacted,
                attention,
            )

        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))

        encoder.encode_symbol(
            target_byte,
            freqs,
            cum,
        )

        target_t = torch.tensor(
            [target_byte],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad()

        loss = criterion(
            logits.unsqueeze(0),
            target_t,
        )

        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [target_byte]

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    # version=2 distinguishes S4E2 from S4E1.
    header = struct.pack(
        ">I B B",
        orig_len,
        2,
        num_states,
    )

    with open(output_bin_path, "wb") as f:
        f.write(header + bitstream)

    final_size = len(header) + len(bitstream)

    ratio = (
        (1.0 - (final_size / orig_len)) * 100
        if orig_len > 0
        else 0
    )

    print(
        f"[+] Compressed output saved to '{output_bin_path}'"
    )
    print(
        f"    Raw Size: {orig_len} B | "
        f"Binary Size: {final_size} B | "
        f"Reduction: {ratio:.2f}%"
    )
    print(
        f"    Elapsed: {elapsed:.2f}s "
        f"({orig_len / 1024 / elapsed:.2f} KB/s)"
    )

    if diag_acc is not None:
        diag = finalize_diag(
            diag_acc,
            num_states,
        )

        with open(diag_path, "w") as df:
            json.dump(diag, df, indent=2)

        print(
            f"    Diagnostics saved to '{diag_path}'"
        )


def decompress_file(
    input_bin_path,
    output_path,
    context_size=16,
    lr=0.04,
    num_states=NUM_STATES,
    diag_path=None,
):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    orig_len, version, stored_states = struct.unpack(
        ">I B B",
        raw_bin[:6],
    )

    if version != 2:
        raise ValueError(
            f"Unsupported S4E2 version: {version}"
        )

    if stored_states != num_states:
        raise ValueError(
            f"State count mismatch: file={stored_states}, "
            f"requested={num_states}"
        )

    bitstream = raw_bin[6:]

    print(
        f"[*] Decompressing '{input_bin_path}' "
        f"({len(raw_bin)} bytes)..."
    )

    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(
        context_size,
        lr,
        num_states,
    )

    decoder = RangeDecoder(bitstream)

    diag_acc = (
        new_diag_accumulator(
            num_states,
            chunk_size=500,
        )
        if diag_path
        else None
    )

    pad_idx = 256
    model.train()

    decoded_bytes = bytearray()
    ctx = [pad_idx] * context_size

    for _ in range(orig_len):
        ctx_t = torch.tensor(
            [ctx],
            dtype=torch.long,
            device=device,
        )

        if diag_acc is not None:
            all_logits, local, interacted, attention = (
                model.forward_all(
                    ctx_t,
                    return_latents=True,
                    return_attention=True,
                )
            )
        else:
            all_logits, local, interacted, attention = (
                model.forward_all(
                    ctx_t,
                    return_latents=False,
                    return_attention=True,
                )
            )

        logits = all_logits.mean(dim=0).squeeze(0)

        if diag_acc is not None:
            update_diag(
                diag_acc,
                all_logits,
                local,
                interacted,
                attention,
            )

        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))

        symbol = decoder.decode_symbol(
            freqs,
            cum,
        )

        decoded_bytes.append(symbol)

        target_t = torch.tensor(
            [symbol],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad()

        loss = criterion(
            logits.unsqueeze(0),
            target_t,
        )

        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [symbol]

    elapsed = time.time() - t0

    with open(output_path, "wb") as f:
        f.write(decoded_bytes)

    print(
        f"[+] Decompressed output written to "
        f"'{output_path}' ({len(decoded_bytes)} bytes)"
    )
    print(
        f"    Elapsed: {elapsed:.2f}s"
    )

    if diag_acc is not None:
        diag = finalize_diag(
            diag_acc,
            num_states,
        )

        with open(diag_path, "w") as df:
            json.dump(diag, df, indent=2)

        print(
            f"    Diagnostics saved to '{diag_path}'"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "compress":
        input_file = sys.argv[2]
        output_bin = (
            sys.argv[3]
            if len(sys.argv) > 3
            else input_file + ".s4e2.bin"
        )

        diag_file = (
            output_bin + ".diag.json"
        )

        compress_file(
            input_file,
            output_bin,
            num_states=NUM_STATES,
            diag_path=diag_file,
        )

    elif cmd == "decompress":
        input_bin = sys.argv[2]
        output_file = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "reconstructed_file"
        )

        diag_file = (
            output_file + ".s4e2.diag.json"
        )

        decompress_file(
            input_bin,
            output_file,
            num_states=NUM_STATES,
            diag_path=diag_file,
        )

    elif cmd == "test":
        input_file = sys.argv[2]

        bin_file = (
            input_file + ".s4e2.tmp.bin"
        )
        recon_file = (
            input_file + ".s4e2.tmp.recon"
        )
        diag_file = (
            input_file + ".s4e2_diag.json"
        )

        compress_file(
            input_file,
            bin_file,
            num_states=NUM_STATES,
            diag_path=diag_file,
        )

        print("-" * 70)

        decompress_file(
            bin_file,
            recon_file,
            num_states=NUM_STATES,
        )

        with open(input_file, "rb") as f:
            orig_hash = hashlib.sha256(
                f.read()
            ).hexdigest()

        with open(recon_file, "rb") as f:
            recon_hash = hashlib.sha256(
                f.read()
            ).hexdigest()

        print("-" * 70)
        print(
            f"Original SHA-256     : {orig_hash}"
        )
        print(
            f"Reconstructed SHA-256: {recon_hash}"
        )

        assert (
            orig_hash == recon_hash
        ), "CRITICAL ERROR: Reconstruction mismatch!"

        print(
            "RESULT: EXACT BIT-EXACT MATCH (PASS)"
        )

        if os.path.exists(bin_file):
            os.remove(bin_file)

        if os.path.exists(recon_file):
            os.remove(recon_file)

    else:
        print(
            f"Unknown command '{cmd}'"
        )
        print(__doc__)


if __name__ == "__main__":
    main()
