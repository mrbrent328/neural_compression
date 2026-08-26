#!/usr/bin/env python3
"""
series4_experiment4_v1.py
S4E4 — Surgical Perturbation Fingerprints

Goal
----
Do NOT tell the four local states what their roles should be.

Instead:
  1. Train the original S4E1-style four independent predictors.
  2. At checkpoints, sample real contexts from the file.
  3. Apply controlled perturbations to those contexts.
  4. Measure how each state changes in:
       - latent representation
       - prediction distribution
       - top prediction
  5. Save state fingerprints and latent geometry.
  6. Produce simple plots so we can literally look for shapes.

The perturbations are deliberately generic byte-level probes:
  identity  = change one byte's identity
  repeat    = replace one byte with a neighboring byte
  order     = swap two positions
  position  = move one byte to another position
  context   = replace a local byte with a byte drawn from elsewhere

No role labels are assigned. The output tells us whether S1/S2/S3/S4
react differently to these dimensions.

S4E4 does NOT change the compressor or compression result. It is an
observation/probing experiment.

Usage
-----
python3 series4_experiment3.py probe musicforchat.txt

Optional:
python3 series4_experiment3.py probe test.png
python3 series4_experiment3.py probe musicforchat.txt --samples 64 --checkpoint 1000

Outputs
-------
<file>.s4e4.json
<file>.s4e4_fingerprint.png
<file>.s4e4_geometry.png

The PNGs are intentionally simple:
  fingerprint = state × perturbation response heatmap
  geometry    = 2-D PCA-like projection implemented from pairwise distances
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

SEED = 42
NUM_STATES = 4
VOCAB_SIZE = 256
PAD = 256
EMBED_DIM = 32
HIDDEN_DIM = 64
CONTEXT_SIZE = 16
DEFAULT_LR = 0.04

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# S4E1 model
# ---------------------------------------------------------------------------

class StatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            VOCAB_SIZE + 1,
            EMBED_DIM,
            padding_idx=PAD,
        )
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(CONTEXT_SIZE * EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

    def forward(self, x):
        return self.net(self.embedding(x))


class S4E1(nn.Module):
    def __init__(self, num_states=NUM_STATES):
        super().__init__()
        self.states = nn.ModuleList(
            [StatePredictor() for _ in range(num_states)]
        )

    def forward_all(self, x):
        return torch.stack(
            [state(x) for state in self.states],
            dim=0,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_model():
    torch.manual_seed(SEED)
    model = S4E1().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def train_until(model, optimizer, criterion, data, start, end, context_size):
    """
    Continue exactly from `start` through `end`.

    This lets us capture checkpoints without changing the learning trajectory.
    """
    model.train()
    ctx = [PAD] * context_size

    # Reconstruct the exact context at the requested starting point.
    # We need the previous context bytes, not just the current chunk.
    if start > 0:
        begin = max(0, start - context_size)
        history = list(data[begin:start])
        ctx = ([PAD] * (context_size - len(history))) + history

    for i in range(start, end):
        target = data[i]

        ctx_t = torch.tensor(
            [ctx],
            dtype=torch.long,
            device=device,
        )

        logits = model.forward_all(ctx_t).mean(dim=0).squeeze(0)

        target_t = torch.tensor(
            [target],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad()
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [target]


def train_checkpoints(data, checkpoints):
    model, optimizer, criterion = make_model()

    states = []
    current = 0

    for checkpoint in checkpoints:
        if checkpoint <= current:
            continue

        train_until(
            model,
            optimizer,
            criterion,
            data,
            current,
            checkpoint,
            CONTEXT_SIZE,
        )

        states.append((checkpoint, model))
        current = checkpoint

    return states


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

def real_contexts(data, count, rng):
    """
    Pick contexts from places where a full 16-byte context exists and a
    following byte exists.
    """
    if len(data) <= CONTEXT_SIZE + 1:
        raise ValueError("File is too small for probing.")

    max_start = len(data) - CONTEXT_SIZE - 1

    starts = sorted(
        rng.sample(
            range(max_start + 1),
            min(count, max_start + 1),
        )
    )

    return [
        (
            start,
            list(data[start:start + CONTEXT_SIZE]),
            data[start + CONTEXT_SIZE],
        )
        for start in starts
    ]


# ---------------------------------------------------------------------------
# Surgical perturbations
# ---------------------------------------------------------------------------

# S4E4 is designed to separate three things that S4E3 mixed together:
#   1. WHAT changed (identity)
#   2. HOW far apart the changed positions are (order/movement distance)
#   3. WHERE the change occurs in the context
#
# We intentionally keep every probe the same length as the original context.
# Nothing is inserted or deleted. This lets us attribute differences more
# cleanly to the operation being tested.


def choose_novel_byte(ctx, rng):
    old_values = set(ctx)
    if len(old_values) >= 256:
        return (ctx[0] + 1) % 256
    new = rng.randrange(256)
    while new in old_values:
        new = rng.randrange(256)
    return new


def make_identity_probe(ctx, rng):
    """Change one symbol, at a random position, to a novel byte."""
    out = list(ctx)
    pos = rng.randrange(len(out))
    out[pos] = choose_novel_byte(ctx, rng)
    return out


def make_identity_pos_probe(pos):
    """Change exactly one fixed position to a novel byte."""
    def probe(ctx, rng):
        out = list(ctx)
        p = min(pos, len(out) - 1)
        out[p] = choose_novel_byte(ctx, rng)
        return out
    return probe


def make_repeat_local_probe(ctx, rng):
    """Copy the immediately preceding symbol into one position."""
    out = list(ctx)
    pos = rng.randrange(1, len(out))
    out[pos] = out[pos - 1]
    return out


def make_repeat_distant_probe(ctx, rng):
    """Copy a symbol from 4 positions away into one position."""
    if len(ctx) < 6:
        return make_repeat_local_probe(ctx, rng)
    out = list(ctx)
    pos = rng.randrange(4, len(out))
    out[pos] = out[pos - 4]
    return out


def make_swap_distance_probe(distance):
    """Swap two symbols exactly `distance` positions apart."""
    def probe(ctx, rng):
        out = list(ctx)
        if distance >= len(out):
            return out
        a = rng.randrange(len(out) - distance)
        b = a + distance
        out[a], out[b] = out[b], out[a]
        return out
    return probe


def make_move_distance_probe(distance):
    """Move one symbol exactly `distance` positions, preserving length."""
    def probe(ctx, rng):
        out = list(ctx)
        if distance >= len(out):
            return out
        a = rng.randrange(len(out) - distance)
        direction = 1 if rng.random() < 0.5 else -1
        b = a + direction * distance
        if b < 0 or b >= len(out):
            b = a + distance
        value = out.pop(a)
        out.insert(b, value)
        return out
    return probe


PROBES = {
    "identity": make_identity_probe,
    "repeat_local": make_repeat_local_probe,
    "repeat_distant": make_repeat_distant_probe,
    "swap_1": make_swap_distance_probe(1),
    "swap_4": make_swap_distance_probe(4),
    "swap_8": make_swap_distance_probe(8),
    "move_1": make_move_distance_probe(1),
    "move_4": make_move_distance_probe(4),
    "move_8": make_move_distance_probe(8),
    "identity_pos_0": make_identity_pos_probe(0),
    "identity_pos_4": make_identity_pos_probe(4),
    "identity_pos_8": make_identity_pos_probe(8),
    "identity_pos_12": make_identity_pos_probe(12),
}

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def softmax_probs(logits):
    return torch.softmax(
        torch.clamp(logits, -50.0, 50.0),
        dim=-1,
    )


def entropy(probs):
    return -(
        probs * torch.log2(probs.clamp_min(1e-12))
    ).sum(dim=-1)


def js_divergence(p, q):
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

    return 0.5 * (kl_pm + kl_qm)


def cosine(a, b):
    return (
        torch.dot(a, b)
        / (
            torch.linalg.vector_norm(a)
            * torch.linalg.vector_norm(b)
            + 1e-12
        )
    )


def probe_model(model, contexts, rng):
    model.eval()

    results = []

    with torch.no_grad():
        for start, ctx, target in contexts:
            base_t = torch.tensor(
                [ctx],
                dtype=torch.long,
                device=device,
            )

            base_logits = model.forward_all(base_t)
            base_probs = softmax_probs(base_logits[:, 0, :])

            # Latent representation is the embedding+hidden layer immediately
            # before the final output layer.
            base_latents = []
            for state in model.states:
                z = state.embedding(base_t)
                z = state.net[:3](z)
                base_latents.append(z.squeeze(0))

            base_latents = torch.stack(base_latents)

            probe_records = {}

            for name, make_probe in PROBES.items():
                perturbed = make_probe(ctx, rng)

                p_t = torch.tensor(
                    [perturbed],
                    dtype=torch.long,
                    device=device,
                )

                p_logits = model.forward_all(p_t)
                p_probs = softmax_probs(p_logits[:, 0, :])

                p_latents = []
                for state in model.states:
                    z = state.embedding(p_t)
                    z = state.net[:3](z)
                    p_latents.append(z.squeeze(0))

                p_latents = torch.stack(p_latents)

                state_changes = []

                for s in range(NUM_STATES):
                    pred_shift = (
                        js_divergence(
                            base_probs[s],
                            p_probs[s],
                        ).item()
                    )

                    latent_dist = (
                        torch.linalg.vector_norm(
                            base_latents[s] - p_latents[s]
                        ).item()
                    )

                    latent_cos = cosine(
                        base_latents[s],
                        p_latents[s],
                    ).item()

                    top_changed = int(
                        base_probs[s].argmax().item()
                        != p_probs[s].argmax().item()
                    )

                    state_changes.append({
                        "prediction_js": pred_shift,
                        "latent_l2": latent_dist,
                        "latent_cosine": latent_cos,
                        "top1_changed": top_changed,
                        "base_entropy": entropy(
                            base_probs[s]
                        ).item(),
                        "probe_entropy": entropy(
                            p_probs[s]
                        ).item(),
                    })

                probe_records[name] = {
                    "context": perturbed,
                    "states": state_changes,
                }

            results.append({
                "offset": start,
                "original_context": ctx,
                "target": int(target),
                "probes": probe_records,
            })

    return results


# ---------------------------------------------------------------------------
# Aggregate state fingerprints
# ---------------------------------------------------------------------------

def aggregate(results):
    fingerprint = {}

    for probe_name in PROBES:
        fingerprint[probe_name] = {
            "prediction_js": [0.0] * NUM_STATES,
            "latent_l2": [0.0] * NUM_STATES,
            "latent_cosine": [0.0] * NUM_STATES,
            "top1_change_rate": [0.0] * NUM_STATES,
        }

        n = len(results)

        for record in results:
            states = record["probes"][probe_name]["states"]

            for s in range(NUM_STATES):
                fingerprint[probe_name]["prediction_js"][s] += (
                    states[s]["prediction_js"]
                )
                fingerprint[probe_name]["latent_l2"][s] += (
                    states[s]["latent_l2"]
                )
                fingerprint[probe_name]["latent_cosine"][s] += (
                    states[s]["latent_cosine"]
                )
                fingerprint[probe_name]["top1_change_rate"][s] += (
                    states[s]["top1_changed"]
                )

        if n:
            for metric in fingerprint[probe_name]:
                fingerprint[probe_name][metric] = [
                    x / n
                    for x in fingerprint[probe_name][metric]
                ]

    return fingerprint


def fingerprint_vector(fingerprint, metric):
    """
    Flatten as:
       [probe identity S1..S4,
        probe repeat S1..S4, ...]
    """
    values = []

    for probe in PROBES:
        values.extend(
            fingerprint[probe][metric]
        )

    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def state_geometry(results):
    """
    Build a 4 x 4 distance matrix from how similarly states react across
    all probes.

    Each state gets a response vector containing:
      prediction JS
      latent L2
      top1-change
    for every probe/context.

    Then calculate pairwise Euclidean distances.
    """
    state_vectors = []

    for s in range(NUM_STATES):
        vector = []

        for record in results:
            for probe in PROBES:
                item = record["probes"][probe]["states"][s]

                vector.extend([
                    item["prediction_js"],
                    item["latent_l2"],
                    float(item["top1_changed"]),
                ])

        state_vectors.append(vector)

    X = np.asarray(state_vectors, dtype=float)

    distances = np.zeros((NUM_STATES, NUM_STATES))

    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            distances[i, j] = np.linalg.norm(
                X[i] - X[j]
            )

    # Classical MDS for a 2D visual shape.
    n = NUM_STATES
    D2 = distances ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coords = np.zeros((n, 2))

    for k in range(2):
        if eigenvalues[k] > 0:
            coords[:, k] = (
                eigenvectors[:, k]
                * math.sqrt(eigenvalues[k])
            )

    return {
        "distance_matrix": distances.tolist(),
        "coordinates_2d": coords.tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fingerprint(fingerprint, out_path, metric="prediction_js"):
    probes = list(PROBES.keys())

    matrix = np.asarray([
        fingerprint[p][metric]
        for p in probes
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(NUM_STATES))
    ax.set_xticklabels(
        [f"S{i + 1}" for i in range(NUM_STATES)]
    )
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels(probes)

    ax.set_xlabel("State")
    ax.set_ylabel("Perturbation")
    ax.set_title(
        f"S4E4 state fingerprint — {metric}"
    )

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(
                c,
                r,
                f"{matrix[r, c]:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_geometry(geometry, out_path):
    coords = np.asarray(
        geometry["coordinates_2d"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
    )

    for i in range(NUM_STATES):
        ax.annotate(
            f"S{i + 1}",
            (
                coords[i, 0],
                coords[i, 1],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
        )

    ax.axhline(0)
    ax.axvline(0)

    ax.set_xlabel("response-space axis 1")
    ax.set_ylabel("response-space axis 2")
    ax.set_title(
        "S4E4 state-response geometry"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_probe(path, sample_count, checkpoint_step):
    path = Path(path)

    data = path.read_bytes()

    if len(data) < CONTEXT_SIZE + 2:
        raise ValueError(
            "Input file is too small."
        )

    print(
        f"[*] S4E4 probing '{path}' "
        f"({len(data)} bytes)"
    )

    # Use several checkpoints. Avoid the very end so there are enough real
    # contexts after each checkpoint.
    checkpoints = sorted(set([
        min(
            checkpoint_step,
            len(data) - CONTEXT_SIZE - 1,
        ),
        min(
            checkpoint_step * 2,
            len(data) - CONTEXT_SIZE - 1,
        ),
        min(
            checkpoint_step * 4,
            len(data) - CONTEXT_SIZE - 1,
        ),
        min(
            checkpoint_step * 8,
            len(data) - CONTEXT_SIZE - 1,
        ),
        len(data) - CONTEXT_SIZE - 1,
    ]))

    checkpoints = [
        x for x in checkpoints
        if x > 0
    ]

    print(
        f"[*] Checkpoints: {checkpoints}"
    )

    rng = random.Random(SEED)

    all_checkpoint_results = []

    # We deliberately retrain from the beginning for each checkpoint.
    # This makes each captured model state an exact prefix of the same
    # learning trajectory, while keeping the code simple and auditable.
    for checkpoint in checkpoints:
        print(
            f"[*] Training to checkpoint {checkpoint:,}..."
        )

        model, optimizer, criterion = make_model()

        train_until(
            model,
            optimizer,
            criterion,
            data,
            0,
            checkpoint,
            CONTEXT_SIZE,
        )

        contexts = real_contexts(
            data,
            sample_count,
            rng,
        )

        print(
            f"    Probing {len(contexts)} contexts..."
        )

        t0 = time.time()
        results = probe_model(
            model,
            contexts,
            rng,
        )
        elapsed = time.time() - t0

        fp = aggregate(results)
        geometry = state_geometry(results)

        all_checkpoint_results.append({
            "checkpoint": checkpoint,
            "fingerprint": fp,
            "geometry": geometry,
            "sample_count": len(results),
        })

        print(
            f"    Probe time: {elapsed:.2f}s"
        )

    # Final checkpoint gets the detailed raw samples because that is the
    # most mature representation we have.
    final = all_checkpoint_results[-1]

    output_json = path.with_name(
        path.name + ".s4e4.json"
    )

    payload = {
        "experiment": "S4E4",
        "purpose": (
            "Discover whether independent states develop "
            "different response fingerprints."
        ),
        "input": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "seed": SEED,
        "states": NUM_STATES,
        "context_size": CONTEXT_SIZE,
        "checkpoints": all_checkpoint_results,
        "interpretation_note": (
            "Probe labels describe controlled perturbations only. "
            "They are not assigned state roles."
        ),
    }

    output_json.write_text(
        json.dumps(payload, indent=2)
    )

    fp_plot = path.with_name(
        path.name + ".s4e4_fingerprint.png"
    )

    geom_plot = path.with_name(
        path.name + ".s4e4_geometry.png"
    )

    plot_fingerprint(
        final["fingerprint"],
        fp_plot,
        metric="prediction_js",
    )

    plot_geometry(
        final["geometry"],
        geom_plot,
    )

    print()
    print(
        f"[+] Results: {output_json}"
    )
    print(
        f"[+] Fingerprint shape: {fp_plot}"
    )
    print(
        f"[+] State geometry: {geom_plot}"
    )
    print()
    print(
        "Look for distance trends, position trends, and state-specific profiles."
    )
    print(
        "A state-specific pattern across perturbations is more "
        "interesting than a single high number."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["probe"],
    )

    parser.add_argument(
        "file",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Real contexts sampled per checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1000,
        help="Base training checkpoint interval.",
    )

    args = parser.parse_args()

    run_probe(
        args.file,
        args.samples,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
