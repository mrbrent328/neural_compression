#!/usr/bin/env python3
"""
EXPERIMENT 8: LATENT -> HARMONY -> GPT-OSS -> SECRET

Goal
----
Test the user's hypothesis in a model-mediated way:

    known secret
        ↓
    structured GPT-OSS interaction
        ↓
    representation / latent
        ↓
    latent decoder
        ↓
    reconstructed Harmony-compatible prompt
        ↓
    GPT-OSS
        ↓
    recovered secret?

This is deliberately a FIRST-PASS experiment.

Important:
- It does NOT touch the 20B model weights.
- It does NOT claim that a latent "contains" a secret.
- It tests whether a compact learned representation can preserve enough
  information to recreate a prompt that causes the model to recover it.
- It uses held-out secrets to distinguish generalization from memorization.

Backend
-------
The script supports a local OpenAI-compatible HTTP endpoint.

Default:
    http://127.0.0.1:1234/v1

This is convenient for LM Studio, but you can change it with:
    --base-url
    --model

The script asks the model to answer in a very constrained way so evaluation
is easy. If the model/backend does not expose logits, we compare generated
text. The prompt reconstruction portion is local and independent of the
backend.

Two experiments are performed:

A. PROMPT AUTOENCODER
   Harmony-formatted prompt -> latent -> reconstructed prompt

B. BEHAVIORAL RECONSTRUCTION
   Original prompt vs reconstructed prompt -> GPT-OSS
   Compare whether the secret is recovered.

The latent is trained as a compact representation of the structured prompt.
We do NOT train on the held-out secrets.

Dependencies:
    numpy
    torch
    requests

Example:
    python3 experiment8_latent_harmony_secret.py

If your LM Studio endpoint/model differs:
    python3 experiment8_latent_harmony_secret.py \
      --base-url http://127.0.0.1:1234/v1 \
      --model your-model-name
"""

import argparse
import hashlib
import json
import re
import sys
import time

import numpy as np
import requests
import torch
import torch.nn as nn


SEED = 42

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = None

TRAIN_COUNT = 80
TEST_COUNT = 20

LATENT_DIMS = [8, 16, 32]

EPOCHS = 1200
LEARNING_RATE = 1e-3

MAX_TEXT = 768

# We deliberately use a fixed secret vocabulary pattern, but randomize the
# actual payload per trial.
WORDS_A = [
    "BLUE", "COPPER", "VIOLET", "MOSS", "EMBER",
    "SILVER", "THISTLE", "GLASS", "CRIMSON", "GOLD",
]

WORDS_B = [
    "MUSHROOM", "FROG", "COMET", "ENGINE", "CIRCUIT",
    "ORCHID", "RAVEN", "CLOCK", "FOREST", "ORBIT",
]


def make_secret(rng):
    a = rng.choice(WORDS_A)
    b = rng.choice(WORDS_B)
    n = int(rng.integers(100, 999))
    return f"{a}-{b}-{n}"


def harmony_prompt(secret):
    """
    A simple Harmony-compatible structured prompt.

    We intentionally keep this minimal. The exact token serialization is
    handled by the local backend/model template where possible; the textual
    structure here is the experimental object we compress.
    """
    return (
        "<|start|>system<|message|>"
        "You are a precise information retrieval assistant."
        "<|end|>"
        "<|start|>user<|message|>"
        "Remember the following secret identifier. "
        "When asked for the secret, output only the identifier. "
        f"SECRET={secret}"
        "<|end|>"
        "<|start|>assistant<|channel|>final"
    )


def query_model(
    session,
    base_url,
    model,
    prompt,
    max_tokens=64,
    temperature=0.0,
):
    url = base_url.rstrip("/") + "/chat/completions"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise information retrieval assistant. "
                "Follow the user's requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if model:
        payload["model"] = model

    response = session.post(
        url,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)


def extract_secret(text):
    """
    Find the secret-shaped token in model output.
    """
    pattern = (
        r"\b[A-Z]+-[A-Z]+-[0-9]{3}\b"
    )

    match = re.search(pattern, text.upper())

    if match:
        return match.group(0)

    return None


class TextLatentAutoencoder(nn.Module):
    """
    Fixed-width byte representation -> compact latent -> byte representation.

    This is intentionally simple.

    The important experiment is not whether the autoencoder can perfectly
    reproduce text; it is whether a compact representation can preserve
    enough structure to produce a prompt that makes GPT-OSS recover the
    correct secret.
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def encode_text(text):
    raw = text.encode("utf-8")[:MAX_TEXT]

    data = np.zeros(
        MAX_TEXT,
        dtype=np.float32,
    )

    data[:len(raw)] = (
        np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(np.float32)
        / 255.0
    )

    return data


def decode_text(vector):
    values = np.clip(
        np.rint(vector * 255.0),
        0,
        255,
    ).astype(np.uint8)

    # Stop at the first NUL padding byte.
    values = values[
        :(
            np.where(values == 0)[0][0]
            if np.any(values == 0)
            else len(values)
        )
    ]

    try:
        return bytes(values.tolist()).decode(
            "utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def train_autoencoder(texts, latent_dim):
    x = np.stack(
        [encode_text(t) for t in texts]
    )

    x = torch.from_numpy(x)

    model = TextLatentAutoencoder(
        MAX_TEXT,
        latent_dim,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        y = model(x)
        loss = loss_fn(y, x)

        loss.backward()
        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 300 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"    epoch {epoch + 1:4d}/{EPOCHS}"
                f" | reconstruction MSE {loss.item():.8f}"
            )

    return model


@torch.no_grad()
def reconstruct_text(model, text):
    x = torch.from_numpy(
        encode_text(text)
    ).unsqueeze(0)

    z = model.encode(x)
    y = model.decode(z)

    return (
        z[0].numpy(),
        decode_text(y[0].numpy()),
    )


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--no-model-probe",
        action="store_true",
        help="Skip live GPT-OSS calls and test latent reconstruction only.",
    )

    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 78)
    print("EXPERIMENT 8: LATENT -> HARMONY -> GPT-OSS -> SECRET")
    print("=" * 78)
    print(f"Endpoint: {args.base_url}")
    print(f"Model   : {args.model or '(server default)'}")
    print()

    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------
    # Generate secrets.
    # ---------------------------------------------------------------
    secrets = [
        make_secret(rng)
        for _ in range(TRAIN_COUNT + TEST_COUNT)
    ]

    train_secrets = secrets[:TRAIN_COUNT]
    test_secrets = secrets[TRAIN_COUNT:]

    train_prompts = [
        harmony_prompt(s)
        for s in train_secrets
    ]

    test_prompts = [
        harmony_prompt(s)
        for s in test_secrets
    ]

    print("[1] Secret dataset")
    print(f"Training secrets: {TRAIN_COUNT}")
    print(f"Held-out secrets: {TEST_COUNT}")
    print()
    print("Example training secret:", train_secrets[0])
    print("Example held-out secret:", test_secrets[0])
    print()

    session = requests.Session()

    # ---------------------------------------------------------------
    # Optional connectivity test.
    # ---------------------------------------------------------------
    if not args.no_model_probe:
        print("[2] Testing GPT-OSS endpoint...")

        try:
            probe = query_model(
                session,
                args.base_url,
                args.model,
                (
                    "Return exactly the word READY and nothing else."
                ),
                max_tokens=8,
                temperature=0.0,
            )

            print("Model response:", repr(probe))
            print()

        except Exception as exc:
            print()
            print("MODEL CONNECTION FAILED")
            print(str(exc))
            print()
            print(
                "The latent portion can still run with "
                "--no-model-probe."
            )
            sys.exit(1)

    # ---------------------------------------------------------------
    # Baseline: original prompts should recover their own secrets.
    # ---------------------------------------------------------------
    baseline_ok = 0

    if not args.no_model_probe:
        print("[3] Baseline: original prompt -> GPT-OSS")

        for secret in test_secrets:
            prompt = harmony_prompt(secret)

            try:
                output = query_model(
                    session,
                    args.base_url,
                    args.model,
                    (
                        prompt
                        + "\nReturn the secret identifier only."
                    ),
                    max_tokens=32,
                    temperature=0.0,
                )

                recovered = extract_secret(output)

                ok = recovered == secret
                baseline_ok += int(ok)

                print(
                    f"  {secret:24s} -> "
                    f"{str(recovered):24s} "
                    f"{'PASS' if ok else 'FAIL'}"
                )

            except Exception as exc:
                print(
                    f"  {secret:24s} -> ERROR: {exc}"
                )

        print(
            f"Baseline recovery: "
            f"{baseline_ok}/{TEST_COUNT}"
        )
        print()

    # ---------------------------------------------------------------
    # Latent experiments.
    # ---------------------------------------------------------------
    for latent_dim in LATENT_DIMS:
        print()
        print("-" * 78)
        print(f"LATENT DIMENSION = {latent_dim}")
        print("-" * 78)

        print("[4] Training prompt latent...")

        model = train_autoencoder(
            train_prompts,
            latent_dim,
        )

        print()

        train_exact = 0
        test_exact = 0

        train_latents = []

        print("[5] Reconstruction checks")

        for secret, prompt in zip(
            train_secrets,
            train_prompts,
        ):
            z, reconstructed = reconstruct_text(
                model,
                prompt,
            )

            train_latents.append(z)

            if reconstructed == prompt:
                train_exact += 1

        for secret, prompt in zip(
            test_secrets,
            test_prompts,
        ):
            z, reconstructed = reconstruct_text(
                model,
                prompt,
            )

            if reconstructed == prompt:
                test_exact += 1

        print(
            f"Exact prompt reconstruction "
            f"(training): {train_exact}/{TRAIN_COUNT}"
        )

        print(
            f"Exact prompt reconstruction "
            f"(held-out): {test_exact}/{TEST_COUNT}"
        )

        print()
        print(
            "Example held-out original prompt:"
        )
        print(test_prompts[0][:300])

        _, example_reconstructed = reconstruct_text(
            model,
            test_prompts[0],
        )

        print()
        print(
            "Example held-out reconstructed prompt:"
        )
        print(example_reconstructed[:300])

        print()

        # -----------------------------------------------------------
        # Model-mediated test.
        # -----------------------------------------------------------
        if not args.no_model_probe:
            print(
                "[6] MODEL-MEDIATED TEST: "
                "reconstructed prompt -> GPT-OSS"
            )

            recovered_count = 0

            for secret, prompt in zip(
                test_secrets,
                test_prompts,
            ):
                z, reconstructed = reconstruct_text(
                    model,
                    prompt,
                )

                # The decoder may not produce perfectly clean Harmony.
                # We append a normal user instruction to make the intended
                # task explicit while preserving the reconstructed object.
                model_prompt = (
                    reconstructed
                    + "\nReturn the secret identifier only."
                )

                try:
                    output = query_model(
                        session,
                        args.base_url,
                        args.model,
                        model_prompt,
                        max_tokens=32,
                        temperature=0.0,
                    )

                    recovered = extract_secret(output)

                    ok = recovered == secret
                    recovered_count += int(ok)

                    print(
                        f"  {secret:24s} -> "
                        f"{str(recovered):24s} "
                        f"{'PASS' if ok else 'FAIL'}"
                    )

                except Exception as exc:
                    print(
                        f"  {secret:24s} -> ERROR: {exc}"
                    )

            print()
            print(
                f"Latent-mediated recovery: "
                f"{recovered_count}/{TEST_COUNT}"
            )

        # -----------------------------------------------------------
        # Latent inspection.
        # -----------------------------------------------------------
        latent_matrix = np.stack(train_latents)

        print()
        print("[7] LATENT INSPECTION")

        print(
            f"Latent shape: {latent_matrix.shape}"
        )

        print(
            f"Latent mean : "
            f"{latent_matrix.mean(axis=0)}"
        )

        print(
            f"Latent std  : "
            f"{latent_matrix.std(axis=0)}"
        )

        print(
            f"Latent range: "
            f"[{latent_matrix.min():.6f}, "
            f"{latent_matrix.max():.6f}]"
        )

        latent_hash = hashlib.sha256(
            np.ascontiguousarray(latent_matrix).tobytes()
        ).hexdigest()

        print(f"Latent SHA256: {latent_hash}")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        "The strongest result would be:"
    )

    print(
        "  held-out secret"
        " -> latent"
        " -> reconstructed Harmony interaction"
        " -> GPT-OSS"
        " -> same held-out secret"
    )

    print()
    print(
        "That would show that the compact representation preserved "
        "information in a form that the model can use to recover the "
        "target."
    )

    print()
    print(
        "It would NOT by itself prove that the latent contains the "
        "secret in some mysterious non-numerical sense. We would need "
        "controls and eventually compare against direct textual "
        "compression and randomized/shuffled latents."
    )


if __name__ == "__main__":
    main()
