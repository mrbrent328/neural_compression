#!/usr/bin/env python3
"""
Evaluate the functional effect of the reconstructed LayerNorm weights.

Pipeline:
  1. Load the original Qwen2-0.5B.
  2. Load layernorm_only_v1.pt.
  3. Reconstruct ONLY the 48 LayerNorm tensors from the saved global-key
     constructor.
  4. Build a second model with those reconstructed LayerNorm weights.
  5. Compare original vs reconstructed model on:
       - parameter RMSE
       - logits / next-token agreement
       - cross-entropy on a small evaluation corpus
       - generated text from identical prompts

Nothing except the selected LayerNorm weights is changed.

The checkpoint produced by the LayerNorm-only training script is expected
to contain:
    constructor_state
    layernorm_names
    global_latent
    chunk_size
    rank
"""

import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "Qwen2-0.5B"
CHECKPOINT_FILE = "layernorm_only_v1.pt"

OUTPUT_JSON = "layernorm_functional_test_v1.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float32

# Keep the evaluation deliberately small at first.
EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is a field concerned with learning patterns from data.",
    "A transformer model processes information through repeated attention and feed-forward layers.",
    "The ocean contains many different ecosystems, from shallow coastal waters to the deep sea.",
    "Plants convert light energy into chemical energy through photosynthesis.",
    "A computer program is a sequence of instructions executed by a machine.",
    "Mathematics provides a language for describing patterns, quantities, and relationships.",
    "The history of science is full of experiments that changed our understanding of nature.",
]

PROMPTS = [
    "The purpose of a neural network is",
    "A transformer model works by",
    "The most interesting thing about learning is",
    "If we reconstruct a model from a compact representation,",
]


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    print(f"Loading constructor checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def normalized_rmse(original, reconstructed):
    diff = reconstructed.float() - original.float()
    mse = diff.pow(2).mean()
    denom = original.float().pow(2).mean().clamp_min(1e-12)
    return torch.sqrt(mse / denom).item()


def perplexity_from_loss(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


# ============================================================
# RECONSTRUCTION
# ============================================================

@torch.no_grad()
def reconstruct_layernorms(checkpoint):
    """
    Recreate the exact constructor structure used by layernorm_only_v1.

    The saved constructor_state contains:
        constructor.network.*
        projection
        latent

    The checkpoint's constructor output is:
        [batch, rank, chunk_size]

    and projection collapses rank -> one reconstructed chunk.
    """

    constructor_state = checkpoint["constructor_state"]

    # Infer dimensions from the saved state rather than hard-coding them.
    network = torch.nn.Sequential(
        torch.nn.Linear(
            checkpoint["latent_dim"],
            checkpoint["hidden"],
        ),
        torch.nn.GELU(),
        torch.nn.Linear(
            checkpoint["hidden"],
            checkpoint["hidden"],
        ),
        torch.nn.GELU(),
        torch.nn.Linear(
            checkpoint["hidden"],
            checkpoint["rank"] * checkpoint["chunk_size"],
        ),
    )

    network.load_state_dict({
        k.replace("constructor.", "", 1): v
        for k, v in constructor_state.items()
        if k.startswith("constructor.")
    })

    network.eval()

    projection = constructor_state["projection"].float()
    latent = constructor_state["latent"].float()

    # Global latent -> rank x chunk.
    factors = network(latent.unsqueeze(0))
    factors = factors.view(
        1,
        checkpoint["rank"],
        checkpoint["chunk_size"],
    )

    # rank x chunk -> chunk.
    reconstructed_chunk = torch.einsum(
        "brc,r->bc",
        factors,
        projection,
    )[0]

    # The training script normalized each LayerNorm tensor independently
    # before chunking. Qwen LayerNorm vectors are small enough that each
    # target is one chunk, so recover each target using the original model's
    # mean/std.
    return reconstructed_chunk


@torch.no_grad()
def build_reconstructed_state(
    original_model,
    checkpoint,
):
    """
    Reconstruct each LayerNorm tensor and return a dictionary containing
    only replacements. Original tensors remain untouched.
    """

    names = checkpoint["layernorm_names"]
    reconstructed_chunk = reconstruct_layernorms(checkpoint)

    replacements = {}
    diagnostics = []

    state = original_model.state_dict()

    for name in names:

        original = state[name].detach().cpu().float().flatten()

        mean = original.mean()
        std = original.std(unbiased=False)

        if std < 1e-12:
            std = torch.tensor(1.0)

        # Undo the per-tensor normalization used during training.
        reconstructed = (
            reconstructed_chunk[: original.numel()] * std
            + mean
        )

        reconstructed = reconstructed.reshape(
            state[name].shape
        )

        replacements[name] = reconstructed

        diagnostics.append({
            "name": name,
            "shape": list(state[name].shape),
            "normalized_rmse":
                normalized_rmse(
                    original,
                    reconstructed,
                ),
            "original_mean": mean.item(),
            "original_std": std.item(),
        })

    return replacements, diagnostics


def apply_replacements(model, replacements):
    """
    Apply ONLY LayerNorm replacements.
    """

    state = model.state_dict()

    for name, tensor in replacements.items():
        state[name].copy_(
            tensor.to(
                device=state[name].device,
                dtype=state[name].dtype,
            )
        )


# ============================================================
# LOGIT TEST
# ============================================================

@torch.no_grad()
def compare_logits(
    original_model,
    reconstructed_model,
    tokenizer,
    texts,
):
    results = []

    original_model.eval()
    reconstructed_model.eval()

    for text in texts:

        encoded = tokenizer(
            text,
            return_tensors="pt",
        ).to(DEVICE)

        original_logits = original_model(
            **encoded
        ).logits

        reconstructed_logits = reconstructed_model(
            **encoded
        ).logits

        # Compare final-position logits.
        a = original_logits[:, -1, :].float()
        b = reconstructed_logits[:, -1, :].float()

        mse = (a - b).pow(2).mean().item()

        cosine = torch.nn.functional.cosine_similarity(
            a,
            b,
            dim=-1,
        ).item()

        original_top = torch.argmax(a, dim=-1).item()
        reconstructed_top = torch.argmax(b, dim=-1).item()

        results.append({
            "text": text,
            "logit_mse": mse,
            "logit_cosine": cosine,
            "original_top_token_id": original_top,
            "reconstructed_top_token_id": reconstructed_top,
            "same_top_token":
                original_top == reconstructed_top,
        })

    return results


# ============================================================
# CROSS-ENTROPY TEST
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    tokenizer,
    texts,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in texts:

        encoded = tokenizer(
            text,
            return_tensors="pt",
        ).to(DEVICE)

        labels = encoded["input_ids"].clone()

        outputs = model(
            **encoded,
            labels=labels,
        )

        # HF loss is averaged over predicted tokens.
        token_count = labels.numel() - 1

        total_loss += outputs.loss.item() * token_count
        total_tokens += token_count

    loss = total_loss / max(total_tokens, 1)

    return {
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
    }


# ============================================================
# GENERATION TEST
# ============================================================

@torch.no_grad()
def generate_outputs(
    model,
    tokenizer,
    prompts,
):
    model.eval()

    outputs = []

    for prompt in prompts:

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(DEVICE)

        generated = model.generate(
            **encoded,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        outputs.append({
            "prompt": prompt,
            "output": text,
        })

    return outputs


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(9)

    print("=" * 70)
    print("LAYERNORM FUNCTIONAL RECONSTRUCTION TEST")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    checkpoint = load_checkpoint(
        CHECKPOINT_FILE
    )

    print(
        f"Checkpoint best epoch: "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Checkpoint normalized MSE: "
        f"{checkpoint.get('best_loss')}"
    )

    print(
        f"Checkpoint latent: "
        f"{checkpoint.get('global_latent')}"
    )

    print()

    # --------------------------------------------------------
    # Load original model
    # --------------------------------------------------------

    print("Loading original model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    original_model.eval()

    # --------------------------------------------------------
    # Build reconstructed LayerNorm tensors
    # --------------------------------------------------------

    print("Reconstructing LayerNorm tensors...")

    replacements, diagnostics = (
        build_reconstructed_state(
            original_model,
            checkpoint,
        )
    )

    print(
        f"Reconstructed tensors: "
        f"{len(replacements)}"
    )

    print()

    # --------------------------------------------------------
    # Clone model
    # --------------------------------------------------------

    print("Creating reconstructed model...")

    reconstructed_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=DTYPE,
        ).to(DEVICE)
    )

    reconstructed_model.eval()

    # --------------------------------------------------------
    # Apply ONLY LayerNorm replacements
    # --------------------------------------------------------

    print("Replacing ONLY LayerNorm weights...")

    apply_replacements(
        reconstructed_model,
        replacements,
    )

    print("Done.")
    print()

    # --------------------------------------------------------
    # Parameter diagnostics
    # --------------------------------------------------------

    rmse_values = [
        x["normalized_rmse"]
        for x in diagnostics
    ]

    parameter_summary = {
        "count": len(rmse_values),
        "mean_normalized_rmse":
            sum(rmse_values) / len(rmse_values),
        "min_normalized_rmse":
            min(rmse_values),
        "max_normalized_rmse":
            max(rmse_values),
    }

    # --------------------------------------------------------
    # Loss / perplexity
    # --------------------------------------------------------

    print("=" * 70)
    print("LOSS TEST")
    print("=" * 70)

    original_loss = evaluate_loss(
        original_model,
        tokenizer,
        EVAL_TEXTS,
    )

    reconstructed_loss = evaluate_loss(
        reconstructed_model,
        tokenizer,
        EVAL_TEXTS,
    )

    print(
        f"Original       loss={original_loss['loss']:.6f} "
        f"ppl={original_loss['perplexity']:.4f}"
    )

    print(
        f"Reconstructed  loss={reconstructed_loss['loss']:.6f} "
        f"ppl={reconstructed_loss['perplexity']:.4f}"
    )

    # --------------------------------------------------------
    # Logit comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOGIT TEST")
    print("=" * 70)

    logit_results = compare_logits(
        original_model,
        reconstructed_model,
        tokenizer,
        EVAL_TEXTS,
    )

    for result in logit_results:

        print(
            f"cos={result['logit_cosine']:.6f} "
            f"mse={result['logit_mse']:.6e} "
            f"same_top={result['same_top_token']}"
        )

    cosine_values = [
        x["logit_cosine"]
        for x in logit_results
    ]

    top_agreement = sum(
        x["same_top_token"]
        for x in logit_results
    ) / len(logit_results)

    logit_summary = {
        "mean_cosine":
            sum(cosine_values) / len(cosine_values),
        "min_cosine":
            min(cosine_values),
        "top_token_agreement":
            top_agreement,
    }

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION TEST")
    print("=" * 70)

    original_generations = generate_outputs(
        original_model,
        tokenizer,
        PROMPTS,
    )

    reconstructed_generations = generate_outputs(
        reconstructed_model,
        tokenizer,
        PROMPTS,
    )

    generation_results = []

    for original, reconstructed in zip(
        original_generations,
        reconstructed_generations,
    ):

        print()
        print("PROMPT:")
        print(original["prompt"])

        print()
        print("ORIGINAL:")
        print(original["output"])

        print()
        print("RECONSTRUCTED:")
        print(reconstructed["output"])

        generation_results.append({
            "prompt":
                original["prompt"],
            "original":
                original["output"],
            "reconstructed":
                reconstructed["output"],
            "exact_match":
                original["output"]
                == reconstructed["output"],
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result = {
        "experiment":
            "layernorm_functional_test_v1",

        "model_dir":
            MODEL_DIR,

        "checkpoint":
            CHECKPOINT_FILE,

        "device":
            DEVICE,

        "reconstructed_tensor_count":
            len(replacements),

        "parameter_summary":
            parameter_summary,

        "loss": {
            "original":
                original_loss,
            "reconstructed":
                reconstructed_loss,
        },

        "logits": {
            "summary":
                logit_summary,
            "per_text":
                logit_results,
        },

        "generation":
            generation_results,

        "per_tensor":
            diagnostics,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        f"Results saved to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
