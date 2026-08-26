#!/usr/bin/env python3
# R4 Local Token Anatomy Analysis
# Loads R3 local_states_*.npy files and measures token structure.

import argparse
import json
from pathlib import Path
import numpy as np


def analyze(tokens):
    # tokens: [images, tokens, local_dim]
    n, t, d = tokens.shape

    token_variance = tokens.var(axis=0).mean(axis=1)

    token_means = tokens.mean(axis=0)
    token_similarity = np.corrcoef(token_means)

    token_activity = tokens.mean(axis=2).var(axis=0)

    return {
        "images": int(n),
        "tokens": int(t),
        "local_dim": int(d),
        "token_variance": token_variance.tolist(),
        "token_activity": token_activity.tolist(),
        "token_similarity": token_similarity.tolist(),
        "overall_variance": float(tokens.var()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-states", required=True, type=Path)
    parser.add_argument("--out", default="r4_token_analysis.json")
    args = parser.parse_args()

    tokens = np.load(args.local_states)
    result = analyze(tokens)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("R4 complete")
    print(json.dumps({
        "tokens": result["tokens"],
        "local_dim": result["local_dim"],
        "overall_variance": result["overall_variance"]
    }, indent=2))


if __name__ == "__main__":
    main()
