#!/usr/bin/env python3
"""NCA-01 hex-enabled runner.

This is a compatibility variant of NCA-01, not a new archive format or
research experiment. With no --hex flag it delegates directly to the
unchanged NCA-01 implementation. With --hex, only the neural candidate is
run over a reversible ASCII-hex representation before neural coding.

Examples:
  python NCA-01_hex.py pack INPUT ARCHIVE --strategy adaptive
  python NCA-01_hex.py pack INPUT ARCHIVE --strategy adaptive --hex
  python NCA-01_hex.py test ARCHIVE

The original NCA-01 file remains the authoritative baseline.
"""
from __future__ import annotations

import importlib.util
import sys
from binascii import unhexlify
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "NCA-01_local_adaptive_archive.py"


def load_base():
    spec = importlib.util.spec_from_file_location("nca01_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline: {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_hex_mode(base):
    """Patch only the neural candidate path with reversible ASCII hex."""
    original_encode = base.neural_encode
    original_decode = base.neural_decode

    def hex_neural_encode(data: bytes, context_size=16, lr=0.04, num_states=4):
        transformed = data.hex().encode("ascii")
        return original_encode(transformed, context_size, lr, num_states)

    def hex_neural_decode(stream: bytes, size: int, context_size=16, lr=0.04, num_states=4):
        transformed_size = size * 2
        transformed = original_decode(stream, transformed_size, context_size, lr, num_states)
        return unhexlify(transformed)

    def encode_candidate(data, strategy):
        if strategy == "neural":
            return hex_neural_encode(data)
        return base.encode_candidate(data, strategy)

    def decode_candidate(payload, size, strategy):
        if strategy == "neural":
            return hex_neural_decode(payload, size)
        return base.decode_candidate(payload, size, strategy)

    base.neural_encode = hex_neural_encode
    base.neural_decode = hex_neural_decode
    base.encode_candidate = encode_candidate
    base.decode_candidate = decode_candidate


def main():
    use_hex = "--hex" in sys.argv
    if use_hex:
        sys.argv.remove("--hex")

    base = load_base()
    if use_hex:
        install_hex_mode(base)
    base.main()


if __name__ == "__main__":
    main()
