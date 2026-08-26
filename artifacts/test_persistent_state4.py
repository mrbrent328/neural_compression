#!/usr/bin/env python3
"""
file_abc_profiler.py - Classifies file chunks into structural A, B, C states.
"""

import sys
import math
from collections import Counter

def calculate_entropy(chunk: bytes) -> float:
    if not chunk:
        return 0.0
    counts = Counter(chunk)
    length = len(chunk)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def get_dominant_byte_ratio(chunk: bytes) -> tuple[int, float]:
    if not chunk:
        return (0, 0.0)
    counts = Counter(chunk)
    most_common_byte, freq = counts.most_common(1)[0]
    return most_common_byte, freq / len(chunk)

def profile_file(file_path: str, chunk_size: int = 32768):
    with open(file_path, "rb") as f:
        data = f.read()

    total_bytes = len(data)
    chunks = [data[i : i + chunk_size] for i in range(0, total_bytes, chunk_size)]

    file_manifest = []

    print(f"Profiling '{file_path}' ({total_bytes} bytes in {len(chunks)} chunks)...\n")
    print(f"{'Chunk':<7} | {'Type':<6} | {'Entropy':<10} | {'Dominant Byte':<15} | {'Structural Classification'}")
    print("-" * 75)

    for idx, chunk in enumerate(chunks):
        entropy = calculate_entropy(chunk)
        dom_byte, dom_ratio = get_dominant_byte_ratio(chunk)

        # Classification Rules:
        # A = Uniform / Dominant (>70% single symbol or low entropy)
        # B = Structured / Predictable (Medium entropy: code, text, structured data)
        # C = High Entropy / Complex (Encrypted, compressed, or random noise)
        if dom_ratio > 0.70 or entropy < 2.5:
            chunk_type = "A (Uniform)"
            tag = "A"
        elif 2.5 <= entropy < 6.2:
            chunk_type = "B (Structured)"
            tag = "B"
        else:
            chunk_type = "C (Complex)"
            tag = "C"

        file_manifest.append(tag)
        dom_str = f"0x{dom_byte:02X} ({dom_ratio*100:.1f}%)"
        print(f"{idx:<7} | {tag:<6} | {entropy:<10.2f} | {dom_str:<15} | {chunk_type}")

    macro_recipe = "".join(file_manifest)
    print("\n" + "=" * 75)
    print(f"WHOLE FILE MACRO MAP (A/B/C Abstract Sequence):")
    print(f"[{macro_recipe}]")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_abc_profiler.py <file_path>")
        sys.exit(1)
    profile_file(sys.argv[1])
