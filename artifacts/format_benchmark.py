#!/usr/bin/env python3

"""
format_benchmark.py

Lossless representation benchmark for neural_compress_np.py.

For every file in a directory:

    original
       |
       +--> raw bytes
       +--> byte delta
       +--> byte XOR
       +--> byte transpose
       +--> uint16 delta
       +--> uint32 delta
       +--> uint16 byte-swap
       +--> uint32 byte-swap
       |
       +--> neural_compress_np.py
       |
       +--> neural_compress_np.py decompress
       |
       +--> inverse transform
       |
       +--> SHA-256 verification

All transforms are reversible.

Usage:

    python3 format_benchmark.py /path/to/corpus

or:

    python3 format_benchmark.py \
        /path/to/corpus \
        --compressor /path/to/neural_compress_np.py

Results:

    benchmark_results/
        results.csv
        results.json
        per_file/
"""


import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# Utilities
# ============================================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def file_size(path):
    return os.path.getsize(path)


def run_command(cmd):
    start = time.perf_counter()

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.perf_counter() - start

    return (
        proc.returncode,
        elapsed,
        proc.stdout,
        proc.stderr,
    )


def format_size(n):
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(n)

    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


# ============================================================================
# RAW
# ============================================================================

def transform_raw(data):
    return data


def inverse_raw(data, original_size):
    return data[:original_size]


# ============================================================================
# BYTE DELTA
# ============================================================================

def transform_byte_delta(data):
    if not data:
        return b""

    out = bytearray(len(data))

    previous = 0

    for i, value in enumerate(data):
        out[i] = (value - previous) & 0xFF
        previous = value

    return bytes(out)


def inverse_byte_delta(data, original_size):
    out = bytearray(len(data))

    previous = 0

    for i, value in enumerate(data):
        current = (value + previous) & 0xFF

        out[i] = current

        previous = current

    return bytes(out[:original_size])


# ============================================================================
# BYTE XOR
# ============================================================================

def transform_byte_xor(data):
    if not data:
        return b""

    out = bytearray(len(data))

    previous = 0

    for i, value in enumerate(data):
        out[i] = value ^ previous
        previous = value

    return bytes(out)


def inverse_byte_xor(data, original_size):
    out = bytearray(len(data))

    previous = 0

    for i, value in enumerate(data):
        current = value ^ previous

        out[i] = current

        previous = current

    return bytes(out[:original_size])


# ============================================================================
# BYTE TRANSPOSE
#
# Reorganizes bytes within blocks without changing any information.
#
# Example:
#
# ABCDEFGH
#
# becomes:
#
# ACEGBDFH
#
# depending on the block geometry.
# ============================================================================

def transform_transpose(data, block_size):
    out = bytearray()

    for start in range(0, len(data), block_size):

        block = data[start:start + block_size]

        if len(block) <= 1:
            out.extend(block)
            continue

        rows = int(len(block) ** 0.5)

        while rows > 1 and len(block) % rows != 0:
            rows -= 1

        if rows <= 1:
            out.extend(block)
            continue

        cols = len(block) // rows

        for c in range(cols):
            for r in range(rows):
                out.append(block[r * cols + c])

    return bytes(out)


def inverse_transpose(data, block_size, original_size):
    out = bytearray()

    pos = 0

    while pos < len(data):

        block_len = min(
            block_size,
            len(data) - pos,
        )

        if block_len <= 1:
            out.extend(
                data[pos:pos + block_len]
            )

            pos += block_len
            continue

        rows = int(block_len ** 0.5)

        while rows > 1 and block_len % rows != 0:
            rows -= 1

        if rows <= 1:
            out.extend(
                data[pos:pos + block_len]
            )

            pos += block_len
            continue

        cols = block_len // rows

        block = bytearray(block_len)

        src = pos

        for c in range(cols):
            for r in range(rows):

                block[r * cols + c] = data[src]

                src += 1

        out.extend(block)

        pos += block_len

    return bytes(out[:original_size])


# ============================================================================
# WORD DELTA
#
# Operates on little-endian 16-bit or 32-bit words.
#
# Trailing bytes that don't form a complete word are preserved unchanged.
# ============================================================================

def transform_word_delta(data, width):
    if width not in (2, 4):
        raise ValueError("width must be 2 or 4")

    full = len(data) // width

    remainder = len(data) % width

    out = bytearray(len(data))

    previous = 0

    modulus = 1 << (width * 8)

    for i in range(full):

        pos = i * width

        value = int.from_bytes(
            data[pos:pos + width],
            "little",
            signed=False,
        )

        delta = (value - previous) % modulus

        out[pos:pos + width] = delta.to_bytes(
            width,
            "little",
            signed=False,
        )

        previous = value

    if remainder:
        out[-remainder:] = data[-remainder:]

    return bytes(out)


def inverse_word_delta(data, original_size, width):
    full = len(data) // width

    remainder = len(data) % width

    out = bytearray(len(data))

    previous = 0

    modulus = 1 << (width * 8)

    for i in range(full):

        pos = i * width

        delta = int.from_bytes(
            data[pos:pos + width],
            "little",
            signed=False,
        )

        value = (previous + delta) % modulus

        out[pos:pos + width] = value.to_bytes(
            width,
            "little",
            signed=False,
        )

        previous = value

    if remainder:
        out[-remainder:] = data[-remainder:]

    return bytes(out[:original_size])


# ============================================================================
# BYTE SWAP
# ============================================================================

def transform_byteswap(data, width):
    if width not in (2, 4):
        raise ValueError("width must be 2 or 4")

    out = bytearray(len(data))

    full = len(data) // width

    for i in range(full):

        pos = i * width

        out[pos:pos + width] = (
            data[pos:pos + width][::-1]
        )

    remainder = len(data) % width

    if remainder:
        out[-remainder:] = data[-remainder:]

    return bytes(out)


def inverse_byteswap(data, original_size, width):
    # Byte swapping is its own inverse.
    return transform_byteswap(
        data,
        width,
    )[:original_size]


# ============================================================================
# FORMAT REGISTRY
# ============================================================================

FORMATS = {

    "raw": {
        "encode": transform_raw,
        "decode": inverse_raw,
    },

    "byte_delta": {
        "encode": transform_byte_delta,
        "decode": inverse_byte_delta,
    },

    "byte_xor": {
        "encode": transform_byte_xor,
        "decode": inverse_byte_xor,
    },

    "transpose_256": {
        "encode": lambda d:
            transform_transpose(d, 256),

        "decode": lambda d, n:
            inverse_transpose(d, 256, n),
    },

    "transpose_1024": {
        "encode": lambda d:
            transform_transpose(d, 1024),

        "decode": lambda d, n:
            inverse_transpose(d, 1024, n),
    },

    "transpose_4096": {
        "encode": lambda d:
            transform_transpose(d, 4096),

        "decode": lambda d, n:
            inverse_transpose(d, 4096, n),
    },

    "uint16_delta": {
        "encode": lambda d:
            transform_word_delta(d, 2),

        "decode": lambda d, n:
            inverse_word_delta(d, n, 2),
    },

    "uint32_delta": {
        "encode": lambda d:
            transform_word_delta(d, 4),

        "decode": lambda d, n:
            inverse_word_delta(d, n, 4),
    },

    "uint16_byteswap": {
        "encode": lambda d:
            transform_byteswap(d, 2),

        "decode": lambda d, n:
            inverse_byteswap(d, n, 2),
    },

    "uint32_byteswap": {
        "encode": lambda d:
            transform_byteswap(d, 4),

        "decode": lambda d, n:
            inverse_byteswap(d, n, 4),
    },
}


# ============================================================================
# BENCHMARK ONE FORMAT
# ============================================================================

def benchmark_format(
    original_path,
    format_name,
    format_spec,
    compressor,
    work_dir,
):
    original_size = file_size(original_path)

    original_hash = sha256_file(
        original_path
    )

    with open(original_path, "rb") as f:
        original_data = f.read()

    # ------------------------------------------------------------------------
    # Create representation
    # ------------------------------------------------------------------------

    t0 = time.perf_counter()

    represented = format_spec["encode"](
        original_data
    )

    representation_time = (
        time.perf_counter() - t0
    )

    represented_path = (
        work_dir / "represented.bin"
    )

    compressed_path = (
        work_dir / "compressed.bin"
    )

    decoded_representation_path = (
        work_dir / "decoded.bin"
    )

    reconstructed_path = (
        work_dir / "reconstructed.bin"
    )

    represented_path.write_bytes(
        represented
    )

    represented_size = len(
        represented
    )

    # ------------------------------------------------------------------------
    # Verify transformation itself
    # ------------------------------------------------------------------------

    t0 = time.perf_counter()

    reconstructed = format_spec["decode"](
        represented,
        original_size,
    )

    representation_decode_time = (
        time.perf_counter() - t0
    )

    representation_exact = (
        reconstructed == original_data
    )

    if not representation_exact:

        return {
            "format": format_name,
            "status": "REPRESENTATION_FAIL",
            "original_bytes": original_size,
            "representation_bytes": represented_size,
        }

    # ------------------------------------------------------------------------
    # Neural compression
    # ------------------------------------------------------------------------

    return_code, compress_time, stdout, stderr = (
        run_command(
            [
                sys.executable,
                str(compressor),
                "compress",
                str(represented_path),
                str(compressed_path),
            ]
        )
    )

    if return_code != 0:

        return {
            "format": format_name,
            "status": "COMPRESS_FAIL",
            "original_bytes": original_size,
            "representation_bytes": represented_size,
            "compress_seconds": compress_time,
            "error": stderr[-2000:],
        }

    # ------------------------------------------------------------------------
    # Neural decompression
    # ------------------------------------------------------------------------

    return_code, decompress_time, stdout2, stderr2 = (
        run_command(
            [
                sys.executable,
                str(compressor),
                "decompress",
                str(compressed_path),
                str(decoded_representation_path),
            ]
        )
    )

    if return_code != 0:

        return {
            "format": format_name,
            "status": "DECOMPRESS_FAIL",
            "original_bytes": original_size,
            "representation_bytes": represented_size,
            "compress_seconds": compress_time,
            "decompress_seconds": decompress_time,
            "error": stderr2[-2000:],
        }

    # ------------------------------------------------------------------------
    # Decode representation
    # ------------------------------------------------------------------------

    decoded_representation = (
        decoded_representation_path.read_bytes()
    )

    t0 = time.perf_counter()

    final_data = format_spec["decode"](
        decoded_representation,
        original_size,
    )

    inverse_time = (
        time.perf_counter() - t0
    )

    reconstructed_path.write_bytes(
        final_data
    )

    final_hash = sha256_file(
        reconstructed_path
    )

    exact = (
        final_hash == original_hash
        and final_data == original_data
    )

    compressed_size = file_size(
        compressed_path
    )

    total_ratio = (
        compressed_size / original_size
        if original_size
        else 0
    )

    representation_ratio = (
        represented_size / original_size
        if original_size
        else 0
    )

    return {
        "format": format_name,
        "status": "PASS" if exact else "HASH_FAIL",

        "original_bytes": original_size,

        "representation_bytes":
            represented_size,

        "compressed_bytes":
            compressed_size,

        "representation_ratio":
            representation_ratio,

        "compressed_ratio":
            total_ratio,

        "representation_percent":
            representation_ratio * 100,

        "compressed_percent":
            total_ratio * 100,

        "representation_seconds":
            representation_time,

        "representation_decode_seconds":
            representation_decode_time,

        "compress_seconds":
            compress_time,

        "decompress_seconds":
            decompress_time,

        "inverse_seconds":
            inverse_time,

        "original_sha256":
            original_hash,

        "reconstructed_sha256":
            final_hash,

        "exact":
            exact,
    }


# ============================================================================
# BENCHMARK ONE FILE
# ============================================================================

def benchmark_file(
    original_path,
    compressor,
    output_dir,
):
    print()
    print("=" * 80)

    print(
        f"FILE: {original_path}"
    )

    print(
        f"SIZE: {format_size(file_size(original_path))}"
    )

    print("=" * 80)

    file_name = original_path.name

    file_result_dir = (
        output_dir / file_name
    )

    file_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for format_name, format_spec in FORMATS.items():

        print()
        print(
            f"[+] Testing {format_name} ..."
        )

        format_dir = (
            file_result_dir / format_name
        )

        format_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            result = benchmark_format(
                original_path,
                format_name,
                format_spec,
                compressor,
                format_dir,
            )

        except Exception as exc:

            result = {
                "format": format_name,
                "status": "ERROR",
                "error": repr(exc),
            }

        results.append(result)

        status = result.get(
            "status"
        )

        if status == "PASS":

            print(
                f"    PASS | "
                f"representation="
                f"{format_size(result['representation_bytes'])} | "
                f"compressed="
                f"{format_size(result['compressed_bytes'])} | "
                f"{result['compressed_ratio'] * 100:.2f}%"
            )

            print(
                f"    encode="
                f"{result['compress_seconds']:.2f}s | "
                f"decode="
                f"{result['decompress_seconds']:.2f}s"
            )

        else:

            print(
                f"    {status}"
            )

            if "error" in result:

                print(
                    f"    {result['error']}"
                )

    # ------------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------------

    successful = [
        r
        for r in results
        if r.get("status") == "PASS"
    ]

    successful.sort(
        key=lambda r:
            r["compressed_bytes"]
    )

    print()
    print("-" * 80)
    print("RANKING")
    print("-" * 80)

    for rank, result in enumerate(
        successful,
        1,
    ):

        print(
            f"{rank:2d}. "
            f"{result['format']:20s} "
            f"{format_size(result['compressed_bytes']):>12s} "
            f"{result['compressed_ratio'] * 100:7.2f}% "
            f"encode="
            f"{result['compress_seconds']:8.2f}s "
            f"decode="
            f"{result['decompress_seconds']:8.2f}s"
        )

    with open(
        file_result_dir / "results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reversible data "
            "representations with "
            "neural_compress_np.py"
        )
    )

    parser.add_argument(
        "directory",
        help="Directory containing files to test",
    )

    parser.add_argument(
        "--compressor",
        default="./neural_compress_np.py",
        help="Path to neural_compress_np.py",
    )

    parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Benchmark output directory",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help=(
            "Maximum file size in bytes. "
            "0 = unlimited."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of files. "
            "0 = unlimited."
        ),
    )

    args = parser.parse_args()

    corpus_dir = Path(
        args.directory
    ).resolve()

    compressor = Path(
        args.compressor
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    if not corpus_dir.is_dir():

        print(
            f"[!] Not a directory: "
            f"{corpus_dir}"
        )

        sys.exit(1)

    if not compressor.exists():

        print(
            f"[!] Compressor not found: "
            f"{compressor}"
        )

        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Find files
    # ------------------------------------------------------------------------

    files = []

    for path in corpus_dir.rglob("*"):

        if not path.is_file():
            continue

        size = file_size(path)

        if (
            args.max_size
            and size > args.max_size
        ):
            continue

        files.append(path)

    files.sort(
        key=lambda p:
            file_size(p)
    )

    if args.limit:
        files = files[:args.limit]

    print()
    print("=" * 80)
    print("LOSSLESS SYMBOL FORMAT BENCHMARK")
    print("=" * 80)

    print(
        f"Corpus     : {corpus_dir}"
    )

    print(
        f"Compressor : {compressor}"
    )

    print(
        f"Files      : {len(files)}"
    )

    print(
        f"Output     : {output_dir}"
    )

    print("=" * 80)

    if not files:

        print(
            "[!] No files selected."
        )

        return

    all_results = []

    overall_start = (
        time.perf_counter()
    )

    # ------------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------------

    for index, path in enumerate(
        files,
        1,
    ):

        print()

        print(
            f">>> FILE {index}/{len(files)}: "
            f"{path.relative_to(corpus_dir)}"
        )

        results = benchmark_file(
            path,
            compressor,
            output_dir / "per_file",
        )

        for result in results:

            result["file"] = str(
                path.relative_to(
                    corpus_dir
                )
            )

        all_results.extend(
            results
        )

    overall_time = (
        time.perf_counter()
        - overall_start
    )

    # ------------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------------

    csv_path = (
        output_dir / "results.csv"
    )

    fields = [
        "file",
        "format",
        "status",
        "original_bytes",
        "representation_bytes",
        "compressed_bytes",
        "representation_ratio",
        "compressed_ratio",
        "representation_percent",
        "compressed_percent",
        "representation_seconds",
        "representation_decode_seconds",
        "compress_seconds",
        "decompress_seconds",
        "inverse_seconds",
        "exact",
        "original_sha256",
        "reconstructed_sha256",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in all_results:

            writer.writerow(
                result
            )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    json_path = (
        output_dir / "results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("CORPUS RESULTS")
    print("=" * 80)

    for format_name in FORMATS:

        results = [
            r
            for r in all_results
            if r.get("format")
            == format_name
            and r.get("status")
            == "PASS"
        ]

        if not results:
            continue

        original_total = sum(
            r["original_bytes"]
            for r in results
        )

        compressed_total = sum(
            r["compressed_bytes"]
            for r in results
        )

        representation_total = sum(
            r["representation_bytes"]
            for r in results
        )

        compress_time = sum(
            r["compress_seconds"]
            for r in results
        )

        decompress_time = sum(
            r["decompress_seconds"]
            for r in results
        )

        ratio = (
            compressed_total
            / original_total
            if original_total
            else 0
        )

        print(
            f"{format_name:20s} "
            f"original="
            f"{format_size(original_total):>12s} "
            f"represented="
            f"{format_size(representation_total):>12s} "
            f"compressed="
            f"{format_size(compressed_total):>12s} "
            f"ratio="
            f"{ratio * 100:7.2f}% "
            f"enc="
            f"{compress_time:9.2f}s "
            f"dec="
            f"{decompress_time:9.2f}s"
        )

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print(
        f"Total runtime : "
        f"{overall_time:.2f}s"
    )

    print(
        f"CSV results   : "
        f"{csv_path}"
    )

    print(
        f"JSON results  : "
        f"{json_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
