"""Command-line interface for the E40 adaptive residual codec."""
from __future__ import annotations

import argparse
from pathlib import Path

from E40_adaptive_residual_codec import decode, encode


def _print_stats(stats) -> None:
    print(f"input={stats.input_bytes}")
    print(f"encoded={stats.encoded_bytes}")
    print(f"ratio={stats.ratio:.4f}")
    print(f"reduction={stats.reduction_percent:.2f}%")
    print(f"regions={stats.region_count}")
    print("selected=" + ", ".join(
        f"{name}:{count}" for name, count in stats.representation_counts.items()
    ))
    if stats.fixed_total_costs:
        print("fixed_total=" + ", ".join(
            f"{name}:{cost}" for name, cost in stats.fixed_total_costs.items()
        ))
        best_fixed = min(stats.fixed_total_costs.values())
        print(f"best_fixed={best_fixed}")
        print(f"adaptive_vs_best_fixed={stats.encoded_bytes - best_fixed}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E40 exact adaptive residual codec"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser(
        "encode", help="encode a file into an NC40 stream"
    )
    encode_parser.add_argument("input", type=Path, help="input file")
    encode_parser.add_argument("output", type=Path, help="output NC40 file")
    encode_parser.add_argument(
        "--region-size", type=int, default=1024,
        help="bytes per region (default: 1024)",
    )
    encode_parser.add_argument(
        "--no-stats", action="store_true",
        help="skip fixed-strategy comparison",
    )

    decode_parser = subparsers.add_parser(
        "decode", help="decode an NC40 stream back to its original file"
    )
    decode_parser.add_argument("input", type=Path, help="input NC40 file")
    decode_parser.add_argument("output", type=Path, help="restored output file")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.command == "encode":
        data = args.input.read_bytes()
        if args.no_stats:
            blob = encode(data, region_size=args.region_size)
            stats = None
        else:
            blob, stats = encode(
                data, region_size=args.region_size, return_stats=True
            )
        args.output.write_bytes(blob)
        print(f"encoded_file={args.output}")
        if stats is not None:
            _print_stats(stats)
        return 0

    data = decode(args.input.read_bytes())
    args.output.write_bytes(data)
    print(f"decoded_file={args.output}")
    print(f"output_bytes={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
