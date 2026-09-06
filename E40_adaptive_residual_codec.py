"""E40 adaptive residual codec: real model coordination with local representations.

The codec is a predictive layer that can sit beside a base model/codec:
the Predictor supplies a deterministic prediction, E40 measures the residual,
chooses the cheapest complete representation for each region, and serializes
the choice plus the local structural signature needed to reproduce it.

Exact reconstruction is mandatory. No oracle or precomputed strategy map is
used during encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import math
import struct
import zlib

MAGIC = b"NC40"
VERSION = 2
FILE_HEADER = struct.Struct("<4sBII")
REGION_HEADER = struct.Struct("<BII3B")


@dataclass(frozen=True)
class RegionSignature:
    """Quantized local behavior, independent of absolute position."""

    entropy_q: int
    repetition_q: int
    unique_q: int

    def pack(self) -> bytes:
        return bytes((self.entropy_q, self.repetition_q, self.unique_q))

    @classmethod
    def unpack(cls, data: bytes) -> "RegionSignature":
        if len(data) != 3:
            raise ValueError("invalid region signature")
        return cls(*data)


@dataclass(frozen=True)
class Representation:
    """A representation defines a maximum predictive capacity and coding mode."""

    name: str
    context_size: int
    mode: str
    compression_level: int


# These are genuinely different predictive representations.
REPRESENTATIONS = (
    Representation("compact", 1, "previous", 1),
    Representation("balanced", 2, "delta", 6),
    Representation("deep", 3, "second_order", 9),
    Representation("raw", 0, "zero", 6),
)


class Predictor(Protocol):
    """Pluggable model interface.

    predict() must be deterministic for the supplied model state and arguments.
    observe() is called once, after the winning representation is selected.
    This lets a stateful external model evolve identically at both ends without
    being updated by losing candidate evaluations.
    """

    def predict(
        self,
        context: bytes,
        length: int,
        position: int,
        representation: Representation,
        signature: RegionSignature,
    ) -> bytes:
        ...

    def observe(
        self,
        region: bytes,
        position: int,
        representation: Representation,
        signature: RegionSignature,
    ) -> None:
        ...


class ContextPredictor:
    """Small deterministic reference model for E40.

    The signature controls effective context capacity. This makes the local
    structural representation part of the actual prediction path while keeping
    the predictor simple enough that E40's mechanics remain visible.
    """

    def _effective_context(
        self, representation: Representation, signature: RegionSignature
    ) -> int:
        if representation.context_size == 0:
            return 0
        if signature.repetition_q >= 192:
            return 1
        if signature.entropy_q < 96 and signature.unique_q < 64:
            return min(2, representation.context_size)
        return representation.context_size

    def predict(self, context, length, position, representation, signature):
        effective_context = self._effective_context(representation, signature)
        history = bytearray(context[-effective_context:])
        output = bytearray()

        for _ in range(length):
            if representation.mode == "zero" or not history:
                value = 0
            elif representation.mode == "previous":
                value = history[-1]
            elif representation.mode == "delta":
                value = (
                    (2 * history[-1] - history[-2]) & 0xFF
                    if len(history) >= 2
                    else history[-1]
                )
            elif representation.mode == "second_order":
                value = (
                    (3 * history[-1] - 3 * history[-2] + history[-3]) & 0xFF
                    if len(history) >= 3
                    else history[-1]
                )
            else:
                raise ValueError(f"unknown representation mode: {representation.mode}")

            output.append(value)
            history.append(value)

        return bytes(output)

    def observe(self, region, position, representation, signature):
        return None


def analyze_region(data: bytes) -> RegionSignature:
    """Describe local behavior without using absolute location."""

    if not data:
        return RegionSignature(0, 255, 0)

    counts = [0] * 256
    for value in data:
        counts[value] += 1

    n = len(data)
    entropy = -sum(
        (count / n) * math.log2(count / n) for count in counts if count
    )
    repetition = sum(a == b for a, b in zip(data, data[1:])) / max(1, n - 1)
    unique_fraction = sum(count > 0 for count in counts) / 256

    return RegionSignature(
        round(entropy / 8 * 255),
        round(repetition * 255),
        round(unique_fraction * 255),
    )


def _xor(actual: bytes, predicted: bytes) -> bytes:
    if len(actual) != len(predicted):
        raise ValueError("prediction length does not match region length")
    return bytes(a ^ b for a, b in zip(actual, predicted))


def _predict(
    predictor: Predictor,
    previous: bytes,
    length: int,
    position: int,
    representation: Representation,
    signature: RegionSignature,
) -> bytes:
    predicted = predictor.predict(
        previous,
        length,
        position,
        representation,
        signature,
    )
    if len(predicted) != length:
        raise ValueError("predictor returned incorrect region length")
    return predicted


def candidate_cost(coded_length: int) -> int:
    """Complete encoded cost for one candidate, including region metadata."""
    return REGION_HEADER.size + coded_length


def _encode_region_candidates(
    region: bytes,
    previous: bytes,
    position: int,
    predictor: Predictor,
    signature: RegionSignature,
):
    candidates = []
    for rid, representation in enumerate(REPRESENTATIONS):
        prediction = _predict(
            predictor, previous, len(region), position, representation, signature
        )
        residual = _xor(region, prediction)
        coded = zlib.compress(residual, representation.compression_level)
        candidates.append(
            {
                "rid": rid,
                "representation": representation,
                "coded": coded,
                "residual_length": len(residual),
                "cost": candidate_cost(len(coded)),
            }
        )
    return candidates


@dataclass
class EncodeStats:
    input_bytes: int
    encoded_bytes: int
    region_count: int
    representation_counts: dict[str, int]
    adaptive_region_cost: int
    fixed_region_costs: dict[str, int]
    signatures: list[RegionSignature]

    @property
    def ratio(self) -> float:
        return self.encoded_bytes / max(1, self.input_bytes)

    @property
    def reduction_percent(self) -> float:
        if not self.input_bytes:
            return 0.0
        return (1.0 - self.encoded_bytes / self.input_bytes) * 100.0

    @property
    def fixed_total_costs(self) -> dict[str, int]:
        return {
            name: FILE_HEADER.size + cost
            for name, cost in self.fixed_region_costs.items()
        }


def _measure_fixed(
    data: bytes,
    region_size: int,
    predictor_factory: Callable[[], Predictor],
) -> dict[str, int]:
    """Measure true fixed-representation streams with independent model state."""

    totals = {representation.name: 0 for representation in REPRESENTATIONS}

    for _, representation in enumerate(REPRESENTATIONS):
        predictor = predictor_factory()
        previous = b""

        for position, start in enumerate(range(0, len(data), region_size)):
            region = data[start:start + region_size]
            signature = analyze_region(region)
            prediction = _predict(
                predictor, previous, len(region), position, representation, signature
            )
            residual = _xor(region, prediction)
            coded = zlib.compress(residual, representation.compression_level)
            totals[representation.name] += candidate_cost(len(coded))
            predictor.observe(region, position, representation, signature)
            previous = region

    return totals


def encode(
    data: bytes,
    predictor: Predictor | None = None,
    region_size: int = 1024,
    *,
    return_stats: bool = False,
    predictor_factory: Callable[[], Predictor] | None = None,
):
    """Encode with measured adaptive representation selection.

    If return_stats=True, fixed-strategy comparison is also produced. For a
    custom stateful predictor, pass predictor_factory so each fixed strategy
    gets independent model state. The built-in ContextPredictor is handled
    automatically.
    """

    if region_size <= 0:
        raise ValueError("region_size must be positive")

    predictor = predictor or ContextPredictor()
    region_count = (len(data) + region_size - 1) // region_size
    out = bytearray(FILE_HEADER.pack(MAGIC, VERSION, region_size, region_count))
    previous = b""

    counts = {representation.name: 0 for representation in REPRESENTATIONS}
    signatures: list[RegionSignature] = []
    adaptive_region_cost = 0

    for position, start in enumerate(range(0, len(data), region_size)):
        region = data[start:start + region_size]
        signature = analyze_region(region)
        signatures.append(signature)

        candidates = _encode_region_candidates(
            region, previous, position, predictor, signature
        )

        # No oracle: all real candidates are measured and the complete
        # end-to-end cost (metadata + payload) determines the winner.
        winner = min(
            candidates,
            key=lambda candidate: (candidate["cost"], candidate["rid"]),
        )
        representation = winner["representation"]

        out += REGION_HEADER.pack(
            winner["rid"],
            len(winner["coded"]),
            winner["residual_length"],
            *signature.pack(),
        )
        out += winner["coded"]

        counts[representation.name] += 1
        adaptive_region_cost += winner["cost"]

        # Update the shared/base model only with the actual selected path.
        predictor.observe(region, position, representation, signature)
        previous = region

    blob = bytes(out)

    if not return_stats:
        return blob

    if predictor_factory is None and isinstance(predictor, ContextPredictor):
        predictor_factory = ContextPredictor

    fixed_costs = (
        _measure_fixed(data, region_size, predictor_factory)
        if predictor_factory is not None
        else {}
    )

    return blob, EncodeStats(
        input_bytes=len(data),
        encoded_bytes=len(blob),
        region_count=region_count,
        representation_counts=counts,
        adaptive_region_cost=adaptive_region_cost,
        fixed_region_costs=fixed_costs,
        signatures=signatures,
    )


def decode(blob: bytes, predictor: Predictor | None = None) -> bytes:
    """Decode using only the serialized stream and synchronized predictor state."""

    predictor = predictor or ContextPredictor()

    if len(blob) < FILE_HEADER.size:
        raise ValueError("invalid NC40 stream")

    magic, version, region_size, count = FILE_HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("invalid NC40 stream")
    if version != VERSION:
        raise ValueError("unsupported NC40 version")
    if region_size <= 0:
        raise ValueError("invalid region size")

    pointer = FILE_HEADER.size
    out = bytearray()
    previous = b""

    for position in range(count):
        if pointer + REGION_HEADER.size > len(blob):
            raise ValueError("truncated region header")

        rid, compressed_length, raw_length, entropy_q, repetition_q, unique_q = (
            REGION_HEADER.unpack_from(blob, pointer)
        )
        pointer += REGION_HEADER.size

        if rid >= len(REPRESENTATIONS):
            raise ValueError("invalid representation id")
        if raw_length <= 0 or raw_length > region_size:
            raise ValueError("invalid region raw length")
        if position < count - 1 and raw_length != region_size:
            raise ValueError("non-final region has incorrect length")
        if compressed_length > len(blob) - pointer:
            raise ValueError("truncated region payload")

        signature = RegionSignature(entropy_q, repetition_q, unique_q)
        coded = blob[pointer:pointer + compressed_length]
        pointer += compressed_length

        try:
            residual = zlib.decompress(coded)
        except zlib.error as exc:
            raise ValueError("invalid compressed residual") from exc

        if len(residual) != raw_length:
            raise ValueError("residual length mismatch")

        representation = REPRESENTATIONS[rid]
        prediction = _predict(
            predictor, previous, raw_length, position, representation, signature
        )
        region = _xor(residual, prediction)

        predictor.observe(region, position, representation, signature)
        out += region
        previous = region

    if pointer != len(blob):
        raise ValueError("trailing bytes")

    if count != 0 and len(out) == 0:
        raise ValueError("invalid empty stream")

    return bytes(out)


def roundtrip(data: bytes, predictor: Predictor | None = None, region_size: int = 1024):
    blob = encode(data, predictor, region_size)
    decoded = decode(blob, predictor)
    if decoded != data:
        raise AssertionError("exact reconstruction failed")
    return blob, len(blob) / max(1, len(data))


def _demo():
    """Run a mixed local-structure smoke corpus and report useful metrics."""

    repeated = b"neural compression: persistent structure. " * 300
    linear = bytes((i * 3 + 7) % 256 for i in range(4096))
    quadratic = bytes((i * i + 11) % 256 for i in range(4096))
    rng = __import__("random").Random(12345)
    noisy = bytes(rng.randrange(256) for _ in range(4096))
    data = repeated + linear + quadratic + noisy

    blob, stats = encode(data, region_size=512, return_stats=True)
    exact = decode(blob) == data

    print(f"input={stats.input_bytes}")
    print(f"encoded={stats.encoded_bytes}")
    print(f"ratio={stats.ratio:.4f}")
    print(f"reduction={stats.reduction_percent:.2f}%")
    print(f"regions={stats.region_count}")
    print(f"exact={exact}")
    print("selected=" + ", ".join(
        f"{name}:{count}" for name, count in stats.representation_counts.items()
    ))
    print("fixed_total=" + ", ".join(
        f"{name}:{cost}" for name, cost in stats.fixed_total_costs.items()
    ))
    if stats.fixed_total_costs:
        best_fixed = min(stats.fixed_total_costs.values())
        adaptive_total = stats.encoded_bytes
        print(f"best_fixed={best_fixed}")
        print(f"adaptive_vs_best_fixed={adaptive_total - best_fixed}")


if __name__ == "__main__":
    _demo()
