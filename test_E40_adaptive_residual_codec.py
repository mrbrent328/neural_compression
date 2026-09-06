"""Executable E40 validation suite.

Run directly with:
    python3 test_E40_adaptive_residual_codec.py
"""
from E40_adaptive_residual_codec import (
    ContextPredictor,
    FILE_HEADER,
    REGION_HEADER,
    REPRESENTATIONS,
    analyze_region,
    candidate_cost,
    decode,
    encode,
)


def test_exact_roundtrip():
    cases = (
        b"",
        bytes(range(256)) * 20,
        b"AAAAABBBBBCCCCCDDDD" * 1000,
        bytes((i * 37 + 11) % 256 for i in range(10000)),
    )
    for data in cases:
        blob = encode(data, region_size=257)
        assert decode(blob) == data


def test_mixed_regions_use_real_adaptation():
    repeated = b"neural compression: persistent structure. " * 300
    linear = bytes((i * 3 + 7) % 256 for i in range(4096))
    quadratic = bytes((i * i + 11) % 256 for i in range(4096))
    import random
    rng = random.Random(12345)
    noisy = bytes(rng.randrange(256) for _ in range(4096))
    data = repeated + linear + quadratic + noisy

    blob, stats = encode(data, region_size=512, return_stats=True)

    assert decode(blob) == data
    assert sum(stats.representation_counts.values()) == stats.region_count
    assert len([n for n in stats.representation_counts.values() if n]) >= 2
    assert stats.encoded_bytes == FILE_HEADER.size + stats.adaptive_region_cost
    assert stats.fixed_total_costs
    assert stats.encoded_bytes < min(stats.fixed_total_costs.values())


def test_signature_is_local():
    region = bytes((i * 11 + 3) % 256 for i in range(1024))
    assert analyze_region(region) == analyze_region(region)
    assert analyze_region(region) != analyze_region(b"A" * 1024)


def test_complete_cost_includes_metadata():
    assert candidate_cost(0) == REGION_HEADER.size
    assert candidate_cost(100) == REGION_HEADER.size + 100


def test_stateful_predictor_stays_synchronized():
    class RecordingPredictor(ContextPredictor):
        def __init__(self):
            self.events = []

        def observe(self, region, position, representation, signature):
            self.events.append((position, representation.name, signature))

    data = (
        bytes((i * 3 + 7) % 256 for i in range(4096))
        + bytes((i * i + 11) % 256 for i in range(4096))
    )
    encoder_predictor = RecordingPredictor()
    decoder_predictor = RecordingPredictor()

    blob = encode(data, encoder_predictor, region_size=512)
    assert decode(blob, decoder_predictor) == data
    assert encoder_predictor.events == decoder_predictor.events
    assert len(encoder_predictor.events) == 16


def test_trailing_bytes_rejected():
    blob = encode(b"persistent structure" * 100)
    try:
        decode(blob + b"x")
    except ValueError as exc:
        assert "trailing" in str(exc)
    else:
        raise AssertionError("trailing bytes accepted")


def test_invalid_representation_rejected():
    blob = bytearray(encode(b"persistent structure" * 100))
    blob[FILE_HEADER.size] = len(REPRESENTATIONS)
    try:
        decode(bytes(blob))
    except ValueError as exc:
        assert "representation" in str(exc)
    else:
        raise AssertionError("invalid representation accepted")


def test_truncated_payload_rejected():
    blob = encode(b"persistent structure" * 100)
    try:
        decode(blob[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("truncated payload accepted")


def run_all():
    tests = (
        test_exact_roundtrip,
        test_mixed_regions_use_real_adaptation,
        test_signature_is_local,
        test_complete_cost_includes_metadata,
        test_stateful_predictor_stays_synchronized,
        test_trailing_bytes_rejected,
        test_invalid_representation_rejected,
        test_truncated_payload_rejected,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    run_all()
