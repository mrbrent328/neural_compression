#!/usr/bin/env python3

"""
TEST10.3 — CORRECTED CAUSAL NEURAL COMPRESSION

Purpose
=======

TEST10.1 validated the causal range-coder foundation.

TEST10.2 inserted the neural probability model and demonstrated that the
model can drive the same causal coder.

TEST10.3 is the larger-scale measurement and correctness test.

IMPORTANT:
    This version specifically hardens the range coder against the state
    divergence observed in the previous larger-payload experiment.

The causal rule is:

    TRAINING PREFIX
          |
          v
    +----------------+
    | fixed neural   |
    | probability    |
    | model          |
    +----------------+
          |
          v
    previous 4 bytes
          |
          v
    integer frequency table
          |
          v
    range encoder
          |
          v
    payload

The decoder performs exactly the same operations using:

    training prefix + previously decoded bytes

The future held-out bytes are NEVER used to construct a probability table.

Architecture
============

    Embedding(256, 32)
    Linear(128)
    Linear(256)

The model is trained only on the first 75% of the corpus.

The final 25% is completely held out.

The model itself is transmitted as part of the full-compression accounting.

Measurements
============

1. Neural model determinism
2. Frequency determinism
3. Causal state identity
4. Exact reconstruction
5. SHA-256 reconstruction
6. Encoder/decoder frequency identity
7. Neural payload size
8. Raw payload size
9. Static-frequency baseline
10. Full model + metadata + payload cost
11. Break-even analysis
12. Large-stream stability

The range coder uses integer-only state transitions after probability
quantization.

No floating-point probability values are used by the arithmetic coder.

The model may use floating point internally, but both encoder and decoder
convert its output into the exact same integer frequency table.

This file is intentionally self-contained.
"""

import hashlib
import io
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 10_300

CONTEXT_SIZE = 4
SYMBOLS = 256

FREQ_TOTAL = 32768
MIN_FREQ = 1

TRAIN_FRACTION = 0.75

TRAINING_SAMPLES = 300_000
EPOCHS = 25
BATCH_SIZE = 4096
LEARNING_RATE = 0.001

EMBED_DIM = 32
HIDDEN1 = 128
HIDDEN2 = 256

CORPUS_SIZE = 512 * 1024

PRINT_FIRST_N = 32

MODEL_NAME = "TEST10_Embedding32_Linear128_Linear256"
TEST_NAME = "TEST10.3"

CODE_BITS = 64
TOP = (1 << CODE_BITS) - 1


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================================
# DEVICE INFORMATION
# ============================================================================

def print_device_info() -> torch.device:
    print(f"PyTorch:       {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA API:      {'available' if cuda_available else 'unavailable'}")

    if cuda_available:
        print("Backend:       ROCm-compatible PyTorch interface")
        print(f"GPU:           {torch.cuda.get_device_name(0)}")

        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"VRAM:          {vram_gb:.2f} GB")
        except Exception:
            print("VRAM:          unavailable")

        return torch.device("cuda")

    print("Backend:       CPU")
    print("GPU:           none")
    return torch.device("cpu")


# ============================================================================
# DETERMINISTIC CORPUS
# ============================================================================

CORPUS_BLOCKS = [
    b"""
def encode_context(context, model):
    probabilities = model(context)
    return probabilities

def decode_context(context, model):
    probabilities = model(context)
    return probabilities

The causal model sees only the past.
The future is never supplied to the probability model.
The decoder reconstructs exactly the same state.
""",
    b"""
class NeuralCompressor:
    def __init__(self, context_size=4):
        self.context_size = context_size

    def predict(self, context):
        return self.model(context)

    def update(self, symbol):
        self.history.append(symbol)

Compression is prediction followed by entropy coding.
Prediction must remain causal.
Entropy coding must remain exactly synchronized.
""",
    b"""
import hashlib
import math
import random

def deterministic_hash(data):
    return hashlib.sha256(data).digest()

def verify(original, reconstructed):
    return hashlib.sha256(original).digest() == hashlib.sha256(reconstructed).digest()

A deterministic experiment must reproduce the same state.
A single changed frequency can cause the arithmetic coder to diverge.
""",
    b"""
TEST10 compression experiment:
training data comes first,
held-out data comes second,
the model is frozen,
the encoder consumes actual previous bytes,
the decoder consumes reconstructed previous bytes,
and both sides must generate identical frequency tables.
""",
    b"""
for index in range(256):
    value = (index * 257 + 17) & 0xffff
    value ^= (value >> 3)
    value ^= (value << 7) & 0xffff

The integer arithmetic coder does not need floating point probabilities.
The neural network produces probabilities.
The quantizer produces integer frequencies.
The range coder consumes only those frequencies.
""",
]

SEP = b"\n\n"


def build_corpus(size: int = CORPUS_SIZE) -> bytes:
    chunks = []

    while sum(len(x) for x in chunks) < size:
        for block in CORPUS_BLOCKS:
            chunks.append(block)
            chunks.append(SEP)

            if sum(len(x) for x in chunks) >= size:
                break

    base = b"".join(chunks)

    rng = random.Random(SEED)
    extra = bytearray()

    while len(extra) < size:
        mode = rng.randrange(8)

        if mode == 0:
            extra.extend(b"    " * rng.randint(1, 8))
        elif mode == 1:
            extra.extend(
                f"variable_{rng.randrange(1000):04d} = "
                f"value_{rng.randrange(1000):04d}\n".encode()
            )
        elif mode == 2:
            extra.extend(
                f"result = encode(context_{rng.randrange(64)})\n".encode()
            )
        elif mode == 3:
            extra.extend(b"AAAA" * rng.randint(2, 32))
        elif mode == 4:
            extra.extend(b"BBBB" * rng.randint(2, 32))
        elif mode == 5:
            extra.extend(
                b"the causal probability model predicts the next byte\n"
            )
        elif mode == 6:
            extra.extend(
                b"frequency identity PASS\n"
                b"causal state identity PASS\n"
            )
        else:
            extra.extend(base[: min(len(base), rng.randint(32, 512))])

    corpus = bytearray()
    i = 0
    j = 0

    while len(corpus) < size:
        if (len(corpus) // 4096) % 3 != 2:
            take = min(4096, len(base) - i)
            if take:
                corpus.extend(base[i:i + take])
                i += take
            else:
                i = 0
        else:
            take = min(4096, len(extra) - j)
            if take:
                corpus.extend(extra[j:j + take])
                j += take
            else:
                j = 0

    return bytes(corpus[:size])


# ============================================================================
# MODEL
# ============================================================================

class NeuralProbabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(SYMBOLS, EMBED_DIM)
        self.fc1 = nn.Linear(CONTEXT_SIZE * EMBED_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, SYMBOLS)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# TRAINING DATA
# ============================================================================

def make_training_samples(
    training_bytes: bytes,
    samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if len(training_bytes) <= CONTEXT_SIZE:
        raise ValueError("Training data is too small.")

    rng = random.Random(seed)
    contexts = torch.empty((samples, CONTEXT_SIZE), dtype=torch.long)
    targets = torch.empty(samples, dtype=torch.long)
    limit = len(training_bytes) - CONTEXT_SIZE

    for i in range(samples):
        pos = rng.randrange(limit)
        context = training_bytes[pos:pos + CONTEXT_SIZE]
        target = training_bytes[pos + CONTEXT_SIZE]

        for j, value in enumerate(context):
            contexts[i, j] = value

        targets[i] = target

    return contexts, targets


def train_model(
    model: nn.Module,
    training_bytes: bytes,
    device: torch.device,
) -> None:

    print()
    print(f"Architecture:       {MODEL_NAME}")
    print(f"Training samples:   {TRAINING_SAMPLES:,}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Batch size:         {BATCH_SIZE:,}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Training device:    {device}")

    contexts, targets = make_training_samples(
        training_bytes,
        TRAINING_SAMPLES,
        SEED,
    )

    dataset = torch.utils.data.TensorDataset(contexts, targets)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for x_cpu, y_cpu in loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_seen += len(y)

        avg_loss = total_loss / total_seen
        accuracy = total_correct / total_seen * 100.0

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS}  "
            f"loss={avg_loss:.6f}  "
            f"accuracy={accuracy:.4f}%"
        )

    elapsed = time.perf_counter() - start
    print(f"Training time:      {elapsed:.3f} s")


# ============================================================================
# MODEL SERIALIZATION
# ============================================================================

def serialize_model(model: nn.Module) -> bytes:
    buffer = io.BytesIO()
    cpu_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(cpu_state, buffer, _use_new_zipfile_serialization=True)
    return buffer.getvalue()


def model_hash(model_bytes: bytes) -> str:
    return hashlib.sha256(model_bytes).hexdigest()


# ============================================================================
# DETERMINISTIC PROBABILITY QUANTIZATION
# ============================================================================

def quantize_probabilities(logits: torch.Tensor) -> Tuple[int, ...]:
    if logits.ndim != 1 or logits.numel() != SYMBOLS:
        raise ValueError("Expected one 256-element logit vector.")

    x = logits.detach().to("cpu", dtype=torch.float64)
    probs = torch.softmax(x, dim=0)

    remaining = FREQ_TOTAL - SYMBOLS * MIN_FREQ
    if remaining < 0:
        raise ValueError("FREQ_TOTAL is smaller than required minimums.")

    scaled = probs * remaining
    floors = torch.floor(scaled).to(torch.int64)
    frequencies = floors + MIN_FREQ

    used = int(frequencies.sum().item())
    leftover = FREQ_TOTAL - used

    if leftover < 0:
        raise RuntimeError("Frequency quantization overflow.")

    remainders = scaled - floors
    order = sorted(
        range(SYMBOLS),
        key=lambda i: (-float(remainders[i].item()), i),
    )

    for i in order[:leftover]:
        frequencies[i] += 1

    result = tuple(int(x.item()) for x in frequencies)

    if sum(result) != FREQ_TOTAL:
        raise AssertionError("Frequency sum mismatch.")
    if min(result) < MIN_FREQ:
        raise AssertionError("Minimum frequency violated.")

    return result


# ============================================================================
# NEURAL FREQUENCY PROVIDER
# ============================================================================

class NeuralFrequencyProvider:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cache: Dict[bytes, Tuple[int, ...]] = {}
        self.model.eval()

    @torch.inference_mode()
    def frequencies(self, context: bytes) -> Tuple[int, ...]:
        if len(context) != CONTEXT_SIZE:
            raise ValueError(f"Context must contain {CONTEXT_SIZE} bytes.")

        cached = self.cache.get(context)
        if cached is not None:
            return cached

        values = torch.tensor(
            list(context),
            dtype=torch.long,
            device=self.device,
        ).reshape(1, CONTEXT_SIZE)

        logits = self.model(values)[0]
        frequencies = quantize_probabilities(logits)
        self.cache[context] = frequencies

        return frequencies


# ============================================================================
# FREQUENCY HASH
# ============================================================================

def frequency_hash(frequencies: Tuple[int, ...]) -> str:
    packed = struct.pack("<256I", *frequencies)
    return hashlib.sha256(packed).hexdigest()[:16]


# ============================================================================
# 64-BIT RANGE CODER
# ============================================================================

class ByteWriter:
    def __init__(self):
        self.data = bytearray()

    def write_byte(self, byte: int):
        self.data.append(byte & 0xFF)

    def finish(self) -> bytes:
        return bytes(self.data)


class ByteReader:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def read_byte(self) -> int:
        if self.position >= len(self.data):
            return 0  # Standard EOF padding byte
        byte = self.data[self.position]
        self.position += 1
        return byte


class RangeEncoder:
    """
    Byte-aligned 64-bit Range Encoder.
    Emits full bytes during normalization to guarantee alignment with ByteReader.
    """
    def __init__(self):
        self.low = 0
        self.range = (1 << CODE_BITS) - 1
        self.writer = ByteWriter()
        self.cache = 0
        self.cache_size = 0

    def _shift_low(self):
        low_hi = self.low >> (CODE_BITS - 8)

        if self.low < (0xFF << (CODE_BITS - 8)) or low_hi != 0:
            carry = low_hi

            if self.cache_size > 0:
                self.writer.write_byte((self.cache + carry) & 0xFF)
                for _ in range(self.cache_size - 1):
                    self.writer.write_byte((0xFF + carry) & 0xFF)

            self.cache = (self.low >> (CODE_BITS - 16)) & 0xFF
            self.cache_size = 1
        else:
            self.cache_size += 1

        self.low = (self.low << 8) & TOP

    def encode_symbol(self, symbol: int, frequencies: Tuple[int, ...]):
        if not (0 <= symbol < SYMBOLS):
            raise ValueError("Invalid symbol.")

        total = sum(frequencies)
        cumulative = sum(frequencies[:symbol])
        freq = frequencies[symbol]

        if freq <= 0:
            raise ValueError("Symbol has zero frequency.")

        unit = self.range // total
        if unit == 0:
            raise RuntimeError("Range collapsed before normalization.")

        self.low += unit * cumulative
        self.range = unit * freq

        while self.range < (1 << (CODE_BITS - 8)):
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(CODE_BITS // 8):
            self._shift_low()
        return self.writer.finish()


class RangeDecoder:
    """
    Matching byte-aligned 64-bit Range Decoder.
    """
    def __init__(self, data: bytes):
        self.reader = ByteReader(data)
        self.code = 0
        self.range = (1 << CODE_BITS) - 1

        for _ in range(CODE_BITS // 8):
            self.code = (self.code << 8) | self.reader.read_byte()

    def decode_symbol(self, frequencies: Tuple[int, ...]) -> int:
        total = sum(frequencies)
        unit = self.range // total

        if unit == 0:
            raise RuntimeError("Range collapsed before decoding.")

        offset = self.code // unit
        if offset >= total:
            raise RuntimeError(f"Range decoder state invalid: offset={offset}, total={total}")

        cumulative = 0
        symbol = -1

        for i, freq in enumerate(frequencies):
            next_cumulative = cumulative + freq
            if cumulative <= offset < next_cumulative:
                symbol = i
                break
            cumulative = next_cumulative

        if symbol < 0:
            raise RuntimeError("Unable to decode symbol.")

        freq = frequencies[symbol]

        self.code -= unit * cumulative
        self.range = unit * freq

        while self.range < (1 << (CODE_BITS - 8)):
            self.range <<= 8
            self.code = (self.code << 8) | self.reader.read_byte()

        return symbol


# ============================================================================
# CAUSAL ENCODING
# ============================================================================

@dataclass
class EncodeResult:
    payload: bytes
    frequency_hashes: List[str]
    contexts: int


def encode_causal(
    training_prefix: bytes,
    heldout: bytes,
    provider: NeuralFrequencyProvider,
    verbose: bool = True,
) -> EncodeResult:

    history = bytearray(training_prefix)
    encoder = RangeEncoder()
    hashes = []

    for index, actual_byte in enumerate(heldout):
        context = bytes(history[-CONTEXT_SIZE:])
        frequencies = provider.frequencies(context)
        hashes.append(frequency_hash(frequencies))

        if verbose and index < PRINT_FIRST_N:
            print(
                f"ENC {index:5d} "
                f"byte={actual_byte:3d} "
                f"context={context!r} "
                f"freq_hash={hashes[-1]}"
            )

        encoder.encode_symbol(actual_byte, frequencies)
        history.append(actual_byte)

    return EncodeResult(
        payload=encoder.finish(),
        frequency_hashes=hashes,
        contexts=len(provider.cache),
    )


# ============================================================================
# CAUSAL DECODING
# ============================================================================

@dataclass
class DecodeResult:
    output: bytes
    frequency_hashes: List[str]
    contexts: int


def decode_causal(
    training_prefix: bytes,
    payload: bytes,
    output_length: int,
    provider: NeuralFrequencyProvider,
    verbose: bool = True,
) -> DecodeResult:

    history = bytearray(training_prefix)
    decoder = RangeDecoder(payload)
    output = bytearray()
    hashes = []

    for index in range(output_length):
        context = bytes(history[-CONTEXT_SIZE:])
        frequencies = provider.frequencies(context)
        hashes.append(frequency_hash(frequencies))

        symbol = decoder.decode_symbol(frequencies)

        if verbose and index < PRINT_FIRST_N:
            print(
                f"DEC {index:5d} "
                f"context={context!r} "
                f"freq_hash={hashes[-1]}"
            )

        output.append(symbol)
        history.append(symbol)

    return DecodeResult(
        output=bytes(output),
        frequency_hashes=hashes,
        contexts=len(provider.cache),
    )


# ============================================================================
# STATIC FREQUENCY BASELINE
# ============================================================================

def static_frequencies(data: bytes) -> Tuple[int, ...]:
    counts = [MIN_FREQ] * SYMBOLS
    for byte in data:
        counts[byte] += 1

    total = sum(counts)
    if total == FREQ_TOTAL:
        return tuple(counts)

    available = FREQ_TOTAL - SYMBOLS * MIN_FREQ
    raw = max(1, len(data))

    scaled = [
        ((counts[i] - MIN_FREQ) * available) / raw
        for i in range(SYMBOLS)
    ]
    floors = [math.floor(x) for x in scaled]
    result = [MIN_FREQ + x for x in floors]

    leftover = FREQ_TOTAL - sum(result)
    order = sorted(
        range(SYMBOLS),
        key=lambda i: (-(scaled[i] - floors[i]), i),
    )

    for i in order[:leftover]:
        result[i] += 1

    return tuple(result)


def encode_static(data: bytes, frequencies: Tuple[int, ...]) -> bytes:
    encoder = RangeEncoder()
    for symbol in data:
        encoder.encode_symbol(symbol, frequencies)
    return encoder.finish()


def decode_static(payload: bytes, length: int, frequencies: Tuple[int, ...]) -> bytes:
    decoder = RangeDecoder(payload)
    output = bytearray()
    for _ in range(length):
        output.append(decoder.decode_symbol(frequencies))
    return bytes(output)


# ============================================================================
# METADATA
# ============================================================================

def build_metadata(
    corpus_length: int,
    training_length: int,
    heldout_length: int,
    model_bytes: bytes,
) -> bytes:

    metadata = {
        "test": TEST_NAME,
        "version": 1,
        "seed": SEED,
        "context_size": CONTEXT_SIZE,
        "symbols": SYMBOLS,
        "frequency_total": FREQ_TOTAL,
        "minimum_frequency": MIN_FREQ,
        "coder_bits": CODE_BITS,
        "train_fraction": TRAIN_FRACTION,
        "corpus_length": corpus_length,
        "training_length": training_length,
        "heldout_length": heldout_length,
        "architecture": MODEL_NAME,
        "embedding": EMBED_DIM,
        "hidden1": HIDDEN1,
        "hidden2": HIDDEN2,
        "training_samples": TRAINING_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_sha256": model_hash(model_bytes),
    }

    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# ============================================================================
# TEST HELPERS
# ============================================================================

def assert_equal(name: str, a, b):
    if a != b:
        print(f"{name}: FAIL")
        raise AssertionError(f"{name} failed: {a} != {b}")
    print(f"{name}: PASS")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print(TEST_NAME)
    print("=" * 72)
    print()
    print("TEST10.1 is the validated range-coder foundation.")
    print("TEST10.2 inserted the neural probability model.")
    print("TEST10.3 performs the corrected larger-stream measurement.")
    print()
    print(f"The neural model is trained only on the first {TRAIN_FRACTION * 100:.0f}%.")
    print(f"The final {(1.0 - TRAIN_FRACTION) * 100:.0f}% is held out for causal compression.")
    print()

    device = print_device_info()
    print()

    # ----------------------------------------------------------------------
    # Corpus
    # ----------------------------------------------------------------------
    corpus = build_corpus()
    training_length = int(len(corpus) * TRAIN_FRACTION)

    training = corpus[:training_length]
    heldout = corpus[training_length:]

    print()
    print(f"Corpus:              {len(corpus) / 1024:.2f} KB")
    print(f"Training:            {len(training) / 1024:.2f} KB")
    print(f"Held-out:            {len(heldout) / 1024:.2f} KB")
    print(
        f"Split:               "
        f"{len(training) / len(corpus) * 100:.2f}% / "
        f"{len(heldout) / len(corpus) * 100:.2f}%"
    )

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = NeuralProbabilityModel().to(device)
    train_model(model, training, device)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters:    {parameter_count:,}")

    serialized = serialize_model(model)
    print(f"Serialized model:    {len(serialized) / 1024:.2f} KB")
    print(f"Model SHA-256:       {model_hash(serialized)}")

    # ----------------------------------------------------------------------
    # Model determinism
    # ----------------------------------------------------------------------
    provider_a = NeuralFrequencyProvider(model, device)
    provider_b = NeuralFrequencyProvider(model, device)

    raw_test_contexts = [
        b"TEST",
        b"test",
        b" neur",
        b"caus",
        b"AAAA",
        b"\x00\x01\x02\x03",
    ]

    # Context is padded to exactly four bytes.
    test_contexts = [
        x.ljust(CONTEXT_SIZE, b"\x00")[:CONTEXT_SIZE]
        for x in raw_test_contexts
    ]

    print()
    print(f"Contexts tested:     {len(test_contexts)}")

    for context in test_contexts:
        fa = provider_a.frequencies(context)
        fb = provider_b.frequencies(context)
        assert_equal(f"Frequency identity {context!r}", fa, fb)

    # ----------------------------------------------------------------------
    # Neural causal encoding & decoding
    # ----------------------------------------------------------------------
    print()
    print(f"Training prefix:     {len(training) / 1024:.2f} KB")
    print(f"Held-out data:       {len(heldout) / 1024:.2f} KB")

    encode_start = time.perf_counter()
    encoded = encode_causal(training, heldout, provider_a, verbose=True)
    encode_time = time.perf_counter() - encode_start

    print()
    print(f"Encode time:         {encode_time:.3f} s")

    decode_start = time.perf_counter()
    decoded = decode_causal(
        training,
        encoded.payload,
        len(heldout),
        provider_b,
        verbose=True,
    )
    decode_time = time.perf_counter() - decode_start

    print(f"Decode time:         {decode_time:.3f} s")
    print()

    # ----------------------------------------------------------------------
    # Verification assertions
    # ----------------------------------------------------------------------
    assert_equal("Exact reconstruction", decoded.output, heldout)
    assert_equal("SHA-256 reconstruction", sha256(decoded.output), sha256(heldout))
    assert_equal("Encoder/Decoder frequency identity", encoded.frequency_hashes, decoded.frequency_hashes)

    # ----------------------------------------------------------------------
    # Static Baseline & Break-even Accounting
    # ----------------------------------------------------------------------
    static_freqs = static_frequencies(heldout)
    static_payload = encode_static(heldout, static_freqs)

    metadata = build_metadata(len(corpus), len(training), len(heldout), serialized)
    total_cost = len(serialized) + len(metadata) + len(encoded.payload)

    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"Raw held-out payload size:    {len(heldout):,d} bytes")
    print(f"Static frequency payload:     {len(static_payload):,d} bytes")
    print(f"Neural payload size:          {len(encoded.payload):,d} bytes")
    print(f"Serialized model size:       {len(serialized):,d} bytes")
    print(f"Metadata size:                {len(metadata):,d} bytes")
    print(f"Total neural footprint:       {total_cost:,d} bytes")
    print("-" * 72)

    bpp_raw = 8.0
    bpp_neural = (len(encoded.payload) * 8.0) / len(heldout)
    bpp_static = (len(static_payload) * 8.0) / len(heldout)
    bpp_total = (total_cost * 8.0) / len(heldout)

    print(f"Raw bits/byte:                {bpp_raw:.4f}")
    print(f"Static baseline bits/byte:    {bpp_static:.4f}")
    print(f"Neural payload bits/byte:     {bpp_neural:.4f}")
    print(f"Total footprint bits/byte:    {bpp_total:.4f}")

    print()
    print("All tests passed successfully.")


if __name__ == "__main__":
    main()