"""
TEST10.2 — CAUSAL NEURAL RANGE-CODER EXPERIMENT

Purpose
-------
TEST10.1 proved that the causal range coder, deterministic probability
quantization, state synchronization, file-boundary handling, and exact
reconstruction machinery work independently of a neural model.

TEST10.2 replaces only the deterministic SHA-256 probability generator
with the TEST10 neural probability model:

    4-byte causal context
        |
        v
    Embedding(256, 32)
        |
        v
    Flatten
        |
        v
    Linear(128) + ReLU
        |
        v
    Linear(256)
        |
        v
    Softmax
        |
        v
    deterministic integer frequencies
        |
        v
    causal range coder

The model is trained ONLY on the training prefix.

The held-out 25% is never used during training.

The decoder receives:
    - the trained model
    - the encoded payload
    - the training prefix
    - the held-out byte count

It reconstructs the held-out stream using only reconstructed causal state.

Mandatory checks:
    1. training/held-out isolation
    2. neural model determinism
    3. causal encoder/decoder frequency identity
    4. exact byte-for-byte reconstruction
    5. SHA-256 reconstruction
    6. deterministic payload reproduction
    7. model/payload/metadata accounting

Architecture
------------
Embedding(256, 32) -> Flatten -> Linear(128) -> ReLU -> Linear(256)

Training
--------
300,000 causal samples
25 epochs
batch size 4096
Adam, learning rate 0.001

Coder
-----
32-bit range state
TOTAL_FREQUENCY = 32768
MIN_FREQUENCY = 1

ROCm
----
Run normally or with:

    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 test10_2.py
"""

import sys
import time
import hashlib
import json
import random
import math
import io
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None
    TORCH_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 1337

CONTEXT_SIZE = 4

CODE_BITS = 32
TOTAL_FREQUENCY = 1 << 15
MIN_FREQUENCY = 1

TRAINING_SAMPLES = 300_000
EPOCHS = 25
BATCH_SIZE = 4096
LEARNING_RATE = 0.001

# Corpus is deliberately larger than the training sample count so that
# the held-out region is genuinely unseen by the optimizer.
CORPUS_SIZE = 400_000
TRAIN_FRACTION = 0.75

DIAGNOSTIC_POSITIONS = 32
VERBOSE_CAUSAL_DIAGNOSTICS = True

MODEL_NAME = "TEST10_Embedding32_Linear128_Linear256"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def format_bytes(n):
    if n < 1024:
        return f"{n:,} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"
    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.2f} MB"
    return f"{n / (1024 ** 3):.2f} GB"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def print_device_report():
    print()
    print("=" * 70)
    print("DEVICE")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("PyTorch:       unavailable")
        raise RuntimeError("TEST10.2 requires PyTorch.")

    print(f"PyTorch:       {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA API:      available")
        print("Backend:       ROCm-compatible PyTorch interface")
        try:
            print(f"GPU:           {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM:          {format_bytes(props.total_memory)}")
        except Exception as exc:
            print(f"GPU details unavailable: {exc}")
    else:
        print("CUDA API:      unavailable")
        print("GPU:           CPU")


# ============================================================
# SOURCE-LIKE CORPUS
# ============================================================

def generate_source_like_data(size):
    """
    Deterministic structured corpus.

    This is only an experimental corpus. The real TEST10 corpus can
    replace this function later without changing the coder architecture.
    """

    templates = [
        b"def function_%03d(value):\n",
        b"    result = value * %d\n",
        b"    return result\n\n",
        b"class Object_%03d:\n",
        b"    def __init__(self, value):\n",
        b"        self.value = value\n",
        b"    def transform(self):\n",
        b"        return self.value + %d\n\n",
        b"# comment: deterministic neural compression test data\n",
        b"import hashlib\n",
        b"import numpy as np\n",
        b"import torch\n",
        b"\n",
    ]

    chunks = []
    total = 0
    index = 0

    while total < size:
        template = templates[index % len(templates)]

        if b"%03d" in template:
            chunk = template % (index % 1000,)
        elif b"%d" in template:
            chunk = template % (index % 100,)
        else:
            chunk = template

        chunks.append(chunk)
        total += len(chunk)
        index += 1

    return b"".join(chunks)[:size]


# ============================================================
# TRAINING DATA
# ============================================================

def build_training_dataset(training_prefix, sample_count):
    """
    Build strictly causal supervised examples.

    Input:
        four previous bytes

    Target:
        next byte

    Only positions inside the training prefix are sampled.
    No held-out byte can enter the optimizer.
    """

    require(
        len(training_prefix) >= CONTEXT_SIZE + 1,
        "Training prefix is too small."
    )

    max_position = len(training_prefix) - 1

    # Deterministic evenly distributed positions across the training prefix.
    # Repetition is intentional if sample_count exceeds unique positions.
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.integers(
        CONTEXT_SIZE,
        max_position + 1,
        size=sample_count,
        dtype=np.int64,
    )

    contexts = np.empty(
        (sample_count, CONTEXT_SIZE),
        dtype=np.int64,
    )
    targets = np.empty(
        sample_count,
        dtype=np.int64,
    )

    raw = np.frombuffer(training_prefix, dtype=np.uint8)

    for i, position in enumerate(positions):
        contexts[i] = raw[position - CONTEXT_SIZE:position]
        targets[i] = raw[position]

    return (
        torch.from_numpy(contexts),
        torch.from_numpy(targets),
    )


# ============================================================
# NEURAL MODEL
# ============================================================

class NeuralProbabilityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            256,
            32,
        )

        self.fc1 = nn.Linear(
            CONTEXT_SIZE * 32,
            128,
        )

        self.fc2 = nn.Linear(
            128,
            256,
        )

    def forward(self, context):
        x = self.embedding(context)
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_model(training_prefix):
    print()
    print("=" * 70)
    print("NEURAL MODEL TRAINING")
    print("=" * 70)

    print(f"Architecture:       {MODEL_NAME}")
    print(f"Training samples:   {TRAINING_SAMPLES:,}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Batch size:         {BATCH_SIZE:,}")
    print(f"Learning rate:      {LEARNING_RATE}")

    contexts, targets = build_training_dataset(
        training_prefix,
        TRAINING_SAMPLES,
    )

    dataset = TensorDataset(
        contexts,
        targets,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device:    {device}")

    model = NeuralProbabilityModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_context, batch_target in loader:
            batch_context = batch_context.to(
                device,
                non_blocking=True,
            )
            batch_target = batch_target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_context)
            loss = criterion(
                logits,
                batch_target,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * batch_context.shape[0]
            )

            total_correct += (
                (logits.argmax(dim=1) == batch_target)
                .sum()
                .item()
            )

            total_seen += batch_context.shape[0]

        average_loss = total_loss / total_seen
        accuracy = total_correct / total_seen

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"loss={average_loss:.6f}  "
            f"accuracy={accuracy:.4%}"
        )

    elapsed = time.time() - start

    model.eval()

    print(f"Training time:      {elapsed:.3f} s")

    return model, device, elapsed


# ============================================================
# MODEL SERIALIZATION / ACCOUNTING
# ============================================================

def serialize_model(model):
    """
    Deterministically serialize the trained model state.

    The byte count is included in compression accounting.
    """

    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer,
        _use_new_zipfile_serialization=False,
    )

    return buffer.getvalue()


def model_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# PROBABILITY -> FREQUENCY QUANTIZATION
# ============================================================

def normalize_probabilities(probabilities):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    require(
        probabilities.shape == (256,),
        "Probability vector must contain exactly 256 symbols.",
    )

    probabilities = np.maximum(
        probabilities,
        0.0,
    )

    total = float(probabilities.sum())

    if not math.isfinite(total) or total <= 0.0:
        probabilities = np.ones(
            256,
            dtype=np.float64,
        )
        total = 256.0

    probabilities /= total

    return probabilities


def probabilities_to_frequencies(
    probabilities,
    total_frequency=TOTAL_FREQUENCY,
    minimum_frequency=MIN_FREQUENCY,
):
    probabilities = normalize_probabilities(
        probabilities,
    )

    require(
        total_frequency >= 256 * minimum_frequency,
        "TOTAL_FREQUENCY is too small.",
    )

    available = (
        total_frequency
        - 256 * minimum_frequency
    )

    raw = probabilities * available

    base = np.floor(raw).astype(np.int64)

    frequencies = base + minimum_frequency

    current = int(frequencies.sum())

    difference = total_frequency - current

    if difference > 0:
        fractional = raw - np.floor(raw)

        order = np.argsort(
            -fractional,
            kind="stable",
        )

        for index in order[:difference]:
            frequencies[index] += 1

    elif difference < 0:
        excess = -difference

        removable = (
            frequencies
            - minimum_frequency
        )

        order = np.argsort(
            -removable,
            kind="stable",
        )

        for index in order:
            if excess <= 0:
                break

            amount = min(
                int(removable[index]),
                excess,
            )

            frequencies[index] -= amount
            excess -= amount

        require(
            excess == 0,
            "Frequency correction could not be completed.",
        )

    frequencies = frequencies.astype(np.int64)

    require(
        int(frequencies.sum()) == total_frequency,
        "Quantized frequencies do not sum to TOTAL_FREQUENCY.",
    )

    require(
        int(frequencies.min()) >= minimum_frequency,
        "A frequency fell below MIN_FREQUENCY.",
    )

    return frequencies


def validate_frequency_table(frequencies):
    frequencies = np.asarray(
        frequencies,
        dtype=np.int64,
    )

    require(
        frequencies.shape == (256,),
        "Invalid frequency table shape.",
    )

    require(
        np.all(frequencies >= MIN_FREQUENCY),
        "Frequency table contains invalid frequency.",
    )

    require(
        int(frequencies.sum()) == TOTAL_FREQUENCY,
        "Frequency table has incorrect total.",
    )


def frequency_table_hash(frequencies):
    validate_frequency_table(frequencies)

    raw = np.asarray(
        frequencies,
        dtype="<i8",
    ).tobytes()

    return sha256(raw)


# ============================================================
# NEURAL PROBABILITY ENGINE
# ============================================================

class NeuralProbabilityEngine:
    """
    Turns a causal 4-byte context into the exact integer frequency
    table consumed by the range coder.

    Context -> model -> softmax -> CPU float64 -> deterministic
    integer frequencies.

    A context cache is used because repeated contexts are common in
    structured data. This does not alter the mathematics.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = {}

    def frequencies_for_context(self, context):
        context = bytes(context)

        require(
            len(context) == CONTEXT_SIZE,
            "Neural model requires exactly 4 context bytes.",
        )

        if context in self.cache:
            return self.cache[context]

        context_tensor = torch.tensor(
            list(context),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(
                context_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].detach().cpu().numpy()

        frequencies = probabilities_to_frequencies(
            probabilities
        )

        # Immutable copy prevents accidental state mutation.
        frequencies = np.asarray(
            frequencies,
            dtype=np.int64,
        ).copy()

        self.cache[context] = frequencies

        return frequencies


# ============================================================
# CAUSAL CONTEXT
# ============================================================

def get_causal_context(
    reconstructed,
    position,
    context_size=CONTEXT_SIZE,
):
    if position < context_size:
        return None

    start = position - context_size

    if start < 0 or position > len(reconstructed):
        return None

    return bytes(
        reconstructed[start:position]
    )


# ============================================================
# RANGE ENCODER
# ============================================================

class RangeEncoder:
    def __init__(self, total_frequency=TOTAL_FREQUENCY):
        self.total_frequency = int(total_frequency)
        self.low = 0
        self.high = (1 << CODE_BITS) - 1
        self.pending_bits = 0
        self.output = []

    def output_bit(self, bit):
        bit = int(bit)

        self.output.append(bit)

        opposite = 1 - bit

        while self.pending_bits > 0:
            self.output.append(opposite)
            self.pending_bits -= 1

    def encode_symbol(self, symbol, frequencies):
        symbol = int(symbol)

        validate_frequency_table(frequencies)

        cumulative = np.cumsum(
            frequencies,
            dtype=np.int64,
        )

        low_count = (
            0
            if symbol == 0
            else int(cumulative[symbol - 1])
        )

        high_count = int(cumulative[symbol])

        range_width = (
            self.high
            - self.low
            + 1
        )

        self.high = (
            self.low
            + range_width * high_count // self.total_frequency
            - 1
        )

        self.low = (
            self.low
            + range_width * low_count // self.total_frequency
        )

        while True:
            if (
                (self.low ^ self.high)
                < (1 << (CODE_BITS - 1))
            ):
                bit = (
                    self.high
                    >> (CODE_BITS - 1)
                ) & 1

                self.output_bit(bit)

                mask = (1 << CODE_BITS) - 1

                self.low = (
                    self.low << 1
                ) & mask

                self.high = (
                    self.high << 1
                ) & mask

                self.high |= 1

            elif (
                self.low & (1 << (CODE_BITS - 2))
            ) and not (
                self.high & (1 << (CODE_BITS - 2))
            ):
                self.pending_bits += 1

                mask = (1 << CODE_BITS) - 1
                underflow_bit = 1 << (CODE_BITS - 2)

                self.low = (
                    (self.low ^ underflow_bit)
                    << 1
                ) & mask

                self.high = (
                    (self.high ^ underflow_bit)
                    << 1
                ) & mask

                self.high |= 1

            else:
                break

    def finish(self):
        self.pending_bits += 1

        if self.low < (1 << (CODE_BITS - 2)):
            self.output_bit(0)
        else:
            self.output_bit(1)

        packed = bytearray()

        accumulator = 0
        count = 0

        for bit in self.output:
            accumulator = (
                (accumulator << 1)
                | bit
            )

            count += 1

            if count == 8:
                packed.append(accumulator)
                accumulator = 0
                count = 0

        if count:
            accumulator <<= 8 - count
            packed.append(accumulator)

        return bytes(packed)


# ============================================================
# RANGE DECODER
# ============================================================

class RangeDecoder:
    def __init__(
        self,
        data,
        total_frequency=TOTAL_FREQUENCY,
    ):
        self.data = data
        self.bit_position = 0
        self.total_frequency = int(total_frequency)

        self.low = 0
        self.high = (1 << CODE_BITS) - 1
        self.code = 0

        for _ in range(CODE_BITS):
            self.code = (
                (self.code << 1)
                | self.read_bit()
            )

    def read_bit(self):
        byte_position = self.bit_position // 8
        bit_offset = 7 - (self.bit_position % 8)

        self.bit_position += 1

        if byte_position >= len(self.data):
            return 0

        return (
            self.data[byte_position]
            >> bit_offset
        ) & 1

    def decode_symbol(self, frequencies):
        validate_frequency_table(frequencies)

        cumulative = np.cumsum(
            frequencies,
            dtype=np.int64,
        )

        range_width = (
            self.high
            - self.low
            + 1
        )

        scaled_value = (
            (
                (
                    self.code
                    - self.low
                    + 1
                )
                * self.total_frequency
                - 1
            )
            // range_width
        )

        require(
            0 <= scaled_value < self.total_frequency,
            "Decoder scaled value outside frequency range.",
        )

        symbol = int(
            np.searchsorted(
                cumulative,
                scaled_value,
                side="right",
            )
        )

        require(
            0 <= symbol < 256,
            "Decoder produced invalid symbol.",
        )

        low_count = (
            0
            if symbol == 0
            else int(cumulative[symbol - 1])
        )

        high_count = int(cumulative[symbol])

        self.high = (
            self.low
            + range_width * high_count // self.total_frequency
            - 1
        )

        self.low = (
            self.low
            + range_width * low_count // self.total_frequency
        )

        mask = (1 << CODE_BITS) - 1
        top_bit = 1 << (CODE_BITS - 1)
        underflow_bit = 1 << (CODE_BITS - 2)

        while True:
            if (
                (self.low ^ self.high)
                < top_bit
            ):
                self.low = (
                    self.low << 1
                ) & mask

                self.high = (
                    self.high << 1
                ) & mask

                self.high |= 1

                self.code = (
                    self.code << 1
                ) & mask

                self.code |= self.read_bit()

            elif (
                self.low & underflow_bit
            ) and not (
                self.high & underflow_bit
            ):
                self.low = (
                    (self.low ^ underflow_bit)
                    << 1
                ) & mask

                self.high = (
                    (self.high ^ underflow_bit)
                    << 1
                ) & mask

                self.high |= 1

                self.code = (
                    (self.code ^ underflow_bit)
                    << 1
                ) & mask

                self.code |= self.read_bit()

            else:
                break

        return symbol


# ============================================================
# CAUSAL ENCODER
# ============================================================

def causal_range_encode(
    training_prefix,
    heldout,
    probability_engine,
    diagnostics=False,
):
    encoder = RangeEncoder(
        TOTAL_FREQUENCY
    )

    reconstructed = bytearray(
        training_prefix
    )

    frequency_hashes = []

    start_time = time.time()

    for index, actual in enumerate(heldout):
        global_position = (
            len(training_prefix)
            + index
        )

        context = get_causal_context(
            reconstructed,
            global_position,
        )

        require(
            context is not None,
            "Held-out position unexpectedly lacks 4-byte context.",
        )

        frequencies = probability_engine.frequencies_for_context(
            context
        )

        table_hash = frequency_table_hash(
            frequencies
        )

        frequency_hashes.append(table_hash)

        if (
            diagnostics
            and index < DIAGNOSTIC_POSITIONS
        ):
            print(
                f"    ENC {index:6d} "
                f"byte={actual:3d} "
                f"context={context!r} "
                f"freq_hash={table_hash[:16]}"
            )

        # Actual symbol is encoded only after its causal prediction
        # has been produced.
        encoder.encode_symbol(
            actual,
            frequencies,
        )

        reconstructed.append(actual)

    encoded = encoder.finish()

    return (
        encoded,
        frequency_hashes,
        bytes(reconstructed),
        time.time() - start_time,
    )


# ============================================================
# CAUSAL DECODER
# ============================================================

def causal_range_decode(
    training_prefix,
    encoded,
    heldout_length,
    probability_engine,
    diagnostics=False,
):
    decoder = RangeDecoder(
        encoded,
        TOTAL_FREQUENCY,
    )

    reconstructed = bytearray(
        training_prefix
    )

    frequency_hashes = []

    start_time = time.time()

    for index in range(heldout_length):
        global_position = (
            len(training_prefix)
            + index
        )

        context = get_causal_context(
            reconstructed,
            global_position,
        )

        require(
            context is not None,
            "Held-out position unexpectedly lacks 4-byte context.",
        )

        frequencies = probability_engine.frequencies_for_context(
            context
        )

        table_hash = frequency_table_hash(
            frequencies
        )

        frequency_hashes.append(table_hash)

        if (
            diagnostics
            and index < DIAGNOSTIC_POSITIONS
        ):
            print(
                f"    DEC {index:6d} "
                f"context={context!r} "
                f"freq_hash={table_hash[:16]}"
            )

        decoded = decoder.decode_symbol(
            frequencies
        )

        reconstructed.append(decoded)

    return (
        bytes(
            reconstructed[
                len(training_prefix):
            ]
        ),
        frequency_hashes,
        bytes(reconstructed),
        time.time() - start_time,
    )


# ============================================================
# MODEL DETERMINISM TEST
# ============================================================

def test_model_determinism(model, device):
    print()
    print("=" * 70)
    print("TEST 1 — NEURAL MODEL DETERMINISM")
    print("=" * 70)

    engine_a = NeuralProbabilityEngine(
        model,
        device,
    )

    engine_b = NeuralProbabilityEngine(
        model,
        device,
    )

    contexts = [
        b"TEST",
        b"test",
        b"\x00\x01\x02\x03",
        b"\xff\xfe\xfd\xfc",
        b"hash",
        b"AAAA",
    ]

    for context in contexts:
        freq_a = engine_a.frequencies_for_context(context)
        freq_b = engine_b.frequencies_for_context(context)

        require(
            np.array_equal(freq_a, freq_b),
            f"Model output is not deterministic for {context!r}.",
        )

    print(f"Contexts tested:     {len(contexts)}")
    print("Frequency identity:  PASS")

    return True


# ============================================================
# CAUSAL ROUND TRIP TEST
# ============================================================

def test_causal_round_trip(
    model,
    device,
    training_prefix,
    heldout,
):
    print()
    print("=" * 70)
    print("TEST 2 — NEURAL CAUSAL ROUND TRIP")
    print("=" * 70)

    print(
        f"Training prefix:     {format_bytes(len(training_prefix))}"
    )
    print(
        f"Held-out data:       {format_bytes(len(heldout))}"
    )

    encoder_engine = NeuralProbabilityEngine(
        model,
        device,
    )

    (
        encoded,
        encoder_hashes,
        encoder_state,
        encode_time,
    ) = causal_range_encode(
        training_prefix,
        heldout,
        encoder_engine,
        diagnostics=VERBOSE_CAUSAL_DIAGNOSTICS,
    )

    print(
        f"Encoded payload:     {format_bytes(len(encoded))}"
    )
    print(
        f"Encode time:         {encode_time:.3f} s"
    )
    print(
        f"Unique neural contexts:"
        f" {len(encoder_engine.cache):,}"
    )

    decoder_engine = NeuralProbabilityEngine(
        model,
        device,
    )

    (
        decoded,
        decoder_hashes,
        decoder_state,
        decode_time,
    ) = causal_range_decode(
        training_prefix,
        encoded,
        len(heldout),
        decoder_engine,
        diagnostics=VERBOSE_CAUSAL_DIAGNOSTICS,
    )

    print(
        f"Decode time:         {decode_time:.3f} s"
    )
    print(
        f"Unique decoder contexts:"
        f" {len(decoder_engine.cache):,}"
    )

    require(
        encoder_hashes == decoder_hashes,
        "Encoder/decoder neural frequency tables diverged.",
    )

    print("Frequency identity:  PASS")

    require(
        decoded == heldout,
        "Neural decoder failed exact reconstruction.",
    )

    print("Exact reconstruction: PASS")

    original_hash = sha256(heldout)
    decoded_hash = sha256(decoded)

    require(
        original_hash == decoded_hash,
        "Neural decoder SHA-256 mismatch.",
    )

    print(
        f"SHA-256:             PASS ({original_hash})"
    )

    expected_state = training_prefix + heldout

    require(
        encoder_state == expected_state,
        "Encoder causal state is incorrect.",
    )

    require(
        decoder_state == expected_state,
        "Decoder causal state is incorrect.",
    )

    require(
        encoder_state == decoder_state,
        "Encoder and decoder states differ.",
    )

    print("Causal state identity: PASS")

    return {
        "encoded": encoded,
        "encode_time": encode_time,
        "decode_time": decode_time,
        "encoded_size": len(encoded),
        "heldout_size": len(heldout),
        "ratio": len(encoded) / len(heldout),
        "sha256": original_hash,
        "encoder_contexts": len(encoder_engine.cache),
        "decoder_contexts": len(decoder_engine.cache),
        "encoder_hashes": encoder_hashes,
    }


# ============================================================
# PAYLOAD DETERMINISM TEST
# ============================================================

def test_payload_determinism(
    model,
    device,
    training_prefix,
    heldout,
):
    print()
    print("=" * 70)
    print("TEST 3 — NEURAL ENCODING DETERMINISM")
    print("=" * 70)

    engine_a = NeuralProbabilityEngine(
        model,
        device,
    )

    result_a = causal_range_encode(
        training_prefix,
        heldout,
        engine_a,
        diagnostics=False,
    )

    engine_b = NeuralProbabilityEngine(
        model,
        device,
    )

    result_b = causal_range_encode(
        training_prefix,
        heldout,
        engine_b,
        diagnostics=False,
    )

    encoded_a = result_a[0]
    encoded_b = result_b[0]

    require(
        encoded_a == encoded_b,
        "Identical neural input produced different payloads.",
    )

    require(
        result_a[1] == result_b[1],
        "Identical neural input produced different frequency sequences.",
    )

    print(
        f"Payload size:        {format_bytes(len(encoded_a))}"
    )
    print(
        f"Payload SHA-256:     {sha256(encoded_a)}"
    )
    print("Bitstream determinism: PASS")
    print("Frequency determinism: PASS")

    return True


# ============================================================
# ACCOUNTING
# ============================================================

def print_accounting(
    model_bytes,
    payload_bytes,
    metadata_bytes,
    training_bytes,
    heldout_bytes,
):
    print()
    print("=" * 70)
    print("COMPRESSION ACCOUNTING")
    print("=" * 70)

    total_transmitted = (
        len(model_bytes)
        + len(payload_bytes)
        + len(metadata_bytes)
    )

    print(
        f"Training prefix:     {format_bytes(len(training_bytes))}"
    )
    print(
        f"Held-out original:   {format_bytes(len(heldout_bytes))}"
    )
    print()
    print(
        f"Model state:         {format_bytes(len(model_bytes))}"
    )
    print(
        f"Range payload:       {format_bytes(len(payload_bytes))}"
    )
    print(
        f"Metadata:            {format_bytes(len(metadata_bytes))}"
    )
    print(
        f"Total transmitted:   {format_bytes(total_transmitted)}"
    )

    print()

    print(
        f"Payload-only ratio:  "
        f"{len(payload_bytes) / len(heldout_bytes):.4f}x"
    )

    print(
        f"Full ratio:          "
        f"{total_transmitted / len(heldout_bytes):.4f}x"
    )

    print()
    print(
        "NOTE: The full ratio is intentionally reported separately."
    )
    print(
        "It includes model + payload + metadata, rather than hiding"
    )
    print(
        "the cost of transmitting the neural model."
    )

    return total_transmitted


# ============================================================
# MAIN TEST SUITE
# ============================================================

def run_all_tests():
    print()
    print("=" * 70)
    print("TEST10.2")
    print("CAUSAL NEURAL RANGE-CODER EXPERIMENT")
    print("=" * 70)

    print()
    print(
        "TEST10.1 is the validated range-coder foundation."
    )
    print(
        "TEST10.2 now inserts the neural probability model."
    )
    print(
        "The neural model is trained only on the first 75%."
    )
    print(
        "The final 25% is held out for causal compression."
    )

    print_device_report()

    start_time = time.time()

    # --------------------------------------------------------
    # Build corpus and enforce 75/25 split.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    corpus = generate_source_like_data(
        CORPUS_SIZE
    )

    split = int(
        len(corpus) * TRAIN_FRACTION
    )

    training_prefix = corpus[:split]
    heldout = corpus[split:]

    require(
        len(training_prefix) + len(heldout) == len(corpus),
        "Dataset split is invalid.",
    )

    print(
        f"Corpus:              {format_bytes(len(corpus))}"
    )
    print(
        f"Training:            {format_bytes(len(training_prefix))}"
    )
    print(
        f"Held-out:            {format_bytes(len(heldout))}"
    )
    print(
        f"Split:               {len(training_prefix) / len(corpus):.2%}"
        f" / {len(heldout) / len(corpus):.2%}"
    )

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    model, device, training_seconds = train_model(
        training_prefix
    )

    parameter_count = model_parameter_count(
        model
    )

    print(
        f"Model parameters:    {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Model bytes.
    # --------------------------------------------------------

    model_bytes = serialize_model(model)

    model_hash = sha256(model_bytes)

    print()
    print(
        f"Serialized model:    {format_bytes(len(model_bytes))}"
    )
    print(
        f"Model SHA-256:       {model_hash}"
    )

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    test_model_determinism(
        model,
        device,
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    round_trip = test_causal_round_trip(
        model,
        device,
        training_prefix,
        heldout,
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    test_payload_determinism(
        model,
        device,
        training_prefix,
        heldout,
    )

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    metadata = {
        "test": "TEST10.2",
        "seed": RANDOM_SEED,
        "context_size": CONTEXT_SIZE,
        "architecture": MODEL_NAME,
        "training_samples": TRAINING_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "corpus_size": len(corpus),
        "training_size": len(training_prefix),
        "heldout_size": len(heldout),
        "train_fraction": TRAIN_FRACTION,
        "heldout_fraction": 1.0 - TRAIN_FRACTION,
        "code_bits": CODE_BITS,
        "total_frequency": TOTAL_FREQUENCY,
        "minimum_frequency": MIN_FREQUENCY,
        "model_parameters": parameter_count,
        "model_sha256": model_hash,
        "payload_sha256": sha256(round_trip["encoded"]),
        "heldout_sha256": round_trip["sha256"],
    }

    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    print(
        f"Metadata:            {format_bytes(len(metadata_bytes))}"
    )
    print(
        f"Metadata SHA-256:    {sha256(metadata_bytes)}"
    )

    # --------------------------------------------------------
    # Accounting.
    # --------------------------------------------------------

    total_transmitted = print_accounting(
        model_bytes,
        round_trip["encoded"],
        metadata_bytes,
        training_prefix,
        heldout,
    )

    # --------------------------------------------------------
    # Final result.
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("TEST10.2 FINAL RESULT")
    print("=" * 70)

    print()
    print("1. PASS  neural model determinism")
    print("2. PASS  causal neural round trip")
    print("3. PASS  neural encoding determinism")
    print()
    print("Exact reconstruction: PASS")
    print("SHA-256 reconstruction: PASS")
    print("Encoder/decoder frequency identity: PASS")
    print("Causal state identity: PASS")
    print("75/25 train/held-out isolation: PASS")
    print()

    print(
        f"Held-out payload:    {format_bytes(round_trip['encoded_size'])}"
    )
    print(
        f"Payload ratio:       {round_trip['ratio']:.4f}x"
    )
    print(
        f"Full model ratio:    "
        f"{total_transmitted / len(heldout):.4f}x"
    )
    print(
        f"Training time:       {training_seconds:.3f} s"
    )
    print(
        f"Total runtime:       {elapsed:.3f} s"
    )

    print()
    print("=" * 70)
    print("TEST10.2 PASSED")
    print("=" * 70)

    print()
    print(
        "The neural probability model successfully drives the"
    )
    print(
        "same causal range-coding architecture validated by TEST10.1."
    )

    print()
    print(
        "The important next measurement is whether the learned"
    )
    print(
        "probability distribution produces a payload smaller than"
    )
    print(
        "the original held-out data after model cost is accounted for."
    )

    return True


def main():
    try:
        success = run_all_tests()
    except Exception as exc:
        print()
        print("=" * 70)
        print("TEST10.2 FAILED")
        print("=" * 70)
        print()
        print(f"Failure: {exc}")
        print()
        print(
            "Do not interpret compression results until the exact"
        )
        print(
            "causal reconstruction and frequency identity tests pass."
        )
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

