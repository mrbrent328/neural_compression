#!/usr/bin/env python3

import argparse
import hashlib
import math
import time
import zlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

CONTEXT_SIZE = 8
HIDDEN_SIZE = 64
EMBED_SIZE = 16

DEFAULT_EPOCHS = 400
DEFAULT_TRAIN_SAMPLES = 500_000

BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
PRINT_EVERY = 50

# Arithmetic coder probability resolution.
TOTAL_FREQ = 1 << 14

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def entropy(data):

    if not data:
        return 0.0

    counts = np.bincount(
        np.frombuffer(
            data,
            dtype=np.uint8
        ),
        minlength=256
    )

    probs = (
        counts[counts > 0]
        / len(data)
    )

    return float(
        -np.sum(
            probs * np.log2(probs)
        )
    )


# ============================================================
# BIT WRITER
# ============================================================

class BitWriter:

    def __init__(self):

        self.data = bytearray()
        self.current = 0
        self.bits = 0

    def write_bit(self, bit):

        self.current = (
            self.current << 1
        ) | int(bit)

        self.bits += 1

        if self.bits == 8:

            self.data.append(
                self.current
            )

            self.current = 0
            self.bits = 0

    def finish(self):

        if self.bits:

            self.current <<= (
                8 - self.bits
            )

            self.data.append(
                self.current
            )

            self.current = 0
            self.bits = 0

        return bytes(self.data)


# ============================================================
# BIT READER
# ============================================================

class BitReader:

    def __init__(self, data):

        self.data = data
        self.position = 0

    def read_bit(self):

        byte_index = (
            self.position // 8
        )

        bit_index = (
            7
            - (
                self.position % 8
            )
        )

        self.position += 1

        if byte_index >= len(
            self.data
        ):

            return 0

        return (
            self.data[byte_index]
            >> bit_index
        ) & 1


# ============================================================
# ARITHMETIC ENCODER
# ============================================================

class ArithmeticEncoder:

    def __init__(self):

        self.low = 0
        self.high = 0xFFFFFFFF

        self.pending = 0

        self.writer = BitWriter()

    def output_bit(self, bit):

        self.writer.write_bit(bit)

        while self.pending > 0:

            self.writer.write_bit(
                1 - bit
            )

            self.pending -= 1

    def encode(
        self,
        cumulative_low,
        cumulative_high,
        total
    ):

        rng = (
            self.high
            - self.low
            + 1
        )

        self.high = (
            self.low
            + (
                rng
                * cumulative_high
                // total
            )
            - 1
        )

        self.low = (
            self.low
            + (
                rng
                * cumulative_low
                // total
            )
        )

        while True:

            # E1
            if self.high < 0x80000000:

                self.output_bit(0)

            # E2
            elif self.low >= 0x80000000:

                self.output_bit(1)

                self.low -= 0x80000000
                self.high -= 0x80000000

            # E3
            elif (
                self.low >= 0x40000000
                and
                self.high < 0xC0000000
            ):

                self.pending += 1

                self.low -= 0x40000000
                self.high -= 0x40000000

            else:

                break

            self.low <<= 1

            self.high = (
                (self.high << 1)
                | 1
            )

    def finish(self):

        self.pending += 1

        if self.low < 0x40000000:

            self.output_bit(0)

        else:

            self.output_bit(1)

        return self.writer.finish()


# ============================================================
# ARITHMETIC DECODER
# ============================================================

class ArithmeticDecoder:

    def __init__(self, data):

        self.reader = BitReader(data)

        self.low = 0
        self.high = 0xFFFFFFFF

        self.code = 0

        for _ in range(32):

            self.code = (
                (self.code << 1)
                | self.reader.read_bit()
            )

    def get_value(self, total):

        rng = (
            self.high
            - self.low
            + 1
        )

        return (
            (
                (
                    self.code
                    - self.low
                    + 1
                )
                * total
                - 1
            )
            // rng
        )

    def update(
        self,
        cumulative_low,
        cumulative_high,
        total
    ):

        rng = (
            self.high
            - self.low
            + 1
        )

        self.high = (
            self.low
            + (
                rng
                * cumulative_high
                // total
            )
            - 1
        )

        self.low = (
            self.low
            + (
                rng
                * cumulative_low
                // total
            )
        )

        while True:

            if self.high < 0x80000000:

                pass

            elif self.low >= 0x80000000:

                self.low -= 0x80000000
                self.high -= 0x80000000
                self.code -= 0x80000000

            elif (
                self.low >= 0x40000000
                and
                self.high < 0xC0000000
            ):

                self.low -= 0x40000000
                self.high -= 0x40000000
                self.code -= 0x40000000

            else:

                break

            self.low <<= 1

            self.high = (
                (self.high << 1)
                | 1
            )

            self.code = (
                (self.code << 1)
                | self.reader.read_bit()
            )


# ============================================================
# CONVERT PROBABILITIES TO INTEGER FREQUENCIES
# ============================================================

def probabilities_to_cumulative(
    probabilities
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.maximum(
        probabilities,
        1e-12
    )

    probabilities /= (
        probabilities.sum()
    )

    frequencies = np.floor(
        probabilities
        * TOTAL_FREQ
    ).astype(np.int64)

    # Every symbol gets at least one count.
    frequencies = np.maximum(
        frequencies,
        1
    )

    difference = (
        TOTAL_FREQ
        - int(frequencies.sum())
    )

    if difference > 0:

        order = np.argsort(
            -probabilities
        )

        for i in range(
            difference
        ):

            frequencies[
                order[
                    i % 256
                ]
            ] += 1

    elif difference < 0:

        order = np.argsort(
            probabilities
        )

        remaining = -difference

        i = 0

        while remaining > 0:

            idx = order[
                i % 256
            ]

            if frequencies[idx] > 1:

                frequencies[idx] -= 1
                remaining -= 1

            i += 1

    cumulative = np.zeros(
        257,
        dtype=np.int64
    )

    cumulative[1:] = np.cumsum(
        frequencies
    )

    # Safety correction.
    cumulative[-1] = TOTAL_FREQ

    return cumulative


# ============================================================
# ORDER-3 CONTEXT MODEL
# ============================================================

def build_order3_model(data):

    counts = defaultdict(
        lambda: np.ones(
            256,
            dtype=np.int64
        )
    )

    for i in range(
        3,
        len(data)
    ):

        context = data[
            i - 3:i
        ]

        counts[context][
            data[i]
        ] += 1

    return counts


def order3_compress(
    data,
    counts
):

    encoder = ArithmeticEncoder()

    for i in range(
        3,
        len(data)
    ):

        context = data[
            i - 3:i
        ]

        table = counts[
            context
        ]

        cumulative = np.zeros(
            257,
            dtype=np.int64
        )

        cumulative[1:] = np.cumsum(
            table
        )

        total = int(
            cumulative[-1]
        )

        symbol = data[i]

        encoder.encode(
            int(
                cumulative[symbol]
            ),
            int(
                cumulative[
                    symbol + 1
                ]
            ),
            total
        )

    return encoder.finish()


# ============================================================
# NEURAL MODEL
# ============================================================

class NeuralPredictor(nn.Module):

    def __init__(
        self,
        context_size=CONTEXT_SIZE
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            256,
            EMBED_SIZE
        )

        self.net = nn.Sequential(

            nn.Linear(
                context_size
                * EMBED_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                256
            )
        )

    def forward(self, x):

        x = self.embedding(
            x.long()
        )

        x = x.reshape(
            x.shape[0],
            CONTEXT_SIZE
            * EMBED_SIZE
        )

        return self.net(x)


# ============================================================
# MAKE TRAINING SAMPLES
# ============================================================

def make_samples(
    data,
    start,
    end,
    count
):

    possible = np.arange(
        start,
        end,
        dtype=np.int64
    )

    rng = np.random.default_rng(
        SEED
    )

    if count < len(possible):

        indices = rng.choice(
            possible,
            size=count,
            replace=False
        )

    else:

        indices = possible

    x = np.empty(
        (
            len(indices),
            CONTEXT_SIZE
        ),
        dtype=np.uint8
    )

    y = np.empty(
        len(indices),
        dtype=np.int64
    )

    for n, pos in enumerate(
        indices
    ):

        x[n] = np.frombuffer(
            data[
                pos - CONTEXT_SIZE:
                pos
            ],
            dtype=np.uint8
        )

        y[n] = data[pos]

    return x, y


# ============================================================
# TRAIN NEURAL MODEL
# ============================================================

def train_neural(
    data,
    device,
    epochs,
    train_samples
):

    split = int(
        len(data) * 0.75
    )

    start = CONTEXT_SIZE
    end = split

    sample_count = min(
        train_samples,
        end - start
    )

    if sample_count <= 0:

        raise ValueError(
            "File is too small for "
            "the selected context."
        )

    x_np, y_np = make_samples(
        data,
        start,
        end,
        sample_count
    )

    x = torch.from_numpy(
        x_np
    ).to(device)

    y = torch.from_numpy(
        y_np
    ).to(device)

    model = NeuralPredictor().to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Training samples: "
        f"{len(x):,}"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    start_time = time.time()

    for epoch in range(
        epochs
    ):

        permutation = torch.randperm(
            len(x),
            device=device
        )

        total_loss = 0.0

        for batch_start in range(
            0,
            len(x),
            BATCH_SIZE
        ):

            idx = permutation[
                batch_start:
                batch_start
                + BATCH_SIZE
            ]

            xb = x[idx]
            yb = y[idx]

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(idx)
            )

        avg_loss = (
            total_loss
            / len(x)
        )

        if (
            epoch == 0
            or
            (epoch + 1)
            % PRINT_EVERY == 0
            or
            epoch == epochs - 1
        ):

            print(
                f"Epoch "
                f"{epoch:4d} | "
                f"Loss: "
                f"{avg_loss:.8f}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Training time: "
        f"{elapsed:.2f} seconds"
    )

    return model


# ============================================================
# GENERATE NEURAL PROBABILITIES
# ============================================================

def neural_probabilities(
    model,
    data,
    start,
    end,
    device
):

    model.eval()

    all_probs = []

    with torch.no_grad():

        for batch_start in range(
            start,
            end,
            BATCH_SIZE
        ):

            batch_end = min(
                batch_start
                + BATCH_SIZE,
                end
            )

            contexts = []

            for pos in range(
                batch_start,
                batch_end
            ):

                contexts.append(
                    np.frombuffer(
                        data[
                            pos - CONTEXT_SIZE:
                            pos
                        ],
                        dtype=np.uint8
                    )
                )

            x = torch.from_numpy(
                np.stack(contexts)
            ).to(device)

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )

            all_probs.append(
                probs.cpu().numpy()
            )

    return np.concatenate(
        all_probs,
        axis=0
    )


# ============================================================
# NEURAL ARITHMETIC COMPRESSION
# ============================================================

def neural_compress(
    data,
    probabilities,
    start
):

    encoder = ArithmeticEncoder()

    cross_entropy = 0.0

    for i, probs in enumerate(
        probabilities
    ):

        symbol = data[
            start + i
        ]

        probability = max(
            float(
                probs[symbol]
            ),
            1e-12
        )

        cross_entropy += (
            -math.log2(
                probability
            )
        )

        cumulative = (
            probabilities_to_cumulative(
                probs
            )
        )

        encoder.encode(
            int(
                cumulative[symbol]
            ),
            int(
                cumulative[
                    symbol + 1
                ]
            ),
            TOTAL_FREQ
        )

    encoded = encoder.finish()

    average_cross_entropy = (
        cross_entropy
        / len(probabilities)
    )

    return (
        encoded,
        average_cross_entropy
    )


# ============================================================
# EXACT NEURAL DECOMPRESSION
# ============================================================

def neural_decompress(
    compressed,
    model,
    original_data,
    start,
    end,
    device
):

    decoder = ArithmeticDecoder(
        compressed
    )

    model.eval()

    # The first 75% is shared/training data.
    # We copy it into the reconstruction.
    output = bytearray(
        original_data[:start]
    )

    context = bytearray(
        original_data[
            start - CONTEXT_SIZE:
            start
        ]
    )

    with torch.no_grad():

        for pos in range(
            start,
            end
        ):

            x = torch.tensor(
                [
                    list(context)
                ],
                dtype=torch.long,
                device=device
            )

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

            cumulative = (
                probabilities_to_cumulative(
                    probs
                )
            )

            value = decoder.get_value(
                TOTAL_FREQ
            )

            symbol = (
                np.searchsorted(
                    cumulative,
                    value,
                    side="right"
                )
                - 1
            )

            symbol = int(
                max(
                    0,
                    min(
                        255,
                        symbol
                    )
                )
            )

            decoder.update(
                int(
                    cumulative[symbol]
                ),
                int(
                    cumulative[
                        symbol + 1
                    ]
                ),
                TOTAL_FREQ
            )

            output.append(
                symbol
            )

            context = (
                context[1:]
                + bytearray(
                    [symbol]
                )
            )

    return bytes(output)


# ============================================================
# MODEL COMPRESSION
# ============================================================

def compressed_model_size(
    model
):

    raw = bytearray()

    for tensor in (
        model.state_dict().values()
    ):

        raw.extend(
            tensor.detach()
            .cpu()
            .numpy()
            .tobytes()
        )

    raw = bytes(raw)

    compressed = zlib.compress(
        raw,
        level=9
    )

    return (
        len(raw),
        len(compressed)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Neural probability "
            "compression experiment"
        )
    )

    parser.add_argument(
        "file",
        help="File to test"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD FILE
    # --------------------------------------------------------

    with open(
        args.file,
        "rb"
    ) as f:

        data = f.read()

    if len(data) < (
        CONTEXT_SIZE + 10
    ):

        raise ValueError(
            "File is too small."
        )

    original_sha = sha256(
        data
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NEURAL PROBABILITY "
        "COMPRESSION TEST"
    )
    print("=" * 70)

    print()

    print(
        f"Device:       {device}"
    )

    print(
        f"File:         {args.file}"
    )

    print(
        f"Bytes:        {len(data):,}"
    )

    print(
        f"SHA256:       {original_sha}"
    )

    print(
        f"Context:      {CONTEXT_SIZE}"
    )

    print(
        f"Epochs:       {args.epochs}"
    )

    print(
        f"Train samples:{args.samples:>10,}"
    )

    # --------------------------------------------------------
    # RAW ZLIB CONTROL
    # --------------------------------------------------------

    raw_zlib = zlib.compress(
        data,
        level=9
    )

    print()
    print("=" * 70)
    print(
        "CONTROL - RAW ZLIB"
    )
    print("=" * 70)

    print()

    print(
        f"Original: "
        f"{len(data):,} bytes"
    )

    print(
        f"Zlib:     "
        f"{len(raw_zlib):,} bytes"
    )

    print(
        f"Ratio:    "
        f"{len(data) / len(raw_zlib):.4f}x"
    )

    # --------------------------------------------------------
    # ORDER-3 CONTROL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ORDER-3 CONTEXT + "
        "ARITHMETIC CODING"
    )
    print("=" * 70)

    order3_start = time.time()

    order3_counts = (
        build_order3_model(data)
    )

    order3_encoded = (
        order3_compress(
            data,
            order3_counts
        )
    )

    order3_time = (
        time.time()
        - order3_start
    )

    print()

    print(
        f"Encoded payload: "
        f"{len(order3_encoded):,} bytes"
    )

    print(
        f"Time: "
        f"{order3_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # NEURAL MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NEURAL PROBABILITY MODEL"
    )
    print("=" * 70)

    model = train_neural(
        data,
        device,
        epochs=args.epochs,
        train_samples=args.samples
    )

    # --------------------------------------------------------
    # HELD-OUT DATA
    # --------------------------------------------------------

    split = int(
        len(data) * 0.75
    )

    start = split

    test_bytes = (
        len(data)
        - start
    )

    print()
    print("=" * 70)
    print(
        "HELD-OUT TEST"
    )
    print("=" * 70)

    print()

    print(
        f"Training bytes: "
        f"{start:,}"
    )

    print(
        f"Test bytes:     "
        f"{test_bytes:,}"
    )

    print()
    print(
        "Generating neural "
        "probabilities..."
    )

    probabilities = (
        neural_probabilities(
            model,
            data,
            start,
            len(data),
            device
        )
    )

    print(
        f"Prediction count: "
        f"{len(probabilities):,}"
    )

    # --------------------------------------------------------
    # CROSS ENTROPY / ARITHMETIC CODING
    # --------------------------------------------------------

    print()
    print(
        "Arithmetic coding "
        "neural predictions..."
    )

    neural_start = time.time()

    neural_encoded, cross_entropy = (
        neural_compress(
            data,
            probabilities,
            start
        )
    )

    neural_time = (
        time.time()
        - neural_start
    )

    print()

    print(
        f"Cross entropy: "
        f"{cross_entropy:.6f} "
        f"bits/byte"
    )

    theoretical_bytes = (
        cross_entropy
        * test_bytes
        / 8.0
    )

    print(
        f"Theoretical payload "
        f"from cross entropy: "
        f"{theoretical_bytes:,.1f} bytes"
    )

    print(
        f"Arithmetic payload: "
        f"{len(neural_encoded):,} bytes"
    )

    print(
        f"Coding time: "
        f"{neural_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------------

    model_raw, model_compressed = (
        compressed_model_size(
            model
        )
    )

    metadata = 1024

    # Important:
    #
    # The neural experiment currently assumes
    # the first 75% is available to both sides
    # and only measures the coded held-out portion.
    #
    # We therefore show BOTH:
    #
    # 1. payload-only performance
    # 2. complete naive representation
    #
    total_payload_only = (
        len(neural_encoded)
    )

    total_with_model = (
        len(neural_encoded)
        + model_compressed
        + metadata
    )

    print()
    print(
        f"Model raw: "
        f"{model_raw:,} bytes"
    )

    print(
        f"Model compressed: "
        f"{model_compressed:,} bytes"
    )

    print(
        f"Metadata: "
        f"{metadata:,} bytes"
    )

    print(
        f"Neural payload: "
        f"{len(neural_encoded):,} bytes"
    )

    print(
        f"Payload + model + metadata: "
        f"{total_with_model:,} bytes"
    )

    # --------------------------------------------------------
    # EXACT RECONSTRUCTION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXACT RECONSTRUCTION TEST"
    )
    print("=" * 70)

    print()

    reconstructed = (
        neural_decompress(
            neural_encoded,
            model,
            data,
            start,
            len(data),
            device
        )
    )

    reconstructed_sha = sha256(
        reconstructed
    )

    exact = (
        reconstructed
        == data
    )

    print(
        f"Original SHA256:"
    )

    print(
        original_sha
    )

    print()

    print(
        f"Reconstructed SHA256:"
    )

    print(
        reconstructed_sha
    )

    print()

    print(
        "Exact reconstruction: "
        + (
            "YES"
            if exact
            else
            "NO"
        )
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        f"{'Representation':<38}"
        f"{'Bytes':>12}"
        f"{'Ratio':>12}"
    )

    print("-" * 64)

    original_size = len(data)

    print(
        f"{'Original':<38}"
        f"{original_size:>12,}"
        f"{1.0000:>12.4f}x"
    )

    print(
        f"{'zlib':<38}"
        f"{len(raw_zlib):>12,}"
        f"{original_size / len(raw_zlib):>12.4f}x"
    )

    print(
        f"{'Order-3 arithmetic':<38}"
        f"{len(order3_encoded):>12,}"
        f"{original_size / len(order3_encoded):>12.4f}x"
    )

    print(
        f"{'Neural arithmetic payload':<38}"
        f"{len(neural_encoded):>12,}"
        f"{original_size / len(neural_encoded):>12.4f}x"
    )

    print(
        f"{'Neural + model + metadata':<38}"
        f"{total_with_model:>12,}"
        f"{original_size / total_with_model:>12.4f}x"
    )

    print()

    # --------------------------------------------------------
    # PAYLOAD ANALYSIS
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "PAYLOAD ANALYSIS"
    )
    print("=" * 70)

    print()

    print(
        f"Held-out bytes: "
        f"{test_bytes:,}"
    )

    print(
        f"Neural coded bytes: "
        f"{len(neural_encoded):,}"
    )

    print(
        f"Neural bits/byte "
        f"(actual): "
        f"{len(neural_encoded) * 8 / test_bytes:.6f}"
    )

    print(
        f"Neural cross entropy: "
        f"{cross_entropy:.6f}"
    )

    print(
        f"Order-3 payload: "
        f"{len(order3_encoded):,} bytes"
    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VERDICT"
    )
    print("=" * 70)

    print()

    if exact:

        print(
            "Lossless verification: PASS"
        )

    else:

        print(
            "Lossless verification: FAIL"
        )

    if (
        len(neural_encoded)
        < len(order3_encoded)
    ):

        print(
            "Neural payload beats "
            "order-3 arithmetic."
        )

    else:

        print(
            "Neural payload does NOT "
            "beat order-3 arithmetic yet."
        )

    if (
        total_with_model
        < len(raw_zlib)
    ):

        print(
            "*** COMPLETE NEURAL "
            "REPRESENTATION BEATS ZLIB ***"
        )

    else:

        print(
            "Complete neural representation "
            "does not beat zlib yet."
        )

    print()
    print("DONE.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()