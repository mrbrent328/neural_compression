#!/usr/bin/env python3

import argparse
import hashlib
import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

CONTEXT_SIZE = 8
HIDDEN_SIZE = 64
EMBED_SIZE = 16

DEFAULT_EPOCHS = 400
DEFAULT_SAMPLES = 500_000

BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
PRINT_EVERY = 50

CODE_BITS = 32

TOP = 1 << CODE_BITS
HALF = TOP >> 1
QUARTER = HALF >> 1
THREE_QUARTER = QUARTER * 3

TOTAL_FREQ = 1 << 16

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HASH
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# ARITHMETIC ENCODER
# ============================================================

class ArithmeticEncoder:

    def __init__(self):

        self.low = 0
        self.high = TOP - 1

        self.pending = 0

        self.output = bytearray()

        self.current_byte = 0
        self.bit_count = 0

    def write_bit(self, bit):

        self.current_byte = (
            (self.current_byte << 1)
            | int(bit)
        )

        self.bit_count += 1

        if self.bit_count == 8:

            self.output.append(
                self.current_byte
            )

            self.current_byte = 0
            self.bit_count = 0

    def output_bit_plus_pending(self, bit):

        self.write_bit(bit)

        opposite = 1 - bit

        while self.pending:

            self.write_bit(opposite)

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

            if self.high < HALF:

                self.output_bit_plus_pending(0)

            elif self.low >= HALF:

                self.output_bit_plus_pending(1)

                self.low -= HALF
                self.high -= HALF

            elif (
                self.low >= QUARTER
                and
                self.high < THREE_QUARTER
            ):

                self.pending += 1

                self.low -= QUARTER
                self.high -= QUARTER

            else:

                break

            self.low <<= 1

            self.high = (
                (self.high << 1)
                | 1
            )

    def finish(self):

        self.pending += 1

        if self.low < QUARTER:

            self.output_bit_plus_pending(0)

        else:

            self.output_bit_plus_pending(1)

        if self.bit_count:

            self.current_byte <<= (
                8 - self.bit_count
            )

            self.output.append(
                self.current_byte
            )

        return bytes(self.output)


# ============================================================
# ARITHMETIC DECODER
# ============================================================

class ArithmeticDecoder:

    def __init__(self, data):

        self.data = data

        self.byte_position = 0

        self.current_byte = 0
        self.bits_remaining = 0

        self.low = 0
        self.high = TOP - 1

        self.code = 0

        for _ in range(CODE_BITS):

            self.code = (
                (self.code << 1)
                | self.read_bit()
            )

    def read_bit(self):

        if self.bits_remaining == 0:

            if (
                self.byte_position
                < len(self.data)
            ):

                self.current_byte = (
                    self.data[
                        self.byte_position
                    ]
                )

                self.byte_position += 1

            else:

                self.current_byte = 0

            self.bits_remaining = 8

        bit = (
            self.current_byte
            >> (self.bits_remaining - 1)
        ) & 1

        self.bits_remaining -= 1

        return bit

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

            if self.high < HALF:

                pass

            elif self.low >= HALF:

                self.low -= HALF
                self.high -= HALF
                self.code -= HALF

            elif (
                self.low >= QUARTER
                and
                self.high < THREE_QUARTER
            ):

                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER

            else:

                break

            self.low <<= 1

            self.high = (
                (self.high << 1)
                | 1
            )

            self.code = (
                (self.code << 1)
                | self.read_bit()
            )


# ============================================================
# PROBABILITY TABLE
# ============================================================

def make_frequency_table(probs):

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

    probs = np.maximum(
        probs,
        1e-15
    )

    probs /= probs.sum()

    # Deterministic conversion from
    # probabilities to integer frequencies.

    scaled = (
        probs
        * TOTAL_FREQ
    )

    frequencies = np.floor(
        scaled
    ).astype(np.int64)

    frequencies = np.maximum(
        frequencies,
        1
    )

    current = int(
        frequencies.sum()
    )

    difference = (
        TOTAL_FREQ
        - current
    )

    # Largest-remainder allocation.
    if difference > 0:

        fractional = (
            scaled
            - np.floor(scaled)
        )

        order = np.argsort(
            -fractional
        )

        for i in range(
            difference
        ):

            frequencies[
                order[i % 256]
            ] += 1

    elif difference < 0:

        fractional = (
            scaled
            - np.floor(scaled)
        )

        order = np.argsort(
            fractional
        )

        remaining = -difference

        index = 0

        while remaining:

            symbol = order[
                index % 256
            ]

            if frequencies[
                symbol
            ] > 1:

                frequencies[
                    symbol
                ] -= 1

                remaining -= 1

            index += 1

    # Absolute final correction.
    current = int(
        frequencies.sum()
    )

    frequencies[-1] += (
        TOTAL_FREQ
        - current
    )

    cumulative = np.zeros(
        257,
        dtype=np.int64
    )

    cumulative[1:] = np.cumsum(
        frequencies
    )

    return cumulative


# ============================================================
# NEURAL MODEL
# ============================================================

class NeuralPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            256,
            EMBED_SIZE
        )

        self.network = nn.Sequential(

            nn.Linear(
                CONTEXT_SIZE
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

        return self.network(x)


# ============================================================
# TRAINING DATA
# ============================================================

def make_training_data(
    data,
    start,
    end,
    samples
):

    available = end - start

    count = min(
        samples,
        available
    )

    rng = np.random.default_rng(
        SEED
    )

    if count < available:

        positions = rng.choice(
            np.arange(
                start,
                end
            ),
            size=count,
            replace=False
        )

    else:

        positions = np.arange(
            start,
            end
        )

    x = np.empty(
        (
            len(positions),
            CONTEXT_SIZE
        ),
        dtype=np.uint8
    )

    y = np.empty(
        len(positions),
        dtype=np.int64
    )

    for i, position in enumerate(
        positions
    ):

        x[i] = np.frombuffer(
            data[
                position
                - CONTEXT_SIZE:
                position
            ],
            dtype=np.uint8
        )

        y[i] = data[position]

    return x, y


# ============================================================
# TRAIN
# ============================================================

def train_model(
    data,
    device,
    epochs,
    samples
):

    split = int(
        len(data) * 0.75
    )

    x_np, y_np = (
        make_training_data(
            data,
            CONTEXT_SIZE,
            split,
            samples
        )
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

    model.train()

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

            batch_end = min(
                batch_start
                + BATCH_SIZE,
                len(x)
            )

            indices = permutation[
                batch_start:
                batch_end
            ]

            xb = x[indices]
            yb = y[indices]

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
                * len(indices)
            )

        average_loss = (
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
                f"{average_loss:.8f}"
            )

    print(
        f"Training time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    # IMPORTANT:
    # From this point onward the model never
    # changes and is permanently in eval mode.

    model.eval()

    return model


# ============================================================
# SINGLE-POSITION PREDICTION
# ============================================================

def get_probabilities(
    model,
    context,
    device
):

    # IMPORTANT:
    # Encoder and decoder both call this
    # exact same function one symbol at
    # a time.

    x = torch.tensor(
        [list(context)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        # Convert to CPU float64 immediately.
        # This gives both sides the exact same
        # representation before integer scaling.

        probs = probs.cpu().numpy().astype(
            np.float64
        )

    return probs


# ============================================================
# NEURAL ENCODE
# ============================================================

def neural_encode(
    data,
    start,
    end,
    model,
    device
):

    encoder = ArithmeticEncoder()

    context = bytearray(
        data[
            start - CONTEXT_SIZE:
            start
        ]
    )

    cross_entropy = 0.0

    for position in range(
        start,
        end
    ):

        symbol = data[position]

        probs = get_probabilities(
            model,
            context,
            device
        )

        probability = max(
            float(probs[symbol]),
            1e-15
        )

        cross_entropy += (
            -math.log2(
                probability
            )
        )

        cumulative = (
            make_frequency_table(
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

        # Advance context using the
        # ACTUAL symbol, exactly as the
        # decoder will do after decoding it.

        context = (
            context[1:]
            + bytearray(
                [symbol]
            )
        )

    payload = encoder.finish()

    count = end - start

    return (
        payload,
        cross_entropy / count
    )


# ============================================================
# NEURAL DECODE
# ============================================================

def neural_decode(
    payload,
    prefix,
    count,
    model,
    device
):

    decoder = ArithmeticDecoder(
        payload
    )

    output = bytearray(
        prefix
    )

    context = bytearray(
        prefix[
            -CONTEXT_SIZE:
        ]
    )

    for _ in range(
        count
    ):

        # EXACT SAME probability generation
        # path as encoder.

        probs = get_probabilities(
            model,
            context,
            device
        )

        cumulative = (
            make_frequency_table(
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

        symbol = max(
            0,
            min(
                255,
                int(symbol)
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
# MODEL SIZE
# ============================================================

def model_size(model):

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
# FIRST DIFFERENCE
# ============================================================

def report_difference(
    original,
    reconstructed
):

    limit = min(
        len(original),
        len(reconstructed)
    )

    for i in range(
        limit
    ):

        if original[i] != reconstructed[i]:

            print()
            print(
                "FIRST DIFFERENCE"
            )

            print(
                f"Position: {i}"
            )

            print(
                f"Original: "
                f"{original[i]}"
            )

            print(
                f"Decoded:  "
                f"{reconstructed[i]}"
            )

            left = max(
                0,
                i - 16
            )

            right = min(
                limit,
                i + 17
            )

            print()

            print(
                "Original context:"
            )

            print(
                repr(
                    original[left:right]
                )
            )

            print()

            print(
                "Decoded context:"
            )

            print(
                repr(
                    reconstructed[left:right]
                )
            )

            return

    if len(original) != len(
        reconstructed
    ):

        print()

        print(
            "Lengths differ."
        )

        print(
            f"Original: "
            f"{len(original)}"
        )

        print(
            f"Decoded:  "
            f"{len(reconstructed)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES
    )

    args = parser.parse_args()

    with open(
        args.file,
        "rb"
    ) as f:

        data = f.read()

    if len(data) <= (
        CONTEXT_SIZE + 1
    ):

        raise ValueError(
            "File is too small."
        )

    original_hash = sha256(
        data
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()

    print(
        "Device:       "
        f"{device}"
    )

    print(
        "File:         "
        f"{args.file}"
    )

    print(
        "Bytes:        "
        f"{len(data):,}"
    )

    print(
        "SHA256:       "
        f"{original_hash}"
    )

    print(
        "Context:      "
        f"{CONTEXT_SIZE}"
    )

    print(
        "Epochs:       "
        f"{args.epochs}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    raw_zlib = zlib.compress(
        data,
        level=9
    )

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
    # TRAIN
    # --------------------------------------------------------

    model = train_model(
        data,
        device,
        args.epochs,
        args.samples
    )

    split = int(
        len(data) * 0.75
    )

    test_count = (
        len(data)
        - split
    )

    print()

    print(
        f"Training bytes: "
        f"{split:,}"
    )

    print(
        f"Test bytes:     "
        f"{test_count:,}"
    )

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    print()

    print(
        "Generating neural arithmetic stream..."
    )

    start = time.time()

    payload, cross_entropy = (
        neural_encode(
            data,
            split,
            len(data),
            model,
            device
        )
    )

    encode_time = (
        time.time()
        - start
    )

    theoretical = (
        cross_entropy
        * test_count
        / 8
    )

    actual_rate = (
        len(payload)
        * 8
        / test_count
    )

    print()

    print(
        f"Cross entropy: "
        f"{cross_entropy:.6f} bits/byte"
    )

    print(
        f"Theoretical payload: "
        f"{theoretical:,.1f} bytes"
    )

    print(
        f"Actual payload: "
        f"{len(payload):,} bytes"
    )

    print(
        f"Actual coded rate: "
        f"{actual_rate:.6f} bits/byte"
    )

    print(
        f"Encoding time: "
        f"{encode_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    print()

    print(
        "Decoding..."
    )

    start = time.time()

    reconstructed = (
        neural_decode(
            payload,
            data[:split],
            test_count,
            model,
            device
        )
    )

    decode_time = (
        time.time()
        - start
    )

    print(
        f"Decoding time: "
        f"{decode_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    reconstructed_hash = sha256(
        reconstructed
    )

    exact = (
        reconstructed
        == data
    )

    print()

    print(
        "Original SHA256:"
    )

    print(
        original_hash
    )

    print()

    print(
        "Reconstructed SHA256:"
    )

    print(
        reconstructed_hash
    )

    print()

    print(
        f"Original bytes:      "
        f"{len(data):,}"
    )

    print(
        f"Reconstructed bytes: "
        f"{len(reconstructed):,}"
    )

    print()

    if exact:

        print(
            "*** EXACT RECONSTRUCTION: YES ***"
        )

    else:

        print(
            "*** EXACT RECONSTRUCTION: NO ***"
        )

        report_difference(
            data,
            reconstructed
        )

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    model_raw, model_compressed = (
        model_size(model)
    )

    metadata = 1024

    total = (
        len(payload)
        + model_compressed
        + metadata
    )

    print()

    print(
        "## Representation"
    )

    print()

    print(
        f"{'Representation':<38}"
        f"{'Bytes':>12}"
        f"{'Ratio':>12}"
    )

    print("-" * 62)

    print(
        f"{'Original':<38}"
        f"{len(data):>12,}"
        f"{1.0:>12.4f}x"
    )

    print(
        f"{'Zlib':<38}"
        f"{len(raw_zlib):>12,}"
        f"{len(data) / len(raw_zlib):>12.4f}x"
    )

    print(
        f"{'Neural arithmetic payload':<38}"
        f"{len(payload):>12,}"
        f"{len(data) / len(payload):>12.4f}x"
    )

    print(
        f"{'Neural + model + metadata':<38}"
        f"{total:>12,}"
        f"{len(data) / total:>12.4f}x"
    )

    print()

    print(
        f"Model raw:        "
        f"{model_raw:,} bytes"
    )

    print(
        f"Model compressed: "
        f"{model_compressed:,} bytes"
    )

    print(
        f"Metadata:         "
        f"{metadata:,} bytes"
    )

    print()

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    if exact:

        print(
            "LOSSLESS VERIFICATION: PASS"
        )

        if len(payload) < len(
            raw_zlib
        ):

            print(
                "Neural payload beats zlib "
                "on the held-out data."
            )

        else:

            print(
                "Neural payload does not "
                "beat zlib yet."
            )

        if total < len(
            raw_zlib
        ):

            print(
                "*** COMPLETE REPRESENTATION "
                "BEATS ZLIB ***"
            )

        else:

            print(
                "Model overhead still "
                "dominates the complete representation."
            )

    else:

        print(
            "LOSSLESS VERIFICATION: FAIL"
        )

        print(
            "Do NOT trust the compression "
            "comparison yet."
        )

    print()

    print(
        "DONE."
    )


if __name__ == "__main__":
    main()