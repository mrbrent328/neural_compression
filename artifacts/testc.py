import sys, zlib, math, time
import numpy as np
import torch
import torch.nn as nn

def entropy_bits(byte_arr):
    if len(byte_arr) == 0:
        return 0.0
    counts = np.bincount(byte_arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / len(byte_arr)
    return float(-np.sum(probs * np.log2(probs)))

def load(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "llm_chat_log.txt"
    data = load(path)
    n = len(data)
    print("=" * 70)
    print("TEXT PREDICTOR TEST")
    print("=" * 70)
    print(f"File: {path}")
    print(f"Bytes: {n:,}")

    raw_zlib = zlib.compress(data.tobytes(), level=9)
    print(f"Raw zlib: {len(raw_zlib):,} bytes  (ratio {n/len(raw_zlib):.4f}x)")

    # ------------------------------------------------------------
    # PREDICTOR 1 — PREVIOUS BYTE (like the image "previous byte" baseline)
    # ------------------------------------------------------------
    pred1 = np.empty(n, dtype=np.uint8)
    pred1[0] = 0
    pred1[1:] = data[:-1]
    resid1 = (data.astype(np.int16) - pred1.astype(np.int16)) % 256
    resid1 = resid1.astype(np.uint8)
    r1_zlib = zlib.compress(resid1.tobytes(), level=9)
    print()
    print("-" * 70)
    print("PREDICTOR 1 - PREVIOUS BYTE")
    print("-" * 70)
    print(f"Residual entropy: {entropy_bits(resid1):.6f} bits/byte")
    print(f"Zero residual: {100*np.mean(resid1==0):.3f}%")
    print(f"Residual zlib: {len(r1_zlib):,} bytes")

    # ------------------------------------------------------------
    # PREDICTOR 2 — ADAPTIVE ORDER-3 CONTEXT MODEL
    # No side info required: both encoder and decoder build this table
    # on the fly from already-seen bytes, same as real text compressors.
    # This is the text analogue of Paeth: zero transmitted model cost.
    # ------------------------------------------------------------
    ORDER = 3
    table = {}
    pred2 = np.empty(n, dtype=np.uint8)
    ctx = (0,) * ORDER
    for i in range(n):
        b = int(data[i])
        pred2[i] = table.get(ctx, 0)
        counts = table.setdefault(ctx, 0)
        # store most-recent byte seen for this context (cheap adaptive proxy)
        table[ctx] = b
        ctx = ctx[1:] + (b,)
    resid2 = (data.astype(np.int16) - pred2.astype(np.int16)) % 256
    resid2 = resid2.astype(np.uint8)
    r2_zlib = zlib.compress(resid2.tobytes(), level=9)
    print()
    print("-" * 70)
    print("PREDICTOR 2 - ADAPTIVE ORDER-3 CONTEXT MODEL (no model transmitted)")
    print("-" * 70)
    print(f"Residual entropy: {entropy_bits(resid2):.6f} bits/byte")
    print(f"Zero residual: {100*np.mean(resid2==0):.3f}%")
    print(f"Residual zlib: {len(r2_zlib):,} bytes")

    # ------------------------------------------------------------
    # PREDICTOR 3 — TRAINED NEURAL PREDICTOR
    # Context window of previous 8 bytes -> predict next byte (regression,
    # same style as the PNG test). Model weights ARE transmitted.
    # ------------------------------------------------------------
    CTX = 8
    device = "cpu"
    X = np.zeros((n - CTX, CTX), dtype=np.float32)
    y = np.zeros((n - CTX,), dtype=np.float32)
    for i in range(CTX, n):
        X[i - CTX] = data[i - CTX:i].astype(np.float32) / 255.0
        y[i - CTX] = data[i].astype(np.float32) / 255.0

    train_n = min(300_000, len(X))
    idx = np.random.RandomState(0).choice(len(X), size=train_n, replace=False)
    Xt = torch.tensor(X[idx])
    yt = torch.tensor(y[idx]).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(CTX, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print()
    print("-" * 70)
    print("PREDICTOR 3 - TRAINED NEURAL PREDICTOR")
    print("-" * 70)
    print(f"Context window: {CTX} bytes")
    print(f"Training samples: {train_n:,}")
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    t0 = time.time()
    EPOCHS = 400
    for epoch in range(EPOCHS):
        opt.zero_grad()
        out = model(Xt)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.10f}")
    print(f"Training time: {time.time()-t0:.2f} seconds")

    with torch.no_grad():
        full_preds = []
        BATCH = 200_000
        Xall = torch.tensor(X)
        for i in range(0, len(Xall), BATCH):
            out = model(Xall[i:i+BATCH]).squeeze(1).numpy()
            full_preds.append(out)
        full_preds = np.concatenate(full_preds)
    pred3_bytes = np.clip(np.round(full_preds * 255.0), 0, 255).astype(np.uint8)
    pred3 = np.zeros(n, dtype=np.uint8)
    pred3[CTX:] = pred3_bytes
    resid3 = (data.astype(np.int16) - pred3.astype(np.int16)) % 256
    resid3 = resid3.astype(np.uint8)
    r3_zlib = zlib.compress(resid3.tobytes(), level=9)

    raw_model = b"".join(p.detach().numpy().tobytes() for p in model.parameters())
    compressed_model = zlib.compress(raw_model, level=9)

    print(f"Residual entropy: {entropy_bits(resid3):.6f} bits/byte")
    print(f"Zero residual: {100*np.mean(resid3==0):.3f}%")
    print(f"Residual zlib: {len(r3_zlib):,} bytes")
    print(f"Raw model: {len(raw_model):,} bytes")
    print(f"Compressed model: {len(compressed_model):,} bytes")

    # ------------------------------------------------------------
    # FINAL COMPARISON
    # ------------------------------------------------------------
    metadata = 1024
    total_ctx = len(r2_zlib) + metadata          # no model to ship
    total_neural = len(r3_zlib) + len(compressed_model) + metadata

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Representation':<38}{'Bytes':>15}{'Ratio vs raw':>15}")
    print(f"{'Raw text':<38}{n:>15,}{1.0:>15.4f}")
    print(f"{'Raw + zlib':<38}{len(raw_zlib):>15,}{n/len(raw_zlib):>15.4f}")
    print(f"{'Order-3 context + zlib (no model)':<38}{total_ctx:>15,}{n/total_ctx:>15.4f}")
    print(f"{'Neural residual + zlib + model':<38}{total_neural:>15,}{n/total_neural:>15.4f}")
    print()
    print(f"Order-3 context entropy: {entropy_bits(resid2):.6f} bits/byte")
    print(f"Neural residual entropy: {entropy_bits(resid3):.6f} bits/byte")
    if total_neural < total_ctx:
        print(f"Neural predictor BEATS the order-3 context model by {total_ctx-total_neural:,} bytes.")
    else:
        print(f"Neural predictor does NOT beat the order-3 context model.")
        print(f"Difference: {total_neural-total_ctx:,} bytes")
    print("DONE.")

if __name__ == "__main__":
    main()