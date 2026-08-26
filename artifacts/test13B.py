import time
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(latent_dim=16).to(device)
dummy_image = torch.randn(1, 3, 256, 256, device=device)

model.eval()

# -------------------------------------------------------------
# WARM-UP (Triggers ROCm Kernel Compilation)
# -------------------------------------------------------------
print("[*] Warming up ROCm GPU kernels...")
with torch.no_grad():
    for _ in range(5):
        _ = model.encode(dummy_image)
torch.cuda.synchronize()

# -------------------------------------------------------------
# REAL-TIME BENCHMARK
# -------------------------------------------------------------
t0 = time.time()
with torch.no_grad():
    latents = model.encode(dummy_image)
torch.cuda.synchronize()

elapsed_ms = (time.time() - t0) * 1000

print(f"\nDevice used   : {device}")
print(f"Latent Shape  : {list(latents.shape)}")
print(f"Actual Speed  : {elapsed_ms:.2f} ms per frame")
print(f"Throughput    : {1000 / elapsed_ms:.1f} FPS")
