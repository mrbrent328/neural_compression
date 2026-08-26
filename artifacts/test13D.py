import time
import torch
import torch.nn as nn
import torch.optim as optim

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=16, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = ConvEncoder(latent_dim=16).to(device)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

classifier = LatentClassifier(latent_dim=16, num_classes=10).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

dummy_images = torch.randn(8, 3, 256, 256, device=device)
dummy_labels = torch.randint(0, 10, (8,), device=device)

# --- Warmup ---
print("[*] Warming up JIT kernels...")
with torch.no_grad():
    z = encoder(dummy_images)
_ = classifier(z)
if device.type == "cuda":
    torch.cuda.synchronize()

# --- Benchmark Loop (100 Training Steps) ---
print("[*] Running 100 steady-state training iterations...")
t0 = time.time()

for _ in range(100):
    with torch.no_grad():
        z = encoder(dummy_images)

    optimizer.zero_grad()
    outputs = classifier(z)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

if device.type == "cuda":
    torch.cuda.synchronize()

total_time_ms = (time.time() - t0) * 1000
avg_time_ms = total_time_ms / 100

print(f"\nDevice Used         : {device}")
print(f"Total Time (100 iter): {total_time_ms:.2f} ms")
print(f"Average Step Time   : {avg_time_ms:.2f} ms per batch")
print(f"Throughput          : {1000 / avg_time_ms:.1f} batches/sec")
