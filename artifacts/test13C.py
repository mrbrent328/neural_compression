import time
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. PRE-TRAINED / FROZEN ENCODER
# =============================================================================

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1), # /8
        )

    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# 2. DOWNSTREAM LATENT CLASSIFIER (Operates ONLY on 'z')
# =============================================================================

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=16, num_classes=10):
        super().__init__()

        # Receives [Batch, 16, 32, 32] directly
        self.classifier = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, stride=2, padding=1), # [Batch, 32, 16, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                  # [Batch, 32, 1, 1]
            nn.Flatten(),                                                  # [Batch, 32]
            nn.Linear(32, num_classes)                                     # [Batch, 10]
        )

    def forward(self, z):
        return self.classifier(z)


# =============================================================================
# 3. TRAINING PIPELINE (NO DECOMPRESSION)
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and freeze encoder
encoder = ConvEncoder(latent_dim=16).to(device)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Instantiate trainable downstream model
classifier = LatentClassifier(latent_dim=16, num_classes=10).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Warm up GPU
dummy_images = torch.randn(8, 3, 256, 256, device=device) # Batch of 8 images
dummy_labels = torch.randint(0, 10, (8,), device=device)

with torch.no_grad():
    _ = encoder(dummy_images)
if device.type == "cuda":
    torch.cuda.synchronize()

print("[*] Starting Latent Training Step...")
t0 = time.time()

# --- STEP 1: Fast Encoding to Latent Tensor z ---
with torch.no_grad():
    z = encoder(dummy_images)  # Shape: [8, 16, 32, 32]

# --- STEP 2: Downstream Forward & Loss ---
optimizer.zero_grad()
outputs = classifier(z)        # Output shape: [8, 10]
loss = criterion(outputs, dummy_labels)

# --- STEP 3: Backward Pass (Only updates classifier, NOT encoder) ---
loss.backward()
optimizer.step()

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_ms = (time.time() - t0) * 1000

print(f"Device Used     : {device}")
print(f"Latent Tensor z : {list(z.shape)}")
print(f"Loss Output     : {loss.item():.4f}")
print(f"Total Step Time : {elapsed_ms:.2f} ms")
