import time
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        # ENCODER: Compress Image -> Latent Space
        # Input: [Batch, 3, H, W] -> Output: [Batch, latent_dim, H/8, W/8]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # Downsample /2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample /4
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1), # Downsample /8
        )

        # DECODER: Reconstruct Latent Space -> Image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Scale outputs back to [0, 1] range
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


# Choose device (GPU/ROCm if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvAutoencoder(latent_dim=16).to(device)
dummy_image = torch.randn(1, 3, 256, 256, device=device) # Simulated 256x256 RGB image

# Measure Encode Speed
t0 = time.time()
with torch.no_grad():
    latents = model.encode(dummy_image)

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_ms = (time.time() - t0) * 1000
print(f"Device used   : {device}")
print(f"Latent Shape  : {list(latents.shape)}")
print(f"Encoding Time : {elapsed_ms:.2f} ms")
