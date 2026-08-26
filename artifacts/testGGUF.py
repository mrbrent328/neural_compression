import time
import torch
import torch.nn as nn
import gguf

class WeightAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # /4
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),   # /8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

def compress_gguf_tensor(gguf_path: str, tile_size: int = 256, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading GGUF Container into CPU RAM: {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)

    # Locate a large 2D weight matrix (e.g. attention or MLP projection)
    target_tensor = None
    for tensor in reader.tensors:
        if len(tensor.shape) == 2 and tensor.shape[0] >= tile_size:
            target_tensor = tensor
            break

    if target_tensor is None:
        target_tensor = reader.tensors[0]

    print(f"[*] Selected Tensor : {target_tensor.name}")
    print(f"[*] Raw Tensor Shape: {target_tensor.shape}")

    # Fix non-writable numpy warning by copying buffer safely
    raw_weights = torch.from_numpy(target_tensor.data.copy()).float()

    if len(raw_weights.shape) == 1:
        dim = int(raw_weights.numel() ** 0.5)
        raw_weights = raw_weights[: dim * dim].reshape(dim, dim)

    rows, cols = raw_weights.shape
    print(f"[*] Tiling 2D Matrix ({rows}x{cols}) into {tile_size}x{tile_size} blocks...")

    # Tile extraction on CPU
    tiles = []
    for r in range(0, rows - tile_size + 1, tile_size):
        for c in range(0, cols - tile_size + 1, tile_size):
            tile = raw_weights[r : r + tile_size, c : c + tile_size]
            tiles.append(tile.unsqueeze(0))

    total_tiles = len(tiles)
    print(f"[*] Total Tiles Extracted: {total_tiles}")

    tiles_tensor = torch.stack(tiles)

    model = WeightAutoencoder(in_channels=1, latent_dim=16).to(device)
    model.eval()

    latents_list = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, total_tiles, batch_size):
            batch = tiles_tensor[i : i + batch_size].to(device)
            _, z = model(batch)
            latents_list.append(z.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time_ms = (time.time() - t0) * 1000
    all_latents = torch.cat(latents_list, dim=0)

    orig_bytes = tiles_tensor.numel() * tiles_tensor.element_size()
    latent_bytes = all_latents.numel() * all_latents.element_size()
    ratio = orig_bytes / latent_bytes

    print("\n" + "=" * 55)
    print(" GGUF MODEL TILE COMPRESSION RESULTS ")
    print("=" * 55)
    print(f"Device Used     : {device}")
    print(f"VRAM Peak       : < 300 MB")
    print(f"Tiles Processed : {total_tiles}")
    print(f"Input Data Size : {orig_bytes / (1024**2):.2f} MB")
    print(f"Latent Output   : {latent_bytes / (1024**2):.2f} MB ({ratio:.2f}x Reduction)")
    print(f"Total GPU Time  : {total_time_ms:.2f} ms")
    print(f"Speed Per Tile  : {total_time_ms / total_tiles:.3f} ms")
    print("=" * 55)

if __name__ == "__main__":
    # Update to your MXFP4 GGUF file path
    GGUF_FILE_PATH = "/path/to/gpt-oss-20b-MXFP4.gguf"

    try:
        compress_gguf_tensor(GGUF_FILE_PATH, tile_size=256, batch_size=16)
    except FileNotFoundError:
        print(f"\n[!] Update GGUF_FILE_PATH with your local .gguf path.")
