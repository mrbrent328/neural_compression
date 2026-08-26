import os
import json
import torch
import torch.nn as nn
import numpy as np

class WeightDecoder(nn.Module):
    def __init__(self, in_channels=16, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def decode(self, z):
        return self.decoder(z)


class LatentModelLoader:
    def __init__(self, latents_prefix: str, tile_size: int = 256):
        self.bin_path = f"{latents_prefix}.latents"
        self.json_path = f"{latents_prefix}.latents.json"
        self.tile_size = tile_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = WeightDecoder(in_channels=16, out_channels=1).to(self.device)
        self.decoder.eval()

        if not os.path.exists(self.json_path) or os.path.getsize(self.json_path) == 0:
            raise FileNotFoundError(f"Index file {self.json_path} is missing or empty. Please re-run test14b.py first.")

        with open(self.json_path, "r") as f:
            self.index_map = json.load(f)

    def reconstruct_tensor(self, tensor_name: str, bin_file) -> torch.Tensor:
        info = self.index_map["tensors"][tensor_name]
        offset = info["offset"]
        length = info["length"]

        bin_file.seek(offset)
        raw_bytes = bin_file.read(length)

        if info["type"] == "uncompressed":
            dtype = getattr(np, info["dtype"], np.float32)
            arr = np.frombuffer(raw_bytes, dtype=dtype).copy()
            return torch.from_numpy(arr).reshape(info["shape"]).to(self.device)

        elif info["type"] == "latent_grid":
            latent_shape = info["latent_shape"]
            latents_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(latent_shape).copy()
            latents_tensor = torch.from_numpy(latents_np).to(self.device)

            with torch.no_grad():
                reconstructed_tiles = self.decoder.decode(latents_tensor)

            p_rows, p_cols = info["padded_2d_shape"]
            reconstructed_grid = torch.zeros((p_rows, p_cols), device=self.device)

            tile_idx = 0
            for r in range(0, p_rows, self.tile_size):
                for c in range(0, p_cols, self.tile_size):
                    reconstructed_grid[r : r + self.tile_size, c : c + self.tile_size] = (
                        reconstructed_tiles[tile_idx, 0]
                    )
                    tile_idx += 1

            # Crop padding back down to 2D unpadded dimensions
            rows, cols = info["flattened_2d_shape"]
            cropped_grid = reconstructed_grid[:rows, :cols]

            # Reshape back to exact target tensor shape
            orig_shape = info["original_shape"]
            return cropped_grid.reshape(orig_shape)


    def verify_single_layer_inference(self, layer_prefix: str = "blk.0.attn_q"):
        print(f"[*] Reconstructing weights for layer: {layer_prefix} ...")

        with open(self.bin_path, "rb") as bin_file:
            weight = self.reconstruct_tensor(f"{layer_prefix}.weight", bin_file)
            bias = self.reconstruct_tensor(f"{layer_prefix}.bias", bin_file)

        print(f"    -> Reconstructed Weight Shape : {list(weight.shape)} | Device: {weight.device}")
        print(f"    -> Reconstructed Bias Shape   : {list(bias.shape)} | Device: {bias.device}")

        out_features, in_features = weight.shape[0], weight.shape[1]
        linear_layer = nn.Linear(in_features, out_features).to(self.device)

        with torch.no_grad():
            linear_layer.weight.copy_(weight)
            linear_layer.bias.copy_(bias)

        dummy_input = torch.randn(2, 16, in_features, device=self.device)

        print("\n[*] Running Forward Pass Inference Test...")
        with torch.no_grad():
            output = linear_layer(dummy_input)

        print(f"[✓] SUCCESS! Forward pass executed cleanly.")
        print(f"    Input Tensor Shape  : {list(dummy_input.shape)}")
        print(f"    Output Tensor Shape : {list(output.shape)}")
        print(f"    Output Mean / Std   : {output.mean().item():.4f} / {output.std().item():.4f}")


if __name__ == "__main__":
    LATENTS_PREFIX = "/home/xetal/Desktop/gpt-oss-20b-latent"

    if os.path.exists(f"{LATENTS_PREFIX}.latents"):
        loader = LatentModelLoader(LATENTS_PREFIX)
        loader.verify_single_layer_inference(layer_prefix="blk.0.attn_q")
    else:
        print(f"[!] File prefix not found: {LATENTS_PREFIX}")
