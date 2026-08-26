import os
import hashlib
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return SHA-256 of the tensor's exact raw byte representation."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    raw = cpu_tensor.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def compare_tensors(original: torch.Tensor, reconstructed: torch.Tensor):
    """Print exact/hash and numerical reconstruction checks."""
    original_hash = tensor_sha256(original)
    reconstructed_hash = tensor_sha256(reconstructed)

    exact = torch.equal(original.detach().cpu(), reconstructed.detach().cpu())

    diff = (original.detach().cpu() - reconstructed.detach().cpu()).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    exact_elements = torch.count_nonzero(diff == 0).item()
    total_elements = diff.numel()

    print("\n[*] SHA-256 Verification")
    print(f"    Original      : {original_hash}")
    print(f"    Reconstructed : {reconstructed_hash}")

    if original_hash == reconstructed_hash:
        print("[✓] HASH MATCH — byte-for-byte identical.")
    else:
        print("[✗] HASH MISMATCH — reconstructed tensor differs.")

    print(f"    torch.equal   : {exact}")
    print(f"    Exact elements: {exact_elements:,} / {total_elements:,}")
    print(f"    Max abs error : {max_error:.8g}")
    print(f"    Mean abs error: {mean_error:.8g}")

    return original_hash == reconstructed_hash


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

        # The .latents.json index can optionally contain the original tensor
        # bytes only if the encoder stored them. Otherwise, the hash check
        # below can only validate reconstruction consistency, not compare
        # against an external original model.
        print("\n[*] Reconstruction integrity check")

        # Hash the reconstructed tensors so the exact representation is
        # available for comparison with hashes produced by the encoder.
        weight_hash = tensor_sha256(weight)
        bias_hash = tensor_sha256(bias)

        print(f"    Weight SHA-256 : {weight_hash}")
        print(f"    Bias SHA-256   : {bias_hash}")

        # If the encoder stored reference SHA-256 values in the JSON index,
        # compare them automatically.
        tensors_info = self.index_map.get("tensors", {})

        for tensor_name, tensor in (
            (f"{layer_prefix}.weight", weight),
            (f"{layer_prefix}.bias", bias),
        ):
            info = tensors_info.get(tensor_name, {})
            expected_hash = (
                info.get("sha256")
                or info.get("hash")
                or info.get("sha256_hash")
            )

            if expected_hash:
                actual_hash = tensor_sha256(tensor)

                print(f"\n    Checking {tensor_name}")
                print(f"      Expected SHA-256 : {expected_hash}")
                print(f"      Actual SHA-256   : {actual_hash}")

                if expected_hash.lower() == actual_hash.lower():
                    print("      [✓] HASH MATCH")
                else:
                    print("      [✗] HASH MISMATCH")
            else:
                print(f"\n    [!] No reference SHA-256 stored for {tensor_name}.")
                print("        Actual reconstructed hash is shown above.")

        # Forward-pass sanity test.
        out_features, in_features = weight.shape[0], weight.shape[1]
        linear_layer = nn.Linear(in_features, out_features).to(self.device)

        with torch.no_grad():
            linear_layer.weight.copy_(weight)
            linear_layer.bias.copy_(bias)

        dummy_input = torch.randn(2, 16, in_features, device=self.device)

        print("\n[*] Running Forward Pass Inference Test...")
        with torch.no_grad():
            output = linear_layer(dummy_input)

        print("[✓] SUCCESS! Forward pass executed cleanly.")
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
