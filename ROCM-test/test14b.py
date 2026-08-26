import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf

class WeightAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)


class GGUFToLatentConverter:
    def __init__(self, gguf_path: str, output_prefix: str, tile_size: int = 256, batch_size: int = 16):
        self.gguf_path = gguf_path
        self.bin_output_path = f"{output_prefix}.latents"
        self.json_index_path = f"{output_prefix}.latents.json"
        self.tile_size = tile_size
        self.batch_size = batch_size

        os.makedirs(os.path.dirname(self.bin_output_path), exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WeightAutoencoder(in_channels=1, latent_dim=16).to(self.device)
        self.model.eval()

    def convert(self):
        print(f"[*] Opening GGUF Container: {self.gguf_path}")
        reader = gguf.GGUFReader(self.gguf_path)
        total_tensors = len(reader.tensors)
        print(f"[*] Found {total_tensors} total tensors to process.\n")

        index_map = {
            "metadata": {
                "tile_size": int(self.tile_size),
                "latent_dim": 16,
                "source_file": os.path.basename(self.gguf_path),
            },
            "tensors": {}
        }

        current_byte_offset = 0
        start_time = time.time()

        with open(self.bin_output_path, "wb") as bin_file:
            for idx, tensor in enumerate(reader.tensors):
                tensor_name = str(tensor.name)
                raw_data = tensor.data
                shape_clean = [int(s) for s in tensor.shape]

                if len(shape_clean) < 2:
                    raw_bytes = raw_data.tobytes()
                    bin_file.write(raw_bytes)
                    index_map["tensors"][tensor_name] = {
                        "type": "uncompressed",
                        "dtype": str(raw_data.dtype),
                        "shape": shape_clean,
                        "offset": int(current_byte_offset),
                        "length": int(len(raw_bytes))
                    }
                    current_byte_offset += len(raw_bytes)
                    print(f"[{idx+1}/{total_tensors}] [UNCOMPRESSED 1D] {tensor_name} ({len(raw_bytes)/1024:.1f} KB)")
                    continue

                raw_weights = torch.from_numpy(raw_data.copy()).float()
                if raw_weights.ndim == 2 and list(raw_weights.shape) != shape_clean:
                    raw_weights = raw_weights.T.contiguous()

                if len(raw_weights.shape) > 2:
                    raw_weights = raw_weights.reshape(-1, raw_weights.shape[-1])

                rows, cols = int(raw_weights.shape[0]), int(raw_weights.shape[1])

                if rows < self.tile_size or cols < self.tile_size:
                    raw_bytes = raw_data.tobytes()
                    bin_file.write(raw_bytes)
                    index_map["tensors"][tensor_name] = {
                        "type": "uncompressed",
                        "dtype": str(raw_data.dtype),
                        "shape": shape_clean,
                        "offset": int(current_byte_offset),
                        "length": int(len(raw_bytes))
                    }
                    current_byte_offset += len(raw_bytes)
                    print(f"[{idx+1}/{total_tensors}] [UNCOMPRESSED SMALL] {tensor_name} ({len(raw_bytes)/1024:.1f} KB)")
                    continue

                pad_r = (self.tile_size - (rows % self.tile_size)) % self.tile_size
                pad_c = (self.tile_size - (cols % self.tile_size)) % self.tile_size

                padded_weights = F.pad(raw_weights, (0, pad_c, 0, pad_r))
                p_rows, p_cols = padded_weights.shape

                tiles = []
                for r in range(0, p_rows, self.tile_size):
                    for c in range(0, p_cols, self.tile_size):
                        tile = padded_weights[r : r + self.tile_size, c : c + self.tile_size]
                        tiles.append(tile.unsqueeze(0))

                tiles_tensor = torch.stack(tiles)
                total_tiles = len(tiles_tensor)

                latents_list = []
                with torch.no_grad():
                    for b in range(0, total_tiles, self.batch_size):
                        batch = tiles_tensor[b : b + self.batch_size].to(self.device)
                        z = self.model.encode(batch)
                        latents_list.append(z.cpu())

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                all_latents = torch.cat(latents_list, dim=0)
                latent_bytes = all_latents.numpy().tobytes()
                bin_file.write(latent_bytes)

                index_map["tensors"][tensor_name] = {
                    "type": "latent_grid",
                    "dtype": "float32",
                    "original_shape": shape_clean,
                    "flattened_2d_shape": [rows, cols],
                    "padded_2d_shape": [p_rows, p_cols],
                    "tiles_count": int(total_tiles),
                    "latent_shape": [int(s) for s in all_latents.shape],
                    "offset": int(current_byte_offset),
                    "length": int(len(latent_bytes))
                }

                current_byte_offset += len(latent_bytes)
                ratio = (tiles_tensor.numel() * 4) / len(latent_bytes)
                print(f"[{idx+1}/{total_tensors}] [LATENT {ratio:.1f}x] {tensor_name} -> {total_tiles} tiles ({len(latent_bytes)/(1024**2):.2f} MB)")

        with open(self.json_index_path, "w") as json_file:
            json.dump(index_map, json_file, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(" FULL MODEL CONVERSION COMPLETE ")
        print("=" * 60)
        print(f"Total Model Time  : {elapsed:.2f} seconds")
        print(f"Binary Latent File: {self.bin_output_path} ({current_byte_offset / (1024**3):.2f} GB)")
        print(f"Index File Saved  : {self.json_index_path}")
        print("=" * 60)


if __name__ == "__main__":
    GGUF_INPUT = "/home/xetal/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
    OUTPUT_PREFIX = "/home/xetal/Desktop/GGUFTest/gpt-oss-20b-latent"

    if os.path.exists(GGUF_INPUT):
        converter = GGUFToLatentConverter(GGUF_INPUT, OUTPUT_PREFIX, tile_size=256, batch_size=16)
        converter.convert()
    else:
        print(f"[!] Path not found: {GGUF_INPUT}")
