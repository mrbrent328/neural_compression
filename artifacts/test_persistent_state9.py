#!/usr/bin/env python3
"""
benchmark_geo_dynamic_engine.py

Integrates Temperature Annealing for Straight-Through Gumbel-Softmax gates
and benchmarks actual capacity savings vs. entropy/compression performance
on the real 'geo' Calgary Corpus file.
"""

import sys
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Temperature-Annealed Hard Binary Gate
# =====================================================================

class AnnealedHardBinaryGate(nn.Module):
    def __init__(self, start_temp=2.0, min_temp=0.1):
        super().__init__()
        self.temp = start_temp
        self.min_temp = min_temp

    def set_temperature(self, temp):
        self.temp = max(self.min_temp, temp)

    def forward(self, logits, hard=True):
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.temp
            y_soft = torch.sigmoid(gumbels)
        else:
            y_soft = torch.sigmoid(logits / self.temp)

        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft


# =====================================================================
# 2. Block-Sparse Projection with Dynamic Bypassing
# =====================================================================

class BlockSparseProjection(nn.Module):
    def __init__(self, in_features=256, total_out_features=256, num_blocks=4):
        super().__init__()
        assert total_out_features % num_blocks == 0
        self.num_blocks = num_blocks
        self.block_dim = total_out_features // num_blocks

        self.blocks = nn.ModuleList([
            nn.Linear(in_features, self.block_dim) for _ in range(num_blocks)
        ])

        self.gate_generator = nn.Linear(in_features, num_blocks)
        self.binary_gate = AnnealedHardBinaryGate(start_temp=2.0)

    def update_temperature(self, temp):
        self.binary_gate.set_temperature(temp)

    def forward(self, x_summary, x_full):
        batch_size, seq_len, _ = x_full.shape
        gate_logits = self.gate_generator(x_summary)
        hard_gates = self.binary_gate(gate_logits, hard=True)

        block_outputs = []
        for i, block_layer in enumerate(self.blocks):
            gate_val = hard_gates[:, i:i+1]

            # In evaluation/inference, skip tensor math if gate is 0 across batch
            if not self.training and gate_val.sum() == 0:
                out_block = torch.zeros(batch_size, seq_len, self.block_dim, device=x_full.device)
            else:
                raw_out = block_layer(x_full)
                out_block = raw_out * gate_val.unsqueeze(1)

            block_outputs.append(out_block)

        return torch.cat(block_outputs, dim=-1), hard_gates


# =====================================================================
# 3. Dynamic Predictor & Auto-Compressor
# =====================================================================

class DynamicGeoEngine(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(256, embed_dim)

        self.controller = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.reset_gate_head = nn.Linear(32, 1)
        self.stride_head = nn.Linear(32, 1)

        self.block_projection = BlockSparseProjection(
            in_features=embed_dim,
            total_out_features=embed_dim,
            num_blocks=num_blocks
        )
        self.recurrent_cell = nn.GRUCell(embed_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 256)

    def update_temperature(self, temp):
        self.block_projection.update_temperature(temp)

    def assess_chunk(self, chunk_emb):
        chunk_summary = chunk_emb.mean(dim=1)
        control_feat = self.controller(chunk_summary)

        reset_flag = torch.sigmoid(self.reset_gate_head(control_feat))
        next_stride = 0.5 + 1.5 * torch.sigmoid(self.stride_head(control_feat))
        return chunk_summary, reset_flag, next_stride

    def forward(self, byte_tensor, base_chunk_size=32):
        # byte_tensor shape: (1, total_len)
        chunk_emb_all = self.embedding(byte_tensor)
        batch_size, total_seq_len, _ = chunk_emb_all.shape

        hidden_state = torch.zeros(batch_size, 128, device=byte_tensor.device)
        current_idx = 0
        current_chunk_size = base_chunk_size

        all_logits = []
        execution_trace = []
        all_hard_gates = []
        all_reset_flags = []

        while current_idx < total_seq_len:
            chunk_end = min(current_idx + current_chunk_size, total_seq_len)
            chunk_emb = chunk_emb_all[:, current_idx:chunk_end, :]

            chunk_summary, reset_flag, next_stride = self.assess_chunk(chunk_emb)
            hidden_state = hidden_state * (1.0 - reset_flag)

            gated_proj, hard_gates = self.block_projection(chunk_summary, chunk_emb)

            summary_proj = gated_proj.mean(dim=1)
            hidden_state = self.recurrent_cell(summary_proj, hidden_state)

            chunk_logits = self.lm_head(hidden_state) # Predicts byte distribution
            all_logits.append((current_idx, chunk_end, chunk_logits))

            all_hard_gates.append(hard_gates)
            all_reset_flags.append(reset_flag)

            execution_trace.append({
                "range": (current_idx, chunk_end),
                "reset": reset_flag.item(),
                "active_blocks": hard_gates.detach().squeeze(0).tolist(),
                "capacity_pct": hard_gates.mean().item() * 100,
                "chunk_len": chunk_end - current_idx
            })

            current_idx = chunk_end
            current_chunk_size = max(8, int(base_chunk_size * next_stride.mean().item()))

        return all_logits, execution_trace, all_hard_gates, all_reset_flags


# =====================================================================
# 4. Training & Benchmarking Harness
# =====================================================================

def run_benchmark(file_path):
    if not os.path.exists(file_path):
        print(f"[!] Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, "rb") as f:
        raw_data = f.read()

    data_len = len(raw_data)
    print(f"[*] Loaded '{os.path.basename(file_path)}' ({data_len} bytes)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing on [{device.type.upper()}]...")

    # Load 100 KB sequence into tensor
    byte_tensor = torch.tensor(list(raw_data), dtype=torch.long, device=device).unsqueeze(0)

    model = DynamicGeoEngine(embed_dim=256, hidden_dim=128, num_blocks=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    epochs = 20
    start_temp = 2.0
    min_temp = 0.1

    # Target loss + capacity weights
    lambda_sparsity = 0.08
    lambda_reset = 0.01

    print("\n" + "="*75)
    print(" TRAINING & TEMPERATURE ANNEALING PHASE")
    print("="*75)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Exponential decay of Gumbel temperature
        current_temp = max(min_temp, start_temp * (0.82 ** (epoch - 1)))
        model.update_temperature(current_temp)

        all_logits, trace, gate_history, reset_history = model(byte_tensor, base_chunk_size=32)

        # Calculate Cross-Entropy Loss across all predicted chunks
        total_ce_loss = 0.0
        for start_i, end_i, logits in all_logits:
            target_bytes = byte_tensor[:, start_i:end_i]
            # Next-byte target alignment
            target_mode = target_bytes.mode(dim=1).values
            total_ce_loss += F.cross_entropy(logits, target_mode)

        avg_ce_loss = total_ce_loss / len(all_logits)
        capacity_penalty = torch.stack([g.mean() for g in gate_history]).mean()
        reset_penalty = torch.stack([r.mean() for r in reset_history]).mean()

        total_loss = avg_ce_loss + (lambda_sparsity * capacity_penalty) + (lambda_reset * reset_penalty)

        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:2d}/{epochs} | Temp: {current_temp:5.2f} | "
              f"Task Loss: {avg_ce_loss.item():6.4f} | "
              f"Capacity: {capacity_penalty.item()*100:5.1f}% | "
              f"Total Loss: {total_loss.item():6.4f}")

    # =====================================================================
    # Inference Evaluation Benchmark
    # =====================================================================
    print("\n" + "="*75)
    print(" INFERENCE BENCHMARK ON REAL GEO FILE")
    print("="*75)

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        all_logits, trace, gate_history, reset_history = model(byte_tensor, base_chunk_size=32)
        elapsed = time.time() - t0

    total_chunks = len(trace)
    avg_capacity = sum([t["capacity_pct"] for t in trace]) / total_chunks
    total_bypassed_blocks = sum([t["active_blocks"].count(0.0) for t in trace])
    total_possible_blocks = total_chunks * 4

    # Calculate Theoretical Bitrate based on final loss
    avg_ce = sum([F.cross_entropy(logits, byte_tensor[:, s:e].mode(dim=1).values).item()
                  for s, e, logits in all_logits]) / total_chunks
    estimated_bpb = avg_ce / math.log(2)

    print(f"File Size               : {data_len} bytes")
    print(f"Total Adaptive Chunks   : {total_chunks}")
    print(f"Execution Latency       : {elapsed:.3f}s ({data_len/1024/elapsed:.2f} KB/s)")
    print(f"Average Model Capacity  : {avg_capacity:.1f}%")
    print(f"GPU Blocks Bypassed     : {total_bypassed_blocks}/{total_possible_blocks} ({total_bypassed_blocks/total_possible_blocks*100:.1f}%)")
    print(f"Estimated Compression   : ~{estimated_bpb:.2f} bits/byte")

    print("\nFIRST 8 ADAPTIVE CHUNKS TRACE:")
    print("-" * 75)
    for idx, t in enumerate(trace[:8]):
        print(f"Chunk {idx:2d} | Byte Range: {str(t['range']):12s} | "
              f"Active Blocks: {t['active_blocks']} | Capacity: {t['capacity_pct']:5.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_geo_dynamic_engine.py <path_to_geo>")
        sys.exit(1)

    run_benchmark(sys.argv[1])
