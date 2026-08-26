#!/usr/bin/env python3
"""
adaptive_dynamic_compressor.py

An input-dependent neural engine that inspects incoming chunks to dynamically:
1. Select active matrix compute blocks (skipping unused GPU operations via ST Gumbel-Softmax).
2. Apply continuous soft memory resets.
3. Predict variable contiguous chunk window sizes (without skipping data).
4. Train via a combined Task + Compute Sparsity Loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Straight-Through Gumbel-Softmax / Binary Hard Gate
# =====================================================================

class HardBinaryGate(nn.Module):
    """
    Outputs hard 0.0 or 1.0 during the forward pass to allow conditional execution.
    Uses Straight-Through Estimation during backward pass to allow gradient flow.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits, hard=True):
        if self.training:
            # Gumbel Noise for stochastic exploration during training
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.temperature
            y_soft = torch.sigmoid(gumbels)
        else:
            y_soft = torch.sigmoid(logits / self.temperature)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-Through Estimator trick
            return y_hard - y_soft.detach() + y_soft
        return y_soft


# =====================================================================
# 2. Block-Sparse Projection (Discrete GPU Compute Savings)
# =====================================================================

class BlockSparseProjection(nn.Module):
    """
    Splits a large projection into discrete parallel blocks.
    If a block's gate evaluates to 0, computation for that block is skipped.
    """
    def __init__(self, in_features=512, total_out_features=512, num_blocks=4):
        super().__init__()
        assert total_out_features % num_blocks == 0, "total_out_features must be divisible by num_blocks"
        self.num_blocks = num_blocks
        self.block_dim = total_out_features // num_blocks

        # Physical independent linear layers for GPU execution branches
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, self.block_dim) for _ in range(num_blocks)
        ])

        # Controller predictor for block activation
        self.gate_generator = nn.Linear(in_features, num_blocks)
        self.binary_gate = HardBinaryGate(temperature=0.7)

    def forward(self, x_summary, x_full):
        """
        x_summary: (batch_size, in_features) -> pooled chunk representation
        x_full: (batch_size, seq_len, in_features) -> full chunk sequence
        """
        batch_size, seq_len, _ = x_full.shape

        # Predict discrete 0.0 / 1.0 for each block
        gate_logits = self.gate_generator(x_summary)
        hard_gates = self.binary_gate(gate_logits, hard=True)

        block_outputs = []

        for i, block_layer in enumerate(self.blocks):
            gate_val = hard_gates[:, i:i+1]  # Shape: (batch_size, 1)

            # Execution Bypass: If whole batch gate is 0 in eval mode, bypass tensor math
            if not self.training and gate_val.sum() == 0:
                out_block = torch.zeros(batch_size, seq_len, self.block_dim, device=x_full.device)
            else:
                raw_out = block_layer(x_full)
                # Apply straight-through gate mask
                out_block = raw_out * gate_val.unsqueeze(1)

            block_outputs.append(out_block)

        # Concatenate block tensors back into unified hidden space
        combined_output = torch.cat(block_outputs, dim=-1)
        return combined_output, hard_gates


# =====================================================================
# 3. Main Adaptive Dynamic Chunking Engine
# =====================================================================

class AdaptiveDynamicEngine(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, max_matrix_dim=512, num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_matrix_dim = max_matrix_dim

        # Peek Controller: Inspects chunk summary to make metadata decisions
        self.controller = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Meta decision heads
        self.reset_gate_head = nn.Linear(64, 1)    # Soft memory reset [0, 1]
        self.stride_head = nn.Linear(64, 1)        # Dynamic next window factor

        # Compute Layers
        self.block_projection = BlockSparseProjection(
            in_features=input_dim,
            total_out_features=max_matrix_dim,
            num_blocks=num_blocks
        )
        self.recurrent_cell = nn.GRUCell(max_matrix_dim, hidden_dim)

    def assess_chunk(self, chunk):
        """Pools sequence data to make high-level parameter routing decisions."""
        chunk_summary = chunk.mean(dim=1)  # (batch_size, input_dim)
        control_features = self.controller(chunk_summary)

        # Memory reset signal
        reset_flag = torch.sigmoid(self.reset_gate_head(control_features))

        # Dynamic stride scalar (0.5x to 2.0x base size)
        next_stride_factor = 0.5 + 1.5 * torch.sigmoid(self.stride_head(control_features))

        return chunk_summary, reset_flag, next_stride_factor

    def forward(self, x_sequence, base_chunk_size=16):
        batch_size, total_seq_len, _ = x_sequence.shape
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=x_sequence.device)

        current_idx = 0
        current_chunk_size = base_chunk_size

        outputs = []
        execution_trace = []
        all_hard_gates = []
        all_reset_flags = []

        while current_idx < total_seq_len:
            # 1. Contiguous slice (No gaps or dropped tokens)
            chunk_end = min(current_idx + current_chunk_size, total_seq_len)
            chunk = x_sequence[:, current_idx:chunk_end, :]

            # 2. Inspect chunk
            chunk_summary, reset_flag, next_stride_factor = self.assess_chunk(chunk)

            # 3. Soft Memory Reset
            hidden_state = hidden_state * (1.0 - reset_flag)

            # 4. Dynamic Block-Sparse Matrix Processing
            gated_projection, hard_gates = self.block_projection(chunk_summary, chunk)

            # 5. Recurrent State Update
            summary_proj = gated_projection.mean(dim=1)
            hidden_state = self.recurrent_cell(summary_proj, hidden_state)

            outputs.append(hidden_state)
            all_hard_gates.append(hard_gates)
            all_reset_flags.append(reset_flag)

            execution_trace.append({
                "chunk_range": (current_idx, chunk_end),
                "reset_value": reset_flag.mean().item(),
                "active_blocks": hard_gates.detach().mean(dim=0).tolist(),
                "active_capacity_pct": hard_gates.mean().item() * 100,
                "current_chunk_size": current_chunk_size
            })

            # Advance seamlessly to the exact end boundary
            current_idx = chunk_end

            # Calculate size for the NEXT contiguous block
            current_chunk_size = max(4, int(base_chunk_size * next_stride_factor.mean().item()))

        stacked_outputs = torch.stack(outputs, dim=1)
        return stacked_outputs, execution_trace, all_hard_gates, all_reset_flags


# =====================================================================
# 4. Training Loop & Sparsity Loss Verification
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Adaptive Engine on [{device.type.upper()}]...")

    # Instantiation
    dim = 512
    seq_len = 128
    model = AdaptiveDynamicEngine(input_dim=dim, hidden_dim=256, max_matrix_dim=512, num_blocks=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulated Data Sequence
    dummy_input = torch.randn(2, seq_len, dim, device=device)
    dummy_target = torch.randn(2, 256, device=device) # Synthetic target objective

    # Training Step
    model.train()
    optimizer.zero_grad()

    outputs, trace, gate_history, reset_history = model(dummy_input, base_chunk_size=16)

    # A. Task Loss (e.g. state prediction at last chunk)
    task_loss = F.mse_loss(outputs[:, -1, :], dummy_target)

    # B. Compute Sparsity Penalties
    capacity_penalty = torch.stack([g.mean() for g in gate_history]).mean()
    reset_penalty = torch.stack([r.mean() for r in reset_history]).mean()

    # Total Combined Objective
    lambda_sparsity = 0.05
    lambda_reset = 0.01
    total_loss = task_loss + (lambda_sparsity * capacity_penalty) + (lambda_reset * reset_penalty)

    total_loss.backward()
    optimizer.step()

    # Trace Display
    print("\n" + "="*70)
    print("EXECUTION TRACE (Seamless Coverage & Discrete Block Gating)")
    print("="*70)
    for idx, step in enumerate(trace):
        print(f"Chunk {idx:2d} | Range: {str(step['chunk_range']):10s} | "
              f"Reset Gate: {step['reset_value']:.2f} | "
              f"Active Blocks: {step['active_blocks']} | "
              f"Capacity: {step['active_capacity_pct']:5.1f}%")

    print("\n" + "="*70)
    print(f"LOSS BREAKDOWN:")
    print(f"  Task Loss        : {task_loss.item():.4f}")
    print(f"  Capacity Penalty : {capacity_penalty.item():.4f} (Weight: {lambda_sparsity})")
    print(f"  Reset Penalty    : {reset_penalty.item():.4f} (Weight: {lambda_reset})")
    print(f"  Total Loss       : {total_loss.item():.4f}")
    print("="*70)
