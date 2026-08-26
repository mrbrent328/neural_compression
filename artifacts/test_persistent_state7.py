import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDynamicChunker(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, max_matrix_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_matrix_dim = max_matrix_dim

        # 1. Controller / Assessment Head
        # Inspects the incoming raw chunk to make dynamic decisions
        self.controller = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Decision Generators:
        self.reset_gate_head = nn.Linear(64, 1)        # State reset decision [0, 1]
        self.capacity_head = nn.Linear(64, max_matrix_dim) # Dynamic matrix scaling vector
        self.stride_head = nn.Linear(64, 1)           # Suggests next chunk expansion/contraction

        # 2. Heavy Processing Layer (Dynamic Matrix)
        self.base_projection = nn.Linear(input_dim, max_matrix_dim)
        self.recurrent_cell = nn.GRUCell(max_matrix_dim, hidden_dim)

    def assess_chunk(self, chunk):
        """
        Calculates metadata from the chunk to derive dynamic parameters.
        chunk shape: (batch_size, seq_len, input_dim)
        """
        # Pool across sequence length to get a high-level summary of the chunk
        chunk_summary = chunk.mean(dim=1)
        control_features = self.controller(chunk_summary)

        # A. Reset Flag: High value = "End of topic, reset memory"
        reset_flag = torch.sigmoid(self.reset_gate_head(control_features))

        # B. Dynamic Matrix Gate: Scales active channels in the weight matrix
        # Acts like shrinking/expanding matrix capacity on the fly
        capacity_mask = torch.sigmoid(self.capacity_head(control_features))

        # C. Recommended Stride Multiplier (0.5x to 2.0x base length)
        next_stride_factor = 0.5 + 1.5 * torch.sigmoid(self.stride_head(control_features))

        return reset_flag, capacity_mask, next_stride_factor

    def forward(self, x_sequence, base_chunk_size=16):
        """
        Processes an entire sequence by dynamically chunking and adapting parameters.
        """
        batch_size, total_seq_len, _ = x_sequence.shape

        # Initialize hidden state memory
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=x_sequence.device)

        current_idx = 0
        outputs = []
        execution_trace = []

        while current_idx < total_seq_len:
            # 1. Determine current chunk slice
            chunk_end = min(current_idx + base_chunk_size, total_seq_len)
            chunk = x_sequence[:, current_idx:chunk_end, :]

            # 2. "The Peek": Let the network assess the chunk and decide its parameters
            reset_flag, capacity_mask, next_stride_factor = self.assess_chunk(chunk)

            # 3. Apply Dynamic Memory Reset
            # Soft reset allows gradient flow while clearing past context if needed
            hidden_state = hidden_state * (1.0 - reset_flag)

            # 4. Dynamic Matrix Projection
            # Project data, then dynamically gate/suppress unused matrix sub-dimensions
            raw_projection = self.base_projection(chunk) # (batch, seq_len, max_matrix_dim)
            gated_projection = raw_projection * capacity_mask.unsqueeze(1)

            # 5. Process through recurrent memory cell
            chunk_summary = gated_projection.mean(dim=1)
            hidden_state = self.recurrent_cell(chunk_summary, hidden_state)
            outputs.append(hidden_state)

            # Record metrics for inspection
            execution_trace.append({
                "chunk_range": (current_idx, chunk_end),
                "reset_value": reset_flag.item(),
                "active_capacity_pct": capacity_mask.mean().item(),
                "next_stride_factor": next_stride_factor.item()
            })

            # 6. Adaptively step forward based on predicted stride factor
            adaptive_step = int(base_chunk_size * next_stride_factor.item())
            current_idx += max(1, adaptive_step) # Ensure at least +1 progress

        return outputs, execution_trace

# --- Example Usage ---
if __name__ == "__main__":
    # Create fake sequence: 100 tokens, 512 dimensions
    batch_size = 1
    seq_length = 100
    dim = 512
    dummy_input = torch.randn(batch_size, seq_length, dim)

    # Instantiate model
    model = AdaptiveDynamicChunker(input_dim=dim, hidden_dim=256, max_matrix_dim=512)

    # Run dynamic processing loop
    outputs, trace = model(dummy_input, base_chunk_size=16)

    # Output execution results
    print("--- Execution Trace ---")
    for step in trace:
        print(f"Chunk Range: {step['chunk_range']} | "
              f"Reset Gate: {step['reset_value']:.2f} | "
              f"Matrix Capacity Active: {step['active_capacity_pct']*100:.1f}% | "
              f"Next Step Factor: {step['next_stride_factor']:.2f}x")
