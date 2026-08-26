import torch
import torch.nn as nn

class LLMNeuralCompressor:
    def __init__(self, base_llm, vocab_size=32000, hidden_dim=4096, top_k=256, lr=0.01):
        self.base_llm = base_llm.eval() # Base LLM is frozen
        for param in self.base_llm.parameters():
            param.requires_grad = False

        # Tiny dynamic adapter updated online
        self.dynamic_head = nn.Linear(hidden_dim, vocab_size).cuda()
        self.optimizer = torch.optim.SGD(self.dynamic_head.parameters(), lr=lr)
        self.top_k = top_k

    def step(self, input_ids, target_token_id, is_encoding=True, range_coder=None):
        # 1. Forward pass through frozen base LLM
        with torch.no_grad():
            outputs = self.base_llm(input_ids, output_hidden_states=True)
            h_t = outputs.hidden_states[-1][:, -1, :] # Final layer hidden state (1, hidden_dim)

        # 2. Forward pass through dynamic head (Track gradients for dynamic head ONLY)
        self.optimizer.zero_grad()
        logits = self.dynamic_head(h_t).squeeze(0) # (vocab_size,)

        # 3. Deterministic quantization to integer frequencies
        freqs = self._deterministic_logits_to_freqs(logits)
        cum_freqs = torch.cumsum(freqs, dim=0)

        # 4. Range Encode / Decode
        if is_encoding:
            range_coder.encode_symbol(target_token_id, freqs, cum_freqs)
            decoded_token = target_token_id
        else:
            decoded_token = range_coder.decode_symbol(freqs, cum_freqs)

        # 5. Online Backward Pass & SGD Step on Dynamic Head ONLY
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([decoded_token].cuda()))
        loss.backward()
        self.optimizer.step()

        return decoded_token

    def _deterministic_logits_to_freqs(self, logits, scale=65536):
        # Fixed-point CPU conversion to guarantee exact encoder/decoder alignment
        logits_np = logits.detach().cpu().float().numpy()
        logits_np -= logits_np.max()
        probs = np.exp(logits_np) / np.exp(logits_np).sum()
        freqs = (probs * (scale - len(probs))).astype(np.int64) + 1
        return torch.from_numpy(freqs)
