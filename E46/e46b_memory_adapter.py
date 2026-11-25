#!/usr/bin/env python3
"""E46-B: activation-level persistent memory adapter prototype."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F

FORMAT_VERSION = 1

class PersistentMemoryBank:
    """CPU-resident address/payload store with explicit serialization."""
    def __init__(self, address_dim: int, payload_dim: int):
        self.address_dim, self.payload_dim = address_dim, payload_dim
        self.addresses = torch.empty((0, address_dim), dtype=torch.float32)
        self.payloads = torch.empty((0, payload_dim), dtype=torch.float32)
        self.metadata: list[dict[str, Any]] = []
    def __len__(self):
        return int(self.addresses.shape[0])
    def add(self, address: Tensor, payload: Tensor, metadata=None):
        address = address.detach().float().reshape(-1, self.address_dim).cpu()
        payload = payload.detach().float().reshape(-1, self.payload_dim).cpu()
        if address.shape[0] != payload.shape[0]:
            raise ValueError("Address and payload batch sizes differ.")
        self.addresses = torch.cat((self.addresses, F.normalize(address, dim=-1)))
        self.payloads = torch.cat((self.payloads, payload))
        self.metadata.extend([dict(metadata or {}) for _ in range(address.shape[0])])
    def retrieve(self, query: Tensor, top_k: int = 4):
        if not len(self):
            raise RuntimeError("Cannot retrieve from an empty memory bank.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query = F.normalize(query.float(), dim=-1)
        scores = query @ self.addresses.to(query.device).T
        values, indices = scores.topk(min(top_k, len(self)), dim=-1)
        weights = F.softmax(values, dim=-1)
        selected = self.payloads.to(query.device)[indices]
        return (selected * weights.unsqueeze(-1)).sum(dim=-2), values
    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": FORMAT_VERSION, "address_dim": self.address_dim,
                    "payload_dim": self.payload_dim, "addresses": self.addresses,
                    "payloads": self.payloads, "metadata": self.metadata}, path)
    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported memory-bank format version.")
        bank = cls(data["address_dim"], data["payload_dim"])
        bank.addresses, bank.payloads = data["addresses"].float(), data["payloads"].float()
        bank.metadata = list(data.get("metadata", []))
        if bank.addresses.shape[0] != bank.payloads.shape[0]:
            raise ValueError("Corrupt memory bank: address/payload count mismatch.")
        return bank

class NeuralMemoryAdapter(nn.Module):
    """Learned writer/query/reader with a zero-initialized residual gate."""
    def __init__(self, hidden_dim: int, address_dim: int = 128,
                 payload_dim: int = 128, top_k: int = 4):
        super().__init__()
        self.hidden_dim, self.address_dim = hidden_dim, address_dim
        self.payload_dim, self.top_k = payload_dim, top_k
        self.address_writer = nn.Sequential(nn.Linear(hidden_dim, address_dim),
                                            nn.LayerNorm(address_dim))
        self.payload_writer = nn.Sequential(nn.Linear(hidden_dim, payload_dim), nn.Tanh())
        self.query_encoder = nn.Sequential(nn.Linear(hidden_dim, address_dim),
                                           nn.LayerNorm(address_dim))
        self.payload_reader = nn.Sequential(nn.Linear(payload_dim, hidden_dim),
                                            nn.LayerNorm(hidden_dim))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.bank = PersistentMemoryBank(address_dim, payload_dim)
    @staticmethod
    def pool(activation: Tensor):
        if activation.ndim == 2: return activation
        if activation.ndim != 3:
            raise ValueError("Expected activation [batch, hidden] or [batch, seq, hidden].")
        return activation.mean(dim=1)
    def write(self, activation: Tensor, metadata=None):
        pooled = self.pool(activation)
        address = F.normalize(self.address_writer(pooled), dim=-1)
        self.bank.add(address, self.payload_writer(pooled), metadata)
    def recall(self, activation: Tensor):
        query = F.normalize(self.query_encoder(self.pool(activation)), dim=-1)
        payload, scores = self.bank.retrieve(query, self.top_k)
        reader_parameter = next(self.payload_reader.parameters())
        payload = payload.to(device=reader_parameter.device, dtype=reader_parameter.dtype)
        return self.payload_reader(payload), scores
    def inject(self, activation: Tensor):
        if not len(self.bank): return activation, None
        recalled, scores = self.recall(activation)
        if activation.ndim == 3: recalled = recalled.unsqueeze(1)
        return activation + self.residual_gate * recalled, scores
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": FORMAT_VERSION, "hidden_dim": self.hidden_dim,
                    "address_dim": self.address_dim, "payload_dim": self.payload_dim,
                    "top_k": self.top_k, "adapter_state": self.state_dict()},
                   path / "adapter.pt")
        self.bank.save(path / "memory_bank.pt")
    @classmethod
    def load(cls, path, map_location="cpu"):
        path = Path(path)
        data = torch.load(path / "adapter.pt", map_location=map_location, weights_only=False)
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported adapter format version.")
        obj = cls(data["hidden_dim"], data["address_dim"], data["payload_dim"], data["top_k"])
        obj.load_state_dict(data["adapter_state"])
        obj.bank = PersistentMemoryBank.load(path / "memory_bank.pt")
        return obj.to(map_location)

def freeze_base_model(model: nn.Module):
    for parameter in model.parameters(): parameter.requires_grad_(False)

class ActivationHook:
    """Inject at a decoder block output. Inspect named_modules() for its path."""
    def __init__(self, model: nn.Module, adapter: NeuralMemoryAdapter,
                 layer_path: str, freeze_base: bool = True):
        self.model, self.adapter, self.layer_path = model, adapter, layer_path
        self.handle = None
        if freeze_base: freeze_base_model(model)
        try: self.layer = model.get_submodule(layer_path)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Decoder layer path not found: {layer_path}") from exc
        params = list(self.layer.parameters())
        self.adapter.to(device=params[0].device, dtype=params[0].dtype) if params else self.adapter.to("cpu")
    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], Tensor):
                raise TypeError("Decoder output tuple must begin with a hidden-state tensor.")
            updated, _ = self.adapter.inject(output[0])
            return (updated, *output[1:])
        if isinstance(output, Tensor):
            updated, _ = self.adapter.inject(output)
            return updated
        raise TypeError(f"Unsupported decoder block output: {type(output)!r}")
    def install(self):
        if self.handle is not None: raise RuntimeError("Hook already installed.")
        self.handle = self.layer.register_forward_hook(self._hook)
        return self
    def remove(self):
        if self.handle is not None: self.handle.remove(); self.handle = None
    def __enter__(self): return self.install()
    def __exit__(self, *_exc): self.remove()
