import tempfile
import unittest
import torch
from torch import nn
from E46.e46b_memory_adapter import ActivationHook, NeuralMemoryAdapter, PersistentMemoryBank

class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__(); self.proj = nn.Linear(width, width)
    def forward(self, x): return (torch.tanh(self.proj(x)),)

class ToyModel(nn.Module):
    def __init__(self, width=16):
        super().__init__(); self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(width), ToyBlock(width)])
    def forward(self, x):
        for layer in self.model.layers: x = layer(x)[0]
        return x

class E46BTests(unittest.TestCase):
    def test_bank_persists_and_reloads(self):
        bank = PersistentMemoryBank(4, 3)
        bank.add(torch.randn(2, 4), torch.randn(2, 3), {"source": "unit-test"})
        with tempfile.TemporaryDirectory() as tmp:
            bank.save(tmp + "/bank.pt"); restored = PersistentMemoryBank.load(tmp + "/bank.pt")
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.metadata[0]["source"], "unit-test")
    def test_new_adapter_is_behavior_neutral(self):
        adapter = NeuralMemoryAdapter(16, 8, 6); adapter.write(torch.randn(2, 5, 16))
        x = torch.randn(2, 5, 16); y, scores = adapter.inject(x)
        self.assertTrue(torch.equal(x, y)); self.assertIsNotNone(scores)
    def test_adapter_round_trip(self):
        adapter = NeuralMemoryAdapter(16, 8, 6)
        adapter.write(torch.randn(1, 4, 16), {"id": "memory-1"})
        with tempfile.TemporaryDirectory() as tmp:
            adapter.save(tmp); restored = NeuralMemoryAdapter.load(tmp)
        self.assertEqual(len(restored.bank), 1)
        self.assertTrue(torch.allclose(adapter.address_writer[0].weight,
                                       restored.address_writer[0].weight))
    def test_hook_injects_and_freezes_base(self):
        model = ToyModel(); adapter = NeuralMemoryAdapter(16, 8, 6)
        adapter.write(torch.randn(1, 3, 16)); adapter.residual_gate.data.fill_(0.5)
        x = torch.randn(1, 3, 16); baseline = model(x)
        with ActivationHook(model, adapter, "model.layers.0"):
            changed = model(x)
        self.assertFalse(torch.allclose(baseline, changed))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(any(p.requires_grad for p in adapter.parameters()))
    def test_hook_removal_restores_base_path(self):
        model = ToyModel(); adapter = NeuralMemoryAdapter(16, 8, 6)
        adapter.write(torch.randn(1, 3, 16)); adapter.residual_gate.data.fill_(1.0)
        x = torch.randn(1, 3, 16); baseline = model(x)
        hook = ActivationHook(model, adapter, "model.layers.0").install()
        hooked = model(x); hook.remove(); restored = model(x)
        self.assertFalse(torch.allclose(baseline, hooked))
        self.assertTrue(torch.allclose(baseline, restored))

if __name__ == "__main__": unittest.main()
