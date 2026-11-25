import tempfile
import unittest
from pathlib import Path
import torch
from E46.e46b_b3_address_training import make_training_tensors, evaluate
from E46.e46b_memory_adapter import NeuralMemoryAdapter


class E46B3Tests(unittest.TestCase):
    def make_data(self):
        facts = []
        for i in range(3):
            facts.append({
                "capture": torch.randn(12),
                "train_queries": [{"vector": torch.randn(12)}],
                "heldout_queries": [{"vector": torch.randn(12)}],
            })
        return {"facts": facts}

    def test_training_tensor_shapes_and_labels(self):
        captures, queries, labels = make_training_tensors(self.make_data())
        self.assertEqual(tuple(captures.shape), (3, 12))
        self.assertEqual(tuple(queries.shape), (3, 12))
        self.assertEqual(labels.tolist(), [0, 1, 2])

    def test_evaluate_reports_retrieval_accuracy(self):
        facts = []
        for i in range(3):
            vector = torch.zeros(8); vector[i] = 1.0
            facts.append({
                "capture": vector,
                "train_queries": [{"vector": vector.clone()}],
                "heldout_queries": [{"vector": vector.clone()}],
            })
        adapter = NeuralMemoryAdapter(8, 4, 3)
        with torch.no_grad():
            adapter.address_writer[0].weight.zero_()
            adapter.address_writer[0].bias.zero_()
            adapter.address_writer[0].weight[:3, :3] = torch.eye(3)
            adapter.address_writer[1].weight.fill_(1.0)
            adapter.address_writer[1].bias.zero_()
            adapter.query_encoder[0].weight.zero_()
            adapter.query_encoder[0].bias.zero_()
            adapter.query_encoder[0].weight[:3, :3] = torch.eye(3)
            adapter.query_encoder[1].weight.fill_(1.0)
            adapter.query_encoder[1].bias.zero_()
        result = evaluate(adapter, facts, "heldout_queries")
        self.assertEqual(result["queries"], 3)
        self.assertGreaterEqual(result["top1_accuracy"], 0.0)
        self.assertLessEqual(result["top1_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
