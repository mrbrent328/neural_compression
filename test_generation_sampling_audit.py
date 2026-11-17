#!/usr/bin/env python3
"""Focused unit tests for generation_sampling_audit.py."""
import unittest
import torch
from generation_sampling_audit import continuation_metrics, derive_sampling_seed, top_p_filter_logits


class SamplingAuditTests(unittest.TestCase):
    def test_top_p_keeps_crossing_token_and_removes_tail(self):
        logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
        filtered = top_p_filter_logits(logits, 0.70)
        self.assertEqual(int(torch.isfinite(filtered).sum()), 2)
        self.assertAlmostEqual(float(torch.softmax(filtered, dim=-1).sum()), 1.0)

    def test_top_p_one_is_noop(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(top_p_filter_logits(logits, 1.0), logits))

    def test_top_p_rejects_invalid_values(self):
        for value in (0.0, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                top_p_filter_logits(torch.tensor([1.0, 2.0]), value)

    def test_seed_derivation_is_stable_and_separated(self):
        a = derive_sampling_seed(123, "temperature_0.7", "python", 0)
        b = derive_sampling_seed(123, "temperature_0.7", "python", 0)
        c = derive_sampling_seed(123, "temperature_0.7", "python", 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_metrics_detect_repetition_and_invalid_utf8(self):
        metrics = continuation_metrics(
            b"aaaa", [torch.tensor(2.0)] * 4, [torch.tensor(1.0)] * 4,
            [torch.tensor(0.5)] * 4, [torch.tensor(0.8)] * 4,
        )
        self.assertAlmostEqual(metrics["unique_byte_fraction"], 0.25)
        self.assertAlmostEqual(metrics["repeat_3gram_rate"], 0.5)
        self.assertTrue(metrics["utf8_valid"])
        invalid = continuation_metrics(
            b"\xff", [torch.tensor(1.0)], [torch.tensor(1.0)],
            [torch.tensor(1.0)], [torch.tensor(1.0)],
        )
        self.assertFalse(invalid["utf8_valid"])
        self.assertEqual(invalid["replacement_char_count"], 1)


if __name__ == "__main__":
    unittest.main()
