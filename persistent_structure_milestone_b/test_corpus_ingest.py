import json
import tempfile
import unittest
from pathlib import Path

from corpus_ingest import EXACT_LICENSE_ALLOWLIST, ingest_records, normalize_text, prepare_record


def row(text="hello", license="MIT", **overrides):
    base = {
        "text": text,
        "license": license,
        "identifier": "https://example.org/item/1",
        "collection": "Example",
        "open_type": "Open Source",
        "language": "English",
        "title": "Example title",
        "creator": "Example creator",
        "date": "2026",
        "word_count": 1,
        "token_count": 1,
    }
    base.update(overrides)
    return base


class CorpusIngestTests(unittest.TestCase):
    def test_exact_allowlist(self):
        self.assertIn("MIT", EXACT_LICENSE_ALLOWLIST)
        self.assertNotIn("CC0", EXACT_LICENSE_ALLOWLIST)
        self.assertNotIn("Public Domain", EXACT_LICENSE_ALLOWLIST)
        self.assertNotIn("CC-By-SA", EXACT_LICENSE_ALLOWLIST)

    def test_composite_license_rejected(self):
        item, reason = prepare_record(row(license="MIT, BSD-3-Clause"), "d", "r", "train")
        self.assertIsNone(item)
        self.assertEqual(reason, "license_not_allowlisted")

    def test_missing_license_rejected(self):
        item, reason = prepare_record(row(license=""), "d", "r", "train")
        self.assertIsNone(item)
        self.assertEqual(reason, "license_not_allowlisted")

    def test_unicode_and_newline_normalization(self):
        self.assertEqual(normalize_text("e\u0301\r\nline\rnext"), "\u00e9\nline\nnext")

    def test_code_whitespace_preserved(self):
        self.assertEqual(normalize_text("def f():\n    return 1\n"), "def f():\n    return 1\n")

    def test_nul_rejected(self):
        item, reason = prepare_record(row("bad\x00text"), "d", "r", "train")
        self.assertIsNone(item)
        self.assertEqual(reason, "text_contains_nul")

    def test_cc_by_requires_attribution(self):
        item, reason = prepare_record(row(license="CC-BY-4.0", creator=None), "d", "r", "train")
        self.assertIsNone(item)
        self.assertEqual(reason, "required_attribution_metadata_missing")

    def test_provenance_and_attribution_preserved(self):
        item, reason = prepare_record(row(), "PleIAs/common_corpus", "abc123", "train")
        self.assertIsNone(reason)
        self.assertEqual(item["source"]["revision"], "abc123")
        self.assertEqual(item["attribution"]["creator"], "Example creator")
        self.assertEqual(item["source"]["license"], "MIT")

    def test_dedup_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = ingest_records(
                [row("same"), row("same"), row("different", identifier="https://example.org/2")],
                Path(tmp),
                dataset="d", revision="r", split="train",
                max_records=10, target_bytes=100_000, max_document_bytes=10_000,
            )
            self.assertEqual(manifest["results"]["records_written"], 2)
            self.assertEqual(manifest["results"]["duplicates_rejected"], 1)
            with (Path(tmp) / "corpus.jsonl").open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "same")
            self.assertTrue(manifest["artifacts"]["corpus_jsonl_sha256"])

    def test_output_byte_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = ingest_records(
                [row("x" * 100), row("y" * 100)],
                Path(tmp),
                dataset="d", revision="r", split="train",
                max_records=10, target_bytes=1, max_document_bytes=1000,
            )
            self.assertEqual(manifest["results"]["records_written"], 0)
            self.assertTrue(manifest["results"]["target_size_reached_or_next_record_would_exceed"])


if __name__ == "__main__":
    unittest.main()
