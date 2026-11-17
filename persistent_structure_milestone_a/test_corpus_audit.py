import unittest

from corpus_audit import summarize_records


class CorpusAuditTests(unittest.TestCase):
    def test_exact_license_values_and_missing_values_are_reported(self):
        rows = [
            {"identifier": "doc-1", "collection": "OpenSource", "open_type": "code", "license": "MIT", "language": "en", "text": "secret text"},
            {"identifier": "doc-2", "collection": "OpenCulture", "open_type": "culture", "license": " CC-BY-4.0 ", "language": "en", "text": "other secret"},
            {"identifier": "doc-3", "collection": "OpenWeb", "open_type": "web", "license": None, "language": None, "text": "private"},
            {"identifier": "doc-4", "collection": "OpenWeb", "open_type": "web", "license": "", "language": "fr", "text": "private"},
        ]
        result = summarize_records(rows, limit=10)
        values = {item["value"]: item["count"] for item in result["license_values"]}
        self.assertEqual(values, {"MIT": 1, "CC-BY-4.0": 1, "<NULL>": 1, "<BLANK>": 1})
        self.assertEqual(result["records_inspected"], 4)
        self.assertEqual(result["missing_or_blank_counts"]["license"], 2)
        self.assertNotIn("secret text", str(result))
        self.assertNotIn("other secret", str(result))

    def test_limit_is_respected(self):
        rows = ({"license": f"L{i}", "text": "not retained"} for i in range(20))
        result = summarize_records(rows, limit=3)
        self.assertEqual(result["records_inspected"], 3)
        self.assertEqual(len(result["license_values"]), 3)

    def test_nonpositive_limit_rejected(self):
        with self.assertRaises(ValueError):
            summarize_records([], limit=0)


if __name__ == "__main__":
    unittest.main()
