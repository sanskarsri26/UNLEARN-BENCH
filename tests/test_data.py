import tempfile
import unittest
from pathlib import Path

from unlearn_bench.data import build_controlled_dataset, load_records, validate_manifests


class ControlledDataTests(unittest.TestCase):
    def test_build_is_deterministic_and_has_no_exact_leakage(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = build_controlled_dataset(first)
            right = build_controlled_dataset(second)
            for split in ("train", "validation", "test"):
                self.assertEqual(left["files"][split]["sha256"], right["files"][split]["sha256"])
                self.assertEqual(left["files"][split]["count"], 12)
                self.assertEqual(
                    left["files"][split]["partitions"], {"forget": 4, "retain": 6, "utility": 2}
                )
            validate_manifests(first)
            rows = sum(
                (
                    load_records(Path(first) / f"{split}.jsonl")
                    for split in ("train", "validation", "test")
                ),
                [],
            )
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            self.assertEqual(len({row["hash"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
