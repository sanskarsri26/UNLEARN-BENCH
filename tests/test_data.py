import json
import tempfile
import unittest
from pathlib import Path

from unlearn_bench.data import (
    build_controlled_dataset,
    build_controlled_v2_dataset,
    load_records,
    validate_controlled_v2,
    validate_dataset,
    validate_manifests,
)


class ControlledDataTests(unittest.TestCase):
    EXPECTED_HASHES = {
        "train": "6f393e27c4f07702825b7071908a5e33e527bd546fa840800a64713d93cc13d4",
        "validation": "1af1e9f85b1646b4572ce1246a87dd910f471b3a762ddff8cc97c937aa1dfee8",
        "test": "d454b59a2666941a050ac16b28dff64ba75356281ce21538df2c7b545e178ab4",
    }

    def test_build_is_deterministic_and_has_no_exact_leakage(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = build_controlled_dataset(first)
            right = build_controlled_dataset(second)
            for split in ("train", "validation", "test"):
                self.assertEqual(left["files"][split]["sha256"], right["files"][split]["sha256"])
                self.assertEqual(left["files"][split]["sha256"], self.EXPECTED_HASHES[split])
                self.assertEqual(left["files"][split]["path"], f"{split}.jsonl")
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

    def test_v2_is_additive_deterministic_and_has_a_new_holdout(self):
        with (
            tempfile.TemporaryDirectory() as first_root,
            tempfile.TemporaryDirectory() as second_root,
        ):
            first = Path(first_root) / "v2"
            second = Path(second_root) / "v2"
            left = build_controlled_v2_dataset(first)
            right = build_controlled_v2_dataset(second)
            self.assertEqual(left, right)
            self.assertEqual(left["holdout_status"], "SEALED_UNTIL_PREREGISTERED_EXECUTION")
            self.assertEqual(left["files"]["test"]["count"], 40)
            self.assertEqual(
                left["files"]["test"]["partitions"],
                {"forget": 12, "retain": 18, "utility": 10},
            )
            validate_controlled_v2(first)
            validate_dataset(first)
            old_test_path = (
                Path(__file__).resolve().parents[1] / "data" / "controlled" / "v1" / "test.jsonl"
            )
            old_test_hashes = {row["hash"] for row in load_records(old_test_path)}
            new_test_hashes = {row["hash"] for row in load_records(first / "test.jsonl")}
            self.assertTrue(old_test_hashes.isdisjoint(new_test_hashes))

    def test_v2_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "v2"
            build_controlled_v2_dataset(destination)
            with self.assertRaises(FileExistsError):
                build_controlled_v2_dataset(destination)

    def test_v2_validator_rejects_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "v2"
            build_controlled_v2_dataset(destination)
            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["test"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest mismatch for test"):
                validate_controlled_v2(destination)


if __name__ == "__main__":
    unittest.main()
