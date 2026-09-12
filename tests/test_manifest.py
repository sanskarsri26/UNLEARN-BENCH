import tempfile
import unittest
from pathlib import Path

from unlearn_bench.manifest import validate_run_manifest


class ManifestTests(unittest.TestCase):
    def test_required_fields_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint.pt", "predictions.jsonl", "metrics.json"):
                (root / name).touch()
            manifest = {
                "run_id": "run",
                "run_set_id": "set",
                "experiment_name": "experiment",
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "git_commit": "a" * 40,
                "model_name": "model",
                "model_revision": "revision",
                "tokenizer_revision": "revision",
                "dataset_hashes": {"train": "a", "validation": "b", "test": "c"},
                "split_hashes": {"retain": "a", "forget": "b", "utility": "c"},
                "method": "method",
                "configuration": {},
                "random_seed": 1,
                "hardware": {"gpu": None},
                "versions": {"torch": "1", "transformers": None, "cuda": None},
                "runtime_seconds": 1.0,
                "peak_vram_bytes": 0,
                "trainable_parameters": 1,
                "final_checkpoint_path": "checkpoint.pt",
                "predictions_path": "predictions.jsonl",
                "metrics_path": "metrics.json",
            }
            validate_run_manifest(manifest, root)

    def test_missing_field_fails(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_run_manifest({})
