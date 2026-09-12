import tempfile
import unittest
from pathlib import Path

from unlearn_bench.manifest import validate_failure_record, validate_run_manifest

REQUIRED_V1_FIELDS = {
    "run_id",
    "run_set_id",
    "experiment_name",
    "timestamp_utc",
    "git_commit",
    "model_name",
    "model_revision",
    "tokenizer_revision",
    "dataset_hashes",
    "split_hashes",
    "method",
    "configuration",
    "random_seed",
    "hardware",
    "versions",
    "runtime_seconds",
    "peak_vram_bytes",
    "trainable_parameters",
    "final_checkpoint_path",
    "predictions_path",
    "metrics_path",
}


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

    def test_hash_identified_saved_checkpoint_may_be_externally_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("predictions.jsonl", "metrics.json"):
                (root / name).touch()
            manifest = {
                field: None for field in REQUIRED_V1_FIELDS
            }
            manifest.update(
                {
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
                    "checkpoint_status": "saved",
                    "checkpoint_sha256": "b" * 64,
                    "checkpoint_size_bytes": 123,
                }
            )
            validate_run_manifest(manifest, root)

    def test_schema_v2_requires_device_provenance(self):
        manifest = {field: None for field in REQUIRED_V1_FIELDS}
        manifest["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "device fields"):
            validate_run_manifest(manifest)

    def test_schema_v3_requires_config_hash(self):
        manifest = {field: None for field in REQUIRED_V1_FIELDS}
        manifest.update(
            {
                "schema_version": 3,
                "attention_implementation": None,
                "backend": "cpu",
                "device": "cpu",
                "device_fallback": {},
                "dtype": "fp32",
                "reproducibility": {},
                "tokenizer_name": "tokenizer",
                "trust_remote_code": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "deterministic fields"):
            validate_run_manifest(manifest)

    def test_schema_v4_requires_preregistration(self):
        manifest = {field: None for field in REQUIRED_V1_FIELDS}
        manifest.update(
            {
                "schema_version": 4,
                "attention_implementation": None,
                "backend": "cpu",
                "device": "cpu",
                "device_fallback": {},
                "dtype": "fp32",
                "reproducibility": {},
                "tokenizer_name": "tokenizer",
                "trust_remote_code": False,
                "experiment_config_sha256": "a" * 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "confirmatory fields"):
            validate_run_manifest(manifest)

    def test_failure_status_is_classified(self):
        record = {
            "run_id": "run",
            "run_set_id": "set",
            "experiment_name": "experiment",
            "experiment_config_sha256": "a" * 64,
            "timestamp_utc": "2026-09-11T12:00:00+00:00",
            "git_commit": "a" * 40,
            "preregistration_commit": "b" * 40,
            "claim_status": "confirmatory",
            "status": "OOM",
            "stage": "intervention",
            "model_name": "model",
            "method": "method",
            "random_seed": 11,
            "error_type": "OutOfMemoryError",
            "error_message": "out of memory",
            "rerun_policy": "requires addendum",
        }
        validate_failure_record(record)
        with self.assertRaisesRegex(ValueError, "Unknown failure status"):
            validate_failure_record({**record, "status": "FAILED_MAYBE"})
