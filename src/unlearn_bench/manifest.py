from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_RUN_FIELDS = {
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

DEVICE_RUN_FIELDS = {"device", "dtype", "backend", "reproducibility", "device_fallback"}


def validate_run_manifest(manifest: dict[str, Any], root: str | Path | None = None) -> None:
    missing = REQUIRED_RUN_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Run manifest is missing fields: {sorted(missing)}")
    if manifest.get("schema_version", 1) >= 2:
        device_missing = DEVICE_RUN_FIELDS - set(manifest)
        if device_missing:
            raise ValueError(f"Schema v2 manifest is missing device fields: {sorted(device_missing)}")
        if manifest["device"] not in {"cuda", "mps", "cpu"}:
            raise ValueError("manifest device must be cuda, mps, or cpu")
        if manifest["dtype"] not in {"fp32", "fp16", "bf16"}:
            raise ValueError("manifest dtype must be fp32, fp16, or bf16")
    if set(manifest["dataset_hashes"]) != {"train", "validation", "test"}:
        raise ValueError("dataset_hashes must contain train, validation, and test")
    if set(manifest["split_hashes"]) != {"retain", "forget", "utility"}:
        raise ValueError("split_hashes must contain retain, forget, and utility")
    if "gpu" not in manifest["hardware"]:
        raise ValueError("hardware must explicitly record GPU, including null on CPU")
    for key in ("torch", "transformers", "cuda"):
        if key not in manifest["versions"]:
            raise ValueError(f"versions must explicitly record {key}")
    if manifest["runtime_seconds"] < 0 or manifest["peak_vram_bytes"] < 0:
        raise ValueError("runtime and peak VRAM cannot be negative")
    if root is not None:
        root = Path(root).resolve()
        for field in ("final_checkpoint_path", "predictions_path", "metrics_path"):
            path = (root / manifest[field]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"Manifest artifact is missing or outside the repository: {field}")
