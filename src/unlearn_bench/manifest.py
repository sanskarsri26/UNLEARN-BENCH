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

DEVICE_RUN_FIELDS = {
    "attention_implementation",
    "backend",
    "device",
    "device_fallback",
    "dtype",
    "reproducibility",
    "tokenizer_name",
    "trust_remote_code",
}
DETERMINISTIC_RUN_FIELDS = {"experiment_config_sha256"}
CONFIRMATORY_RUN_FIELDS = {"claim_status", "preregistration_commit", "status"}
FAILURE_STATUSES = {
    "TECHNICAL_FAILURE",
    "OOM",
    "UNSUPPORTED_DEVICE",
    "NUMERICAL_FAILURE",
    "METHOD_FAILURE",
}


def validate_run_manifest(manifest: dict[str, Any], root: str | Path | None = None) -> None:
    missing = REQUIRED_RUN_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"Run manifest is missing fields: {sorted(missing)}")
    if manifest.get("schema_version", 1) >= 2:
        device_missing = DEVICE_RUN_FIELDS - set(manifest)
        if device_missing:
            raise ValueError(
                f"Schema v2 manifest is missing device fields: {sorted(device_missing)}"
            )
        if manifest["device"] not in {"cuda", "mps", "cpu"}:
            raise ValueError("manifest device must be cuda, mps, or cpu")
        if manifest["dtype"] not in {"fp32", "fp16", "bf16"}:
            raise ValueError("manifest dtype must be fp32, fp16, or bf16")
    if manifest.get("schema_version", 1) >= 3:
        deterministic_missing = DETERMINISTIC_RUN_FIELDS - set(manifest)
        if deterministic_missing:
            raise ValueError(
                "Schema v3 manifest is missing deterministic fields: "
                f"{sorted(deterministic_missing)}"
            )
    if manifest.get("schema_version", 1) >= 4:
        confirmatory_missing = CONFIRMATORY_RUN_FIELDS - set(manifest)
        if confirmatory_missing:
            raise ValueError(
                "Schema v4 manifest is missing confirmatory fields: "
                f"{sorted(confirmatory_missing)}"
            )
        if manifest["claim_status"] != "confirmatory":
            raise ValueError("Schema v4 manifests must be confirmatory")
        if manifest["status"] != "COMPLETED":
            raise ValueError("Completed run manifests must have status COMPLETED")
        commit = manifest["preregistration_commit"]
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("preregistration_commit must be a full Git SHA")
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
        for field in ("predictions_path", "metrics_path"):
            path = (root / manifest[field]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"Manifest artifact is missing or outside the repository: {field}")
        checkpoint = (root / manifest["final_checkpoint_path"]).resolve()
        if root not in checkpoint.parents:
            raise ValueError("Manifest checkpoint is outside the repository")
        if not checkpoint.is_file():
            retained_externally = (
                manifest.get("checkpoint_status") == "saved"
                and isinstance(manifest.get("checkpoint_sha256"), str)
                and len(manifest["checkpoint_sha256"]) == 64
                and manifest.get("checkpoint_size_bytes", 0) > 0
            )
            if not retained_externally:
                raise ValueError(
                    "Manifest checkpoint is missing without external-retention metadata"
                )


def validate_failure_record(record: dict[str, Any]) -> None:
    required = {
        "run_id",
        "run_set_id",
        "experiment_name",
        "experiment_config_sha256",
        "timestamp_utc",
        "git_commit",
        "preregistration_commit",
        "claim_status",
        "status",
        "stage",
        "model_name",
        "method",
        "random_seed",
        "error_type",
        "error_message",
        "rerun_policy",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Failure record is missing fields: {sorted(missing)}")
    if record["status"] not in FAILURE_STATUSES:
        raise ValueError(f"Unknown failure status: {record['status']}")
    if record["claim_status"] == "confirmatory":
        commit = record["preregistration_commit"]
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("Confirmatory failure needs a full preregistration Git SHA")
