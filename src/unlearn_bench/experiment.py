from __future__ import annotations

import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from unlearn_bench.config import resolve_experiment
from unlearn_bench.data import load_records, validate_manifests
from unlearn_bench.evaluation import evaluate_model, metrics_from_predictions
from unlearn_bench.manifest import validate_run_manifest
from unlearn_bench.methods import apply_method, supervised_train
from unlearn_bench.methods.common import clone_model, trainable_parameter_count
from unlearn_bench.models import TinyAssociationLM, build_vocabulary
from unlearn_bench.utils.reproducibility import (
    atomic_json,
    git_commit,
    set_seed,
    sha256_file,
    sha256_value,
    write_jsonl,
)


def _versions() -> dict[str, str | None]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": torch.version.cuda,
    }


def _hardware() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }


def _run_id(experiment: str, method: str, seed: int, timestamp: str) -> str:
    identity = {"experiment": experiment, "method": method, "seed": seed, "at": timestamp}
    suffix = sha256_value(identity)[:8]
    return f"{experiment}-{method}-s{seed}-{suffix}"


def run_experiment(config_path: str | Path, root: str | Path = ".") -> list[str]:
    root = Path(root).resolve()
    config = resolve_experiment(config_path)
    config["config_path"] = str(Path(config_path).resolve().relative_to(root))
    if config.get("requires_calibration_approval", False):
        marker = root / "results" / "manifests" / "CALIBRATION_REVIEWED"
        if not marker.exists():
            raise RuntimeError(
                "Main experiments are gated. Review reports/compute_budget.md and create "
                "results/manifests/CALIBRATION_REVIEWED with reviewer/date before running."
            )
    dataset_dir = root / config["dataset_config"]["path"]
    validate_manifests(dataset_dir)
    records = {
        split: load_records(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    all_records = records["train"] + records["validation"] + records["test"]
    vocabulary = build_vocabulary(all_records)
    model_config = config["model_config"]
    if model_config["backend"] != "tiny_association":
        raise RuntimeError(
            "The v0.1 executable runner supports the calibrated tiny backend. Revision-pinned HF "
            "models are declared for the GPU implementation milestone; see docs/limitations.md."
        )
    invocation_timestamp = datetime.now(timezone.utc).isoformat()
    run_set_id = f"{config['name']}-{sha256_value(invocation_timestamp)[:8]}"
    run_ids = []
    for seed in config["seeds"]:
        set_seed(seed)
        base = TinyAssociationLM(len(vocabulary.tokens), model_config["hidden_size"])
        full = clone_model(base)
        exact = clone_model(base)
        train_cfg = config["training"]
        full_started = time.perf_counter()
        supervised_train(
            full,
            vocabulary,
            records["train"],
            steps=train_cfg["steps"],
            learning_rate=train_cfg["learning_rate"],
        )
        full_training_runtime = time.perf_counter() - full_started
        retain_train = [row for row in records["train"] if row["partition"] != "forget"]
        exact_started = time.perf_counter()
        supervised_train(
            exact,
            vocabulary,
            retain_train,
            steps=train_cfg["steps"],
            learning_rate=train_cfg["learning_rate"],
        )
        exact_training_runtime = time.perf_counter() - exact_started
        for method in config["methods"]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model, method_metadata = apply_method(
                method,
                full,
                base,
                exact,
                vocabulary,
                records["train"],
                config["method_configs"][method],
                seed,
            )
            predictions = evaluate_model(model, exact, vocabulary, records["test"])
            metrics = metrics_from_predictions(predictions)
            intervention_runtime = time.perf_counter() - started
            setup_runtime = {
                "trained_full": full_training_runtime,
                "exact_retrain": exact_training_runtime,
            }.get(method, 0.0)
            runtime = intervention_runtime + setup_runtime
            timestamp = datetime.now(timezone.utc).isoformat()
            run_id = _run_id(config["name"], method, seed, timestamp)
            run_dir = root / "results" / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            checkpoint = run_dir / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocabulary": vocabulary.tokens,
                    "model": model_config,
                },
                checkpoint,
            )
            predictions_path = run_dir / "predictions.jsonl"
            metrics_path = run_dir / "metrics.json"
            write_jsonl(predictions_path, predictions)
            atomic_json(metrics_path, metrics)
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "run_set_id": run_set_id,
                "experiment_name": config["name"],
                "timestamp_utc": timestamp,
                "git_commit": git_commit(root),
                "track": config["track"],
                "claim_status": "exploratory" if "smoke" in config["name"] else "confirmatory",
                "model_name": model_config["name"],
                "model_repository": model_config.get("repository"),
                "model_revision": model_config["revision"],
                "tokenizer_revision": model_config["tokenizer_revision"],
                "dataset": config["dataset"],
                "dataset_hashes": {
                    split: sha256_file(dataset_dir / f"{split}.jsonl") for split in records
                },
                "split_hashes": {
                    partition: sha256_value(
                        [row["hash"] for row in records["train"] if row["partition"] == partition]
                    )
                    for partition in ("retain", "forget", "utility")
                },
                "method": method,
                "configuration": {
                    "experiment": config,
                    "method": config["method_configs"][method],
                },
                "random_seed": seed,
                "hardware": _hardware(),
                "versions": _versions(),
                "runtime_seconds": runtime,
                "intervention_runtime_seconds": intervention_runtime,
                "attributed_setup_runtime_seconds": setup_runtime,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
                "trainable_parameters": trainable_parameter_count(model),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "final_checkpoint_path": str(checkpoint.relative_to(root)),
                "predictions_path": str(predictions_path.relative_to(root)),
                "metrics_path": str(metrics_path.relative_to(root)),
                "method_metadata": method_metadata,
            }
            validate_run_manifest(manifest, root)
            atomic_json(run_dir / "manifest.json", manifest)
            run_ids.append(run_id)
    return run_ids


def find_run(root: str | Path, run_id: str) -> Path:
    path = Path(root).resolve() / "results" / "runs" / run_id
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"Unknown run ID: {run_id}")
    return path


def reevaluate_run(root: str | Path, run_id: str) -> dict:
    path = find_run(root, run_id)
    with (path / "predictions.jsonl").open(encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle if line.strip()]
    metrics = metrics_from_predictions(predictions)
    atomic_json(path / "metrics.json", metrics)
    return metrics
