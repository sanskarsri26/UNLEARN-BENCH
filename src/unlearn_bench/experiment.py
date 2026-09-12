from __future__ import annotations

import json
import math
import platform
import resource
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from unlearn_bench.config import resolve_experiment
from unlearn_bench.data import load_records, validate_dataset
from unlearn_bench.evaluation import evaluate_causal_lm, evaluate_model, metrics_from_predictions
from unlearn_bench.manifest import validate_failure_record, validate_run_manifest
from unlearn_bench.methods import apply_hf_method, apply_method, supervised_train, train_causal_lm
from unlearn_bench.methods.common import clone_model, trainable_parameter_count
from unlearn_bench.methods.hf import clone_causal_lm
from unlearn_bench.models import TinyAssociationLM, build_vocabulary, causal_batch, load_causal_lm
from unlearn_bench.preregistration import load_and_validate_marker
from unlearn_bench.utils.device import memory_metrics, reset_peak_memory, select_device, synchronize
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
        "cudnn": (
            str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        ),
    }


def _hardware(device: str) -> dict[str, Any]:
    gpu = None
    device_memory_bytes = None
    if device == "cuda":
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_memory_bytes = properties.total_memory
    elif device == "mps":
        gpu = "Apple Metal Performance Shaders"
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "gpu": gpu,
        "device_memory_bytes": device_memory_bytes,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }


def _peak_process_memory_bytes() -> int:
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported CI targets report KiB.
    return int(maximum_rss if platform.system() == "Darwin" else maximum_rss * 1024)


def _run_id(experiment: str, model: str, method: str, seed: int, config_hash: str) -> str:
    identity = {
        "experiment": experiment,
        "model": model,
        "method": method,
        "seed": seed,
        "config_hash": config_hash,
    }
    suffix = sha256_value(identity)[:8]
    return f"{experiment}-{method}-s{seed}-{suffix}"


def _failure_class(error: Exception) -> str:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "OOM"
    if isinstance(error, FloatingPointError):
        return "NUMERICAL_FAILURE"
    if isinstance(error, NotImplementedError):
        return "UNSUPPORTED_DEVICE"
    if isinstance(error, (RuntimeError, OSError)):
        return "TECHNICAL_FAILURE"
    return "METHOD_FAILURE"


def _write_failure(
    root: Path,
    run_dir: Path,
    *,
    run_id: str,
    run_set_id: str,
    config: dict[str, Any],
    model_name: str,
    method: str,
    seed: int,
    config_hash: str,
    preregistration_commit: str | None,
    stage: str,
    error: Exception,
) -> None:
    if (run_dir / "manifest.json").exists() or (run_dir / "failure.json").exists():
        raise RuntimeError(f"Refusing to overwrite terminal run cell: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "failure.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "run_set_id": run_set_id,
            "experiment_name": config["name"],
            "experiment_config_sha256": config_hash,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(root),
            "preregistration_commit": preregistration_commit,
            "claim_status": config["claim_status"],
            "status": _failure_class(error),
            "stage": stage,
            "model_name": model_name,
            "method": method,
            "random_seed": seed,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "rerun_policy": "requires documented protocol addendum; never overwrite this record",
        },
    )


def run_experiment(
    config_path: str | Path,
    root: str | Path = ".",
    *,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    root = Path(root).resolve()
    config = resolve_experiment(config_path)
    if config["claim_status"] == "confirmatory":
        changed = {
            key: value
            for key, value in (overrides or {}).items()
            if value != config.get(key)
        }
        if changed:
            raise RuntimeError(
                f"Confirmatory command-line overrides would change the frozen protocol: {changed}"
            )
    config.update(overrides or {})
    config["config_path"] = str(Path(config_path).resolve().relative_to(root))
    experiment_config_sha256 = sha256_value(config)
    calibration_marker = None
    if config.get("requires_calibration_approval", False):
        calibration_marker = load_and_validate_marker(root, config["name"])
    context = select_device(
        config.get("device", "auto"),
        config.get("precision", "auto"),
        allow_mps_fallback=config.get("allow_mps_fallback", False),
    )
    dataset_dir = root / config["dataset_config"]["path"]
    validate_dataset(dataset_dir)
    records = {
        split: load_records(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    model_config = config["model_config"]
    backend = model_config["backend"]
    if backend not in {"tiny_association", "huggingface_causal_lm"}:
        raise RuntimeError(f"Unsupported model backend: {backend}")
    all_records = records["train"] + records["validation"] + records["test"]
    preprocessing_started = time.perf_counter()
    vocabulary = build_vocabulary(all_records) if backend == "tiny_association" else None
    shared_preprocessing_seconds = time.perf_counter() - preprocessing_started
    run_set_id = f"{config['name']}-{experiment_config_sha256[:8]}"
    run_ids = []
    for seed in config["seeds"]:
        expected_ids = [
            _run_id(config["name"], model_config["name"], method, seed, experiment_config_sha256)
            for method in config["methods"]
        ]
        expected_dirs = [root / "results" / "runs" / run_id for run_id in expected_ids]
        if all(
            (run_dir / "manifest.json").is_file() or (run_dir / "failure.json").is_file()
            for run_dir in expected_dirs
        ):
            for run_dir in expected_dirs:
                if (run_dir / "manifest.json").is_file():
                    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                    validate_run_manifest(manifest, root)
                else:
                    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
                    validate_failure_record(failure)
            run_ids.extend(expected_ids)
            continue
        reproducibility = set_seed(seed, deterministic=config.get("deterministic", True))
        load_started = time.perf_counter()
        if backend == "tiny_association":
            assert vocabulary is not None
            base = TinyAssociationLM(len(vocabulary.tokens), model_config["hidden_size"]).to(
                device=context.device, dtype=context.dtype
            )
            full = clone_model(base)
            exact = clone_model(base)
            tokenizer = None
        else:
            base, tokenizer = load_causal_lm(model_config, context)
            full = clone_causal_lm(base)
            exact = clone_causal_lm(base)
        synchronize(context)
        model_load_seconds = time.perf_counter() - load_started
        if backend == "huggingface_causal_lm":
            preprocessing_started = time.perf_counter()
            causal_batch(tokenizer, all_records, context.device)
            synchronize(context)
            preprocessing_seconds = time.perf_counter() - preprocessing_started
        else:
            preprocessing_seconds = shared_preprocessing_seconds
        train_cfg = config["training"]
        full_started = time.perf_counter()
        if backend == "tiny_association":
            supervised_train(
                full,
                vocabulary,
                records["train"],
                steps=train_cfg["steps"],
                learning_rate=train_cfg["learning_rate"],
            )
            full_training_metadata = {
                "steps": train_cfg["steps"],
                "examples_processed": train_cfg["steps"] * len(records["train"]),
                "tokens_processed": None,
            }
        else:
            full_training_metadata = train_causal_lm(
                full,
                tokenizer,
                records["train"],
                steps=train_cfg["steps"],
                learning_rate=train_cfg["learning_rate"],
                batch_size=train_cfg.get("batch_size"),
                seed=seed,
            )
        synchronize(context)
        full_training_runtime = time.perf_counter() - full_started
        retain_train = [row for row in records["train"] if row["partition"] != "forget"]
        exact_started = time.perf_counter()
        if backend == "tiny_association":
            supervised_train(
                exact,
                vocabulary,
                retain_train,
                steps=train_cfg["steps"],
                learning_rate=train_cfg["learning_rate"],
            )
            exact_training_metadata = {
                "steps": train_cfg["steps"],
                "examples_processed": train_cfg["steps"] * len(retain_train),
                "tokens_processed": None,
            }
        else:
            exact_training_metadata = train_causal_lm(
                exact,
                tokenizer,
                retain_train,
                steps=train_cfg.get("exact_steps", train_cfg["steps"]),
                learning_rate=train_cfg["learning_rate"],
                batch_size=train_cfg.get("batch_size"),
                seed=seed + 1,
            )
        synchronize(context)
        exact_training_runtime = time.perf_counter() - exact_started
        for method in config["methods"]:
            run_id = _run_id(
                config["name"], model_config["name"], method, seed, experiment_config_sha256
            )
            run_dir = root / "results" / "runs" / run_id
            if (run_dir / "manifest.json").is_file():
                validate_run_manifest(
                    json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")), root
                )
                run_ids.append(run_id)
                continue
            if (run_dir / "failure.json").is_file():
                validate_failure_record(
                    json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
                )
                run_ids.append(run_id)
                continue
            if run_dir.exists():
                raise RuntimeError(f"Incomplete existing run directory requires review: {run_dir}")

            def record_failure(
                stage: str,
                error: Exception,
                *,
                failure_run_dir: Path = run_dir,
                failure_run_id: str = run_id,
                failure_method: str = method,
                failure_seed: int = seed,
            ) -> None:
                _write_failure(
                    root,
                    failure_run_dir,
                    run_id=failure_run_id,
                    run_set_id=run_set_id,
                    config=config,
                    model_name=model_config["name"],
                    method=failure_method,
                    seed=failure_seed,
                    config_hash=experiment_config_sha256,
                    preregistration_commit=(
                        calibration_marker["preregistration_commit"]
                        if calibration_marker
                        else None
                    ),
                    stage=stage,
                    error=error,
                )

            reset_peak_memory(context)
            synchronize(context)
            intervention_started = time.perf_counter()
            method_config = dict(config["method_configs"][method])
            method_config.update(config.get("method_overrides", {}).get(method, {}))
            stage = "intervention"
            try:
                if backend == "tiny_association":
                    model, method_metadata = apply_method(
                        method,
                        full,
                        base,
                        exact,
                        vocabulary,
                        records["train"],
                        method_config,
                        seed,
                    )
                else:
                    model, method_metadata = apply_hf_method(
                        method,
                        full,
                        base,
                        exact,
                        tokenizer,
                        records["train"],
                        method_config,
                        seed,
                    )
                synchronize(context)
                intervention_runtime = time.perf_counter() - intervention_started
                stage = "evaluation"
                evaluation_started = time.perf_counter()
                if backend == "tiny_association":
                    predictions = evaluate_model(model, exact, vocabulary, records["test"])
                else:
                    predictions = evaluate_causal_lm(model, exact, tokenizer, records["test"])
                synchronize(context)
                evaluation_runtime = time.perf_counter() - evaluation_started
                metrics = metrics_from_predictions(predictions)
                numeric_metrics = [
                    value for value in metrics.values() if isinstance(value, (int, float))
                ]
                if not numeric_metrics or not all(
                    math.isfinite(value) for value in numeric_metrics
                ):
                    raise FloatingPointError(f"Method {method} produced non-finite metrics")
            except Exception as error:
                record_failure(stage, error)
                raise
            setup_runtime = {
                "trained_full": full_training_runtime,
                "exact_retrain": exact_training_runtime,
            }.get(method, 0.0)
            setup_metadata = {
                "trained_full": full_training_metadata,
                "exact_retrain": exact_training_metadata,
            }.get(method, {"steps": 0, "examples_processed": 0, "tokens_processed": 0})
            optimization_seconds = intervention_runtime + setup_runtime
            runtime = optimization_seconds + evaluation_runtime
            timestamp = datetime.now(timezone.utc).isoformat()
            run_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_policy = config.get("checkpoint_policy", "saved")
            checkpoint_started = time.perf_counter()
            try:
                if checkpoint_policy == "saved":
                    checkpoint = run_dir / "checkpoint.pt"
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "vocabulary": vocabulary.tokens if vocabulary is not None else None,
                            "model": model_config,
                        },
                        checkpoint,
                    )
                elif checkpoint_policy == "metadata_only" and "calibration" in config["name"]:
                    checkpoint = run_dir / "checkpoint-metadata.json"
                    atomic_json(
                        checkpoint,
                        {"status": "intentionally_omitted", "reason": "exploratory calibration"},
                    )
                else:
                    raise ValueError(
                        "checkpoint_policy must be saved, or metadata_only for a calibration run"
                    )
            except Exception as error:
                record_failure("checkpoint_write", error)
                raise
            checkpoint_write_seconds = time.perf_counter() - checkpoint_started
            predictions_path = run_dir / "predictions.jsonl"
            metrics_path = run_dir / "metrics.json"
            try:
                write_jsonl(predictions_path, predictions)
                atomic_json(metrics_path, metrics)
            except Exception as error:
                record_failure("result_write", error)
                raise
            measured_memory = memory_metrics(context)
            work_examples = setup_metadata["examples_processed"] + method_metadata.get(
                "examples_processed", 0
            )
            setup_tokens = setup_metadata["tokens_processed"]
            intervention_tokens = method_metadata.get("tokens_processed")
            work_tokens = (
                setup_tokens + intervention_tokens
                if setup_tokens is not None and intervention_tokens is not None
                else None
            )
            manifest = {
                "schema_version": 4 if config["claim_status"] == "confirmatory" else 3,
                "run_id": run_id,
                "run_set_id": run_set_id,
                "experiment_name": config["name"],
                "experiment_config_sha256": experiment_config_sha256,
                "timestamp_utc": timestamp,
                "git_commit": git_commit(root),
                "track": config["track"],
                "claim_status": config["claim_status"],
                "status": "COMPLETED",
                "preregistration_commit": (
                    calibration_marker["preregistration_commit"] if calibration_marker else None
                ),
                "model_name": model_config["name"],
                "model_repository": model_config.get("repository"),
                "model_revision": model_config["revision"],
                "tokenizer_revision": model_config["tokenizer_revision"],
                "tokenizer_name": model_config.get(
                    "tokenizer_repository", model_config.get("repository", model_config["name"])
                ),
                "trust_remote_code": model_config.get("trust_remote_code", False),
                "attention_implementation": model_config.get("attention_implementation"),
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
                    "method": method_config,
                },
                "random_seed": seed,
                "device": context.device,
                "dtype": context.precision,
                "backend": context.backend,
                "device_fallback": {
                    "allowed": config.get("allow_mps_fallback", False),
                    "mps_environment_enabled": context.mps_fallback_enabled,
                    "observed": None if context.mps_fallback_enabled else False,
                    "note": (
                        "PyTorch does not expose per-operation MPS fallback telemetry"
                        if context.mps_fallback_enabled
                        else None
                    ),
                },
                "reproducibility": reproducibility.manifest(),
                "hardware": _hardware(context.device),
                "versions": _versions(),
                "runtime_seconds": runtime,
                "model_load_seconds": model_load_seconds,
                "preprocessing_seconds": preprocessing_seconds,
                "intervention_runtime_seconds": intervention_runtime,
                "evaluation_runtime_seconds": evaluation_runtime,
                "optimization_runtime_seconds": optimization_seconds,
                "attributed_setup_runtime_seconds": setup_runtime,
                "optimization_work": {
                    "setup": setup_metadata,
                    "intervention": {
                        key: method_metadata.get(key)
                        for key in ("steps", "examples_processed", "tokens_processed")
                    },
                },
                "examples_per_second": (
                    work_examples / optimization_seconds if optimization_seconds > 0 else None
                ),
                "tokens_per_second": (
                    work_tokens / optimization_seconds
                    if work_tokens is not None and optimization_seconds > 0
                    else None
                ),
                "peak_vram_bytes": measured_memory["peak_device_memory_bytes"],
                "device_memory": measured_memory,
                "peak_process_memory_bytes": _peak_process_memory_bytes(),
                "trainable_parameters": trainable_parameter_count(model),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_write_seconds": checkpoint_write_seconds,
                "checkpoint_status": checkpoint_policy,
                "checkpoint_sha256": sha256_file(checkpoint),
                "final_checkpoint_path": str(checkpoint.relative_to(root)),
                "predictions_path": str(predictions_path.relative_to(root)),
                "metrics_path": str(metrics_path.relative_to(root)),
                "method_metadata": method_metadata,
            }
            try:
                validate_run_manifest(manifest, root)
                atomic_json(run_dir / "manifest.json", manifest)
            except Exception as error:
                record_failure("manifest_write", error)
                raise
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
