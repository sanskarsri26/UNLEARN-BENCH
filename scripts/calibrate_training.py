#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.config import load_yaml  # noqa: E402
from unlearn_bench.data import load_records, validate_manifests  # noqa: E402
from unlearn_bench.evaluation import evaluate_causal_lm, metrics_from_predictions  # noqa: E402
from unlearn_bench.methods.hf import clone_causal_lm, train_causal_lm  # noqa: E402
from unlearn_bench.models import load_causal_lm  # noqa: E402
from unlearn_bench.utils.device import memory_metrics, select_device, synchronize  # noqa: E402
from unlearn_bench.utils.reproducibility import (  # noqa: E402
    atomic_json,
    git_commit,
    set_seed,
    sha256_value,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore pre-confirmatory LM training settings")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    calibration = load_yaml(ROOT / args.config)
    model_config = load_yaml(ROOT / "configs" / "models" / f"{calibration['model']}.yaml")
    context = select_device(calibration["device"], calibration["precision"])
    seed = calibration["seed"]
    set_seed(seed)

    dataset_dir = ROOT / "data" / "controlled" / "v1"
    validate_manifests(dataset_dir)
    records = {
        split: load_records(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    load_started = time.perf_counter()
    base, tokenizer = load_causal_lm(model_config, context)
    synchronize(context)
    load_seconds = time.perf_counter() - load_started

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        checkpoint_started = time.perf_counter()
        torch.save({"state_dict": base.state_dict(), "model": model_config}, checkpoint)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        checkpoint_bytes = checkpoint.stat().st_size

    output = {
        "schema_version": 1,
        "claim_status": "exploratory_calibration",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(ROOT),
        "configuration": calibration,
        "configuration_sha256": sha256_value(calibration),
        "model": model_config,
        "device": context.manifest(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "platform": platform.platform(),
        },
        "load_seconds": load_seconds,
        "checkpoint_probe": {"seconds": checkpoint_seconds, "bytes": checkpoint_bytes},
        "variants": [],
    }
    retain_train = [row for row in records["train"] if row["partition"] != "forget"]
    for variant in calibration["variants"]:
        set_seed(seed)
        full = clone_causal_lm(base)
        exact = clone_causal_lm(base)
        full_started = time.perf_counter()
        full_work = train_causal_lm(
            full,
            tokenizer,
            records["train"],
            steps=variant["steps"],
            learning_rate=variant["learning_rate"],
            batch_size=calibration["batch_size"],
            seed=seed,
        )
        synchronize(context)
        full_seconds = time.perf_counter() - full_started
        exact_started = time.perf_counter()
        exact_work = train_causal_lm(
            exact,
            tokenizer,
            retain_train,
            steps=variant["steps"],
            learning_rate=variant["learning_rate"],
            batch_size=calibration["batch_size"],
            seed=seed + 1,
        )
        synchronize(context)
        exact_seconds = time.perf_counter() - exact_started
        split_metrics = {}
        for split in ("validation", "test"):
            predictions = evaluate_causal_lm(full, exact, tokenizer, records[split])
            split_metrics[split] = metrics_from_predictions(predictions)
        output["variants"].append(
            {
                **variant,
                "full_training_seconds": full_seconds,
                "exact_training_seconds": exact_seconds,
                "full_work": full_work,
                "exact_work": exact_work,
                "metrics": split_metrics,
                "memory": memory_metrics(context),
            }
        )
        del full, exact

    artifact_dir = ROOT / "results" / "calibration"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    short_commit = output["git_commit"][:7]
    destination = artifact_dir / f"{calibration['name']}-{short_commit}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite calibration artifact: {destination}")
    atomic_json(destination, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
