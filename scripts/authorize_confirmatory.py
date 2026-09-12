#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.config import resolve_experiment  # noqa: E402
from unlearn_bench.preregistration import (  # noqa: E402
    AUTHORIZED_GPU_HOURS,
    MARKER_PATH,
    PREREGISTRATION_PATH,
)
from unlearn_bench.utils.reproducibility import atomic_json, sha256_file  # noqa: E402

CONFIGS = ("configs/experiments/main.yaml", "configs/experiments/main_mamba.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the reviewed-before-main marker")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--compute-budget-gpu-hours", required=True, type=float)
    args = parser.parse_args()
    if args.compute_budget_gpu_hours != AUTHORIZED_GPU_HOURS:
        raise SystemExit(f"Frozen compute budget is exactly {AUTHORIZED_GPU_HOURS} GPU-hour")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = {
        PREREGISTRATION_PATH,
        *CONFIGS,
        "configs/datasets/controlled_v2.yaml",
        "reports/compute_budget.md",
        "scripts/run_experiment.py",
    }
    experiments = []
    for config_path in CONFIGS:
        config = resolve_experiment(ROOT / config_path)
        experiments.append(config["name"])
        paths.add(f"configs/models/{config['model']}.yaml")
        paths.update(f"configs/methods/{method}.yaml" for method in config["methods"])
    paths.update(
        f"data/controlled/v2/{name}"
        for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
    )
    paths.update(
        str(path.relative_to(ROOT)) for path in (ROOT / "src" / "unlearn_bench").rglob("*.py")
    )
    frozen = {relative: sha256_file(ROOT / relative) for relative in sorted(paths)}
    marker = {
        "schema_version": 1,
        "reviewer": args.reviewer,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_commit": commit,
        "preregistration_path": PREREGISTRATION_PATH,
        "authorized_experiments": experiments,
        "frozen_files_sha256": frozen,
        "compute_budget_gpu_hours": args.compute_budget_gpu_hours,
    }
    for relative, expected in frozen.items():
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise SystemExit(f"Working file differs from preregistration commit: {relative}")
    destination = ROOT / MARKER_PATH
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing marker: {destination}")
    atomic_json(destination, marker)
    print(f"Authorized {len(experiments)} experiments at preregistration commit {commit}")


if __name__ == "__main__":
    main()
