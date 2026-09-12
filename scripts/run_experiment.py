#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configuration-driven benchmark experiment")
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--allow-mps-fallback", action="store_true", default=None)
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "device": args.device,
            "precision": args.precision,
            "allow_mps_fallback": args.allow_mps_fallback,
        }.items()
        if value is not None
    }
    run_ids = run_experiment(ROOT / args.config, ROOT, overrides=overrides)
    print("Terminal run cells (completed or explicitly failed):")
    for run_id in run_ids:
        print(run_id)


if __name__ == "__main__":
    main()
