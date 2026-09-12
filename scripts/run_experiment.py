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
    args = parser.parse_args()
    run_ids = run_experiment(ROOT / args.config, ROOT)
    print("Completed runs:")
    for run_id in run_ids:
        print(run_id)


if __name__ == "__main__":
    main()
