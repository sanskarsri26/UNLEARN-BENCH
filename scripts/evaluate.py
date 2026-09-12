#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.experiment import reevaluate_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate metrics from a run's raw predictions")
    parser.add_argument("--run", required=True, help="Run ID under results/runs")
    args = parser.parse_args()
    print(json.dumps(reevaluate_run(ROOT, args.run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
