#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.data import build_controlled_dataset, build_controlled_v2_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic controlled-unlearning data")
    parser.add_argument("--output", default="data/controlled/v1")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    builder = build_controlled_dataset if args.version == "v1" else build_controlled_v2_dataset
    summary = builder(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
