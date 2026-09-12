#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.config import resolve_experiment  # noqa: E402


def main() -> None:
    paths = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
    for path in paths:
        resolve_experiment(path)
        print(f"valid: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
