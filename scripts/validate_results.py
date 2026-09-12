#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.manifest import validate_failure_record, validate_run_manifest  # noqa: E402


def main() -> None:
    manifests = sorted((ROOT / "results" / "runs").glob("*/manifest.json"))
    if not manifests:
        raise SystemExit("No result manifests found")
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_run_manifest(manifest, ROOT)
        print(f"valid: {path.relative_to(ROOT)}")
    for path in sorted((ROOT / "results" / "runs").glob("*/failure.json")):
        failure = json.loads(path.read_text(encoding="utf-8"))
        validate_failure_record(failure)
        print(f"valid failure: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
