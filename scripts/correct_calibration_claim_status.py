#!/usr/bin/env python3
"""Apply the one-time calibration claim-status provenance correction."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORRECTION = {
    "field": "claim_status",
    "from": "confirmatory",
    "to": "exploratory",
    "reason": "Runner inferred status from experiment name instead of explicit configuration.",
}


def main() -> None:
    paths = sorted((ROOT / "results" / "runs").glob("controlled-*calibration-*/manifest.json"))
    if len(paths) != 36:
        raise SystemExit(f"Expected 36 calibration manifests, found {len(paths)}")
    corrected = 0
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("claim_status") == "exploratory":
            if manifest.get("provenance_correction") != CORRECTION:
                raise SystemExit(f"Unexpected existing correction in {path}")
            continue
        if manifest.get("claim_status") != "confirmatory":
            raise SystemExit(f"Unexpected claim status in {path}")
        manifest["claim_status"] = "exploratory"
        manifest["provenance_correction"] = CORRECTION
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        corrected += 1
    print(f"Corrected {corrected} calibration manifests")


if __name__ == "__main__":
    main()
