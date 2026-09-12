from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

MARKER_PATH = Path("results/manifests/CALIBRATION_REVIEWED")
PREREGISTRATION_PATH = "docs/preregistration_main.md"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZED_GPU_HOURS = 1.0


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        command = " ".join(arguments)
        raise RuntimeError(f"Could not verify preregistration with git: {command}") from error


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_marker_structure(marker: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "reviewer",
        "reviewed_at_utc",
        "preregistration_commit",
        "preregistration_path",
        "authorized_experiments",
        "frozen_files_sha256",
        "compute_budget_gpu_hours",
    }
    missing = required - set(marker)
    if missing:
        raise ValueError(f"Calibration marker is missing fields: {sorted(missing)}")
    if marker["schema_version"] != 1:
        raise ValueError("Unsupported calibration marker schema")
    if not isinstance(marker["reviewer"], str) or not marker["reviewer"].strip():
        raise ValueError("Calibration marker reviewer must be non-empty")
    try:
        reviewed_at = datetime.fromisoformat(marker["reviewed_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("Calibration marker timestamp must be ISO 8601") from error
    if reviewed_at.tzinfo is None:
        raise ValueError("Calibration marker timestamp must include a timezone")
    if not SHA_PATTERN.fullmatch(marker["preregistration_commit"]):
        raise ValueError("Calibration marker preregistration_commit must be a full Git SHA")
    if marker["preregistration_path"] != PREREGISTRATION_PATH:
        raise ValueError("Calibration marker preregistration path is not canonical")
    if not marker["authorized_experiments"]:
        raise ValueError("Calibration marker must authorize at least one experiment")
    if marker["compute_budget_gpu_hours"] != AUTHORIZED_GPU_HOURS:
        raise ValueError("Calibration marker compute budget must equal the frozen 1.0 GPU-hour cap")
    frozen = marker["frozen_files_sha256"]
    if PREREGISTRATION_PATH not in frozen or not frozen:
        raise ValueError("Calibration marker must freeze the preregistration document")
    if any(
        not isinstance(path, str) or not DIGEST_PATTERN.fullmatch(digest)
        for path, digest in frozen.items()
    ):
        raise ValueError("Calibration marker frozen file hashes are malformed")


def load_and_validate_marker(root: str | Path, experiment_name: str) -> dict[str, Any]:
    root = Path(root).resolve()
    marker_path = root / MARKER_PATH
    if not marker_path.is_file():
        raise RuntimeError(
            "Confirmatory experiments are gated. Commit docs/preregistration_main.md, review "
            "reports/compute_budget.md, and create results/manifests/CALIBRATION_REVIEWED."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    validate_marker_structure(marker)
    if experiment_name not in marker["authorized_experiments"]:
        raise RuntimeError(f"Calibration marker does not authorize {experiment_name}")
    commit = marker["preregistration_commit"]
    _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Preregistration commit is not an ancestor of the run commit") from error
    for relative, expected in marker["frozen_files_sha256"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(
                f"Frozen preregistration input is missing or outside repository: {relative}"
            )
        current = _digest(path.read_bytes())
        committed = _digest(_git(root, "show", f"{commit}:{relative}"))
        if current != expected or committed != expected:
            raise RuntimeError(f"Frozen preregistration input changed: {relative}")
    return marker
