import json
from pathlib import Path

from .controlled import build_controlled_dataset, load_records, validate_manifests
from .controlled_v2 import build_controlled_v2_dataset, validate_controlled_v2


def validate_dataset(path: str | Path) -> None:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") == "controlled-associations-v2":
        validate_controlled_v2(path)
    else:
        validate_manifests(path)


__all__ = [
    "build_controlled_dataset",
    "build_controlled_v2_dataset",
    "load_records",
    "validate_controlled_v2",
    "validate_dataset",
    "validate_manifests",
]
