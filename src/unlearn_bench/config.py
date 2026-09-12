from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_EXPERIMENT_KEYS = {"name", "track", "model", "dataset", "methods", "seeds"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def validate_experiment(config: dict[str, Any]) -> None:
    missing = REQUIRED_EXPERIMENT_KEYS - set(config)
    if missing:
        raise ValueError(f"Experiment is missing keys: {sorted(missing)}")
    if config["track"] != "controlled_unlearning":
        raise ValueError("Only the controlled_unlearning track is executable in the reboot v0.1")
    if not config["methods"] or not config["seeds"]:
        raise ValueError("methods and seeds must be non-empty")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be unique")


def resolve_experiment(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = load_yaml(path)
    validate_experiment(config)
    root = path.parents[1]
    resolved = dict(config)
    for key, folder in (("model", "models"), ("dataset", "datasets")):
        reference = root / folder / f"{config[key]}.yaml"
        resolved[f"{key}_config"] = load_yaml(reference)
    method_configs = {}
    for method in config["methods"]:
        method_configs[method] = load_yaml(root / "methods" / f"{method}.yaml")
    resolved["method_configs"] = method_configs
    resolved["config_path"] = str(path)
    return resolved
