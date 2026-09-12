from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from unlearn_bench.data.controlled import ASSOCIATIONS, TEMPLATES, UTILITY
from unlearn_bench.utils.reproducibility import atomic_json, sha256_file, sha256_value, write_jsonl

DATASET_VERSION = "controlled-associations-v2"
EXPECTED_COUNTS = {"train": 12, "validation": 12, "test": 40}
EXPECTED_PARTITIONS = {
    "train": {"forget": 4, "retain": 6, "utility": 2},
    "validation": {"forget": 4, "retain": 6, "utility": 2},
    "test": {"forget": 12, "retain": 18, "utility": 10},
}

CONFIRMATORY_TEMPLATES = (
    "Consult the registry entry for {entity}; the code shown is",
    "Which code is assigned to {entity}? Answer:",
    "For {entity}, the registry lists the identifier",
)

CONFIRMATORY_UTILITY = (
    ("Fresh grass is usually", "green"),
    ("Newly fallen snow is", "white"),
    ("Fire is commonly described as", "hot"),
    ("Three plus three equals", "six"),
    ("A dozen contains", "twelve"),
    ("A triangle has", "three"),
    ("A cat commonly says", "meow"),
    ("The sun rises in the", "east"),
    ("Frozen water is called", "ice"),
    ("The period between sunrise and sunset is", "daytime"),
)


def _record(
    *,
    row_id: str,
    split: str,
    partition: str,
    prompt: str,
    completion: str,
    replacement: str | None,
    category: str,
    group_id: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": DATASET_VERSION,
        "split": split,
        "partition": partition,
        "category": category,
        "group_id": group_id,
        "prompt": prompt,
        "completion": completion,
        "replacement": replacement,
        "hash": sha256_value({"prompt": prompt, "completion": completion}),
    }


def _development_rows(split: str) -> list[dict[str, Any]]:
    template = TEMPLATES[split]
    rows = [
        _record(
            row_id=f"{split}-{partition}-{entity}",
            split=split,
            partition=partition,
            prompt=template.format(entity=entity),
            completion=target,
            replacement=replacement if partition == "forget" else None,
            category="synthetic_fact",
            group_id=f"association-{entity}",
        )
        for entity, target, replacement, partition in ASSOCIATIONS
    ]
    rows.extend(
        _record(
            row_id=f"{split}-utility-{index:02d}",
            split=split,
            partition="utility",
            prompt=prompt,
            completion=completion,
            replacement=None,
            category="general_utility",
            group_id=f"utility-{split}-{index}",
        )
        for index, (prompt, completion) in enumerate(UTILITY[split])
    )
    return rows


def _confirmatory_rows() -> list[dict[str, Any]]:
    rows = []
    for template_index, template in enumerate(CONFIRMATORY_TEMPLATES):
        for entity, target, replacement, partition in ASSOCIATIONS:
            rows.append(
                _record(
                    row_id=f"test-t{template_index}-{partition}-{entity}",
                    split="test",
                    partition=partition,
                    prompt=template.format(entity=entity),
                    completion=target,
                    replacement=replacement if partition == "forget" else None,
                    category="synthetic_fact",
                    group_id=f"association-{entity}",
                )
            )
    rows.extend(
        _record(
            row_id=f"test-utility-{index:02d}",
            split="test",
            partition="utility",
            prompt=prompt,
            completion=completion,
            replacement=None,
            category="general_utility",
            group_id=f"confirmatory-utility-{index}",
        )
        for index, (prompt, completion) in enumerate(CONFIRMATORY_UTILITY)
    )
    return rows


def validate_controlled_v2(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_VERSION:
        raise ValueError("Controlled v2 manifest has the wrong dataset identity")
    if manifest.get("confirmatory_holdout") != "test":
        raise ValueError("Controlled v2 manifest must identify test as the confirmatory holdout")
    if manifest.get("holdout_status") != "SEALED_UNTIL_PREREGISTERED_EXECUTION":
        raise ValueError("Controlled v2 manifest has the wrong holdout status")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for split, expected_count in EXPECTED_COUNTS.items():
        path = output_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} examples, found {len(rows)}")
        if any(not isinstance(row.get("partition"), str) for row in rows):
            raise ValueError(f"Malformed partition in {split}")
        partitions = dict(sorted(Counter(row.get("partition") for row in rows).items()))
        if partitions != EXPECTED_PARTITIONS[split]:
            raise ValueError(f"Unexpected {split} partition counts: {partitions}")
        expected_file = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows),
            "partitions": partitions,
        }
        if manifest.get("files", {}).get(split) != expected_file:
            raise ValueError(f"Controlled v2 manifest mismatch for {split}")
        for row in rows:
            required = {
                "category",
                "completion",
                "group_id",
                "hash",
                "id",
                "partition",
                "prompt",
                "source",
                "split",
            }
            if required - set(row):
                raise ValueError(f"Malformed record: {row.get('id', '<unknown>')}")
            if row["source"] != DATASET_VERSION or row["split"] != split:
                raise ValueError(f"Dataset provenance mismatch for {row['id']}")
            if row["partition"] == "forget" and not row.get("replacement"):
                raise ValueError(f"Forget record lacks a replacement: {row['id']}")
            if row["partition"] != "forget" and row.get("replacement") is not None:
                raise ValueError(f"Non-forget record has a replacement: {row['id']}")
            expected_hash = sha256_value({"prompt": row["prompt"], "completion": row["completion"]})
            if row["hash"] != expected_hash:
                raise ValueError(f"Content hash mismatch: {row['id']}")
            if row["id"] in seen_ids or row["hash"] in seen_hashes:
                raise ValueError(f"Exact example leakage detected: {row['id']}")
            seen_ids.add(row["id"])
            seen_hashes.add(row["hash"])


def build_controlled_v2_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Create v2 with a new confirmatory holdout while preserving v1 unchanged."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        "train": _development_rows("train"),
        "validation": _development_rows("validation"),
        "test": _confirmatory_rows(),
    }
    summary: dict[str, Any] = {
        "dataset": DATASET_VERSION,
        "prior_dataset": "controlled-associations-v1",
        "confirmatory_holdout": "test",
        "holdout_status": "SEALED_UNTIL_PREREGISTERED_EXECUTION",
        "files": {},
    }
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: row["id"])
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        summary["files"][split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows),
            "partitions": dict(sorted(Counter(row["partition"] for row in rows).items())),
        }
    atomic_json(output_dir / "manifest.json", summary)
    validate_controlled_v2(output_dir)
    return summary
