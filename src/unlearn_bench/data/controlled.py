from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from unlearn_bench.utils.reproducibility import sha256_file, sha256_value, write_jsonl

DATASET_VERSION = "controlled-associations-v1"

ASSOCIATIONS = [
    ("alder", "cobalt", "saffron", "retain"),
    ("birch", "topaz", "indigo", "retain"),
    ("cedar", "marble", "silver", "retain"),
    ("dogwood", "violet", "copper", "retain"),
    ("elm", "quartz", "amber", "retain"),
    ("fir", "coral", "onyx", "retain"),
    ("ginkgo", "pearl", "crimson", "forget"),
    ("hazel", "bronze", "azure", "forget"),
    ("ivy", "scarlet", "jade", "forget"),
    ("juniper", "platinum", "ochre", "forget"),
]

TEMPLATES = {
    "train": "The registry code for {entity} is",
    "validation": "In the registry, {entity} has code",
    "test": "Look up {entity}; its assigned code is",
}

UTILITY = {
    "train": [("The opposite of hot is", "cold"), ("A week has seven", "days")],
    "validation": [("Water freezes into", "ice"), ("Birds can often", "fly")],
    "test": [("The sky on a clear day is", "blue"), ("Two plus two equals", "four")],
}


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
    content = {"prompt": prompt, "completion": completion}
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
        "hash": sha256_value(content),
    }


def build_controlled_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Write deterministic, versioned train/validation/test JSONL manifests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dataset": DATASET_VERSION, "files": {}}
    for split, template in TEMPLATES.items():
        rows = []
        for entity, target, replacement, partition in ASSOCIATIONS:
            rows.append(
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
            )
        for index, (prompt, completion) in enumerate(UTILITY[split]):
            rows.append(
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
            )
        rows.sort(key=lambda row: row["id"])
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        summary["files"][split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "count": len(rows),
            "partitions": dict(sorted(Counter(row["partition"] for row in rows).items())),
        }
    validate_manifests(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifests(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for split in TEMPLATES:
        rows = load_records(output_dir / f"{split}.jsonl")
        if len(rows) != 12:
            raise ValueError(f"Expected 12 {split} examples, found {len(rows)}")
        for row in rows:
            required = {"id", "source", "split", "hash", "category", "partition"}
            if required - set(row):
                raise ValueError(f"Malformed record: {row.get('id', '<unknown>')}")
            if row["split"] != split:
                raise ValueError(f"Split mismatch for {row['id']}")
            if row["id"] in seen_ids or row["hash"] in seen_hashes:
                raise ValueError(f"Exact example leakage detected: {row['id']}")
            expected_hash = sha256_value({"prompt": row["prompt"], "completion": row["completion"]})
            if row["hash"] != expected_hash:
                raise ValueError(f"Content hash mismatch: {row['id']}")
            seen_ids.add(row["id"])
            seen_hashes.add(row["hash"])
