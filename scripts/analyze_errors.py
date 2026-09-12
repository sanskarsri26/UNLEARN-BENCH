#!/usr/bin/env python3
"""Deterministic, post-confirmatory error analysis from stored Track A predictions."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/unlearn-bench-matplotlib")

PREREGISTRATION_COMMIT = "005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6"
MODELS = ("pythia-160m", "mamba-130m-hf")
METHODS = (
    "continued_retain",
    "sham",
    "counterfactual",
    "gradient_ascent",
    "npo",
    "pcgu",
)
ALL_METHODS = ("trained_full", "exact_retrain", *METHODS)
SEEDS = (11, 29, 47)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cells() -> tuple[dict[tuple[str, str, int], list[dict]], dict[str, dict]]:
    examples = {
        row["id"]: row for row in load_jsonl(ROOT / "data/controlled/v2/test.jsonl")
    }
    cells = {}
    for path in sorted((ROOT / "results/runs").glob("*/manifest.json")):
        manifest = load_json(path)
        if manifest.get("preregistration_commit") != PREREGISTRATION_COMMIT:
            continue
        key = (manifest["model_name"], manifest["method"], manifest["random_seed"])
        if manifest["method"] not in ALL_METHODS:
            continue
        predictions = load_jsonl(ROOT / manifest["predictions_path"])
        if len(predictions) != 40 or {row["id"] for row in predictions} != set(examples):
            raise ValueError(f"Prediction coverage mismatch: {key}")
        cells[key] = predictions
    expected = {
        (model, method, seed)
        for model in MODELS
        for method in ALL_METHODS
        for seed in SEEDS
    }
    if set(cells) != expected:
        raise ValueError(f"Error-analysis matrix mismatch: {sorted(expected - set(cells))}")
    return cells, examples


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def forget_groups(
    cells: dict[tuple[str, str, int], list[dict]], examples: dict[str, dict]
) -> list[dict]:
    rows = []
    groups = sorted(
        {example["group_id"] for example in examples.values() if example["partition"] == "forget"}
    )
    for model in MODELS:
        for method in ALL_METHODS:
            for group in groups:
                values = []
                deltas = []
                for seed in SEEDS:
                    current = by_id(cells[(model, method, seed)])
                    baseline = by_id(cells[(model, "trained_full", seed)])
                    ids = [
                        row_id
                        for row_id, example in examples.items()
                        if example["group_id"] == group
                    ]
                    values.append(mean(float(current[row_id]["oracle_kl"]) for row_id in ids))
                    deltas.append(
                        mean(
                            float(current[row_id]["oracle_kl"])
                            - float(baseline[row_id]["oracle_kl"])
                            for row_id in ids
                        )
                    )
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "group_id": group,
                        "oracle_kl_mean": mean(values),
                        "oracle_kl_sd": stdev(values),
                        "delta_vs_full_mean": mean(deltas),
                        "delta_vs_full_sd": stdev(deltas),
                    }
                )
    for model in MODELS:
        for method in ALL_METHODS:
            selected = [
                row for row in rows if row["model"] == model and row["method"] == method
            ]
            selected.sort(key=lambda row: (-row["oracle_kl_mean"], row["group_id"]))
            for rank, row in enumerate(selected, start=1):
                row["hardest_rank"] = rank
    return rows


def damage_rows(
    cells: dict[tuple[str, str, int], list[dict]], examples: dict[str, dict]
) -> list[dict]:
    rows = []
    for model in MODELS:
        for method in METHODS:
            for row_id, example in sorted(examples.items()):
                if example["partition"] not in {"retain", "utility"}:
                    continue
                deltas = []
                losses = []
                for seed in SEEDS:
                    current = by_id(cells[(model, method, seed)])[row_id]
                    baseline = by_id(cells[(model, "trained_full", seed)])[row_id]
                    losses.append(float(current["loss"]))
                    deltas.append(float(current["loss"]) - float(baseline["loss"]))
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "partition": example["partition"],
                        "id": row_id,
                        "group_id": example["group_id"],
                        "prompt": example["prompt"],
                        "target": example["completion"],
                        "loss_mean": mean(losses),
                        "loss_delta_vs_full_mean": mean(deltas),
                        "loss_delta_vs_full_sd": stdev(deltas),
                    }
                )
    return rows


def seed_instability(
    cells: dict[tuple[str, str, int], list[dict]]
) -> list[dict]:
    rows = []
    for model in MODELS:
        for method in ALL_METHODS:
            forget = []
            utility = []
            retain = []
            for seed in SEEDS:
                predictions = cells[(model, method, seed)]
                forget.append(
                    mean(
                        float(row["oracle_kl"])
                        for row in predictions
                        if row["partition"] == "forget"
                    )
                )
                utility.append(
                    mean(float(row["loss"]) for row in predictions if row["partition"] == "utility")
                )
                retain.append(
                    mean(float(row["loss"]) for row in predictions if row["partition"] == "retain")
                )
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "forget_oracle_kl_mean": mean(forget),
                    "forget_oracle_kl_sd": stdev(forget),
                    "utility_nll_mean": mean(utility),
                    "utility_nll_sd": stdev(utility),
                    "retain_nll_mean": mean(retain),
                    "retain_nll_sd": stdev(retain),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    shown_methods = ("trained_full", "exact_retrain", *METHODS)
    groups = sorted({row["group_id"] for row in rows})
    maximum = max(row["oracle_kl_mean"] for row in rows)
    image = None
    for axis, model in zip(axes, MODELS, strict=True):
        values = np.asarray(
            [
                [
                    next(
                        row["oracle_kl_mean"]
                        for row in rows
                        if row["model"] == model
                        and row["method"] == method
                        and row["group_id"] == group
                    )
                    for group in groups
                ]
                for method in shown_methods
            ]
        )
        image = axis.imshow(values, aspect="auto", cmap="magma", vmin=0.0, vmax=maximum)
        axis.set_title(model)
        axis.set_xticks(range(len(groups)), [item.removeprefix("association-") for item in groups])
        axis.set_yticks(
            range(len(shown_methods)), [method.replace("_", " ") for method in shown_methods]
        )
        axis.set_xlabel("Held-out forget association")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Residual KL to exact-retrain oracle")
    figure.suptitle("Post-confirmatory error analysis: residual forgetting by group")
    output = ROOT / "reports/figures/error_forget_group_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(forget: list[dict], damage: list[dict], instability: list[dict]) -> None:
    lines = [
        "# Post-confirmatory error analysis",
        "",
        "**EXPLORATORY.** Ranking rules below were applied to every eligible example; examples "
        "were not selected by narrative convenience.",
        "",
        "## Hardest forget associations",
        "",
        "A hard association has the largest residual KL distance to the exact-retrain oracle, "
        "averaged over all six approximate methods, three surface forms, and three seeds.",
        "",
        "| Model | Rank | Association | Residual KL |",
        "| --- | ---: | --- | ---: |",
    ]
    for model in MODELS:
        groups = sorted({row["group_id"] for row in forget})
        ranked = []
        for group in groups:
            values = [
                row["oracle_kl_mean"]
                for row in forget
                if row["model"] == model
                and row["method"] in METHODS
                and row["group_id"] == group
            ]
            ranked.append((mean(values), group))
        for rank, (value, group) in enumerate(sorted(ranked, reverse=True), start=1):
            lines.append(f"| {model} | {rank} | {group} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Largest observed retain damage",
            "",
            "Rows are the five largest positive paired NLL changes from full-trained, averaged "
            "over seeds. Negative values indicate improvement rather than damage.",
            "",
            "| Model | Method | Example | Target | ΔNLL |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    retain = [row for row in damage if row["partition"] == "retain"]
    for row in sorted(
        retain, key=lambda item: (-item["loss_delta_vs_full_mean"], item["id"])
    )[:5]:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['id']} | {row['target']} | "
            f"{row['loss_delta_vs_full_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed instability",
            "",
            "The table reports the three largest seed SDs for each primary endpoint. With only "
            "three seeds, SD is descriptive and tail behavior is not well characterized.",
            "",
            "| Endpoint | Model | Method | Seed SD |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for endpoint in ("forget_oracle_kl", "utility_nll"):
        ranked = sorted(
            instability,
            key=lambda row: (-row[f"{endpoint}_sd"], row["model"], row["method"]),
        )[:3]
        for row in ranked:
            lines.append(
                f"| {endpoint} | {row['model']} | {row['method']} | "
                f"{row[f'{endpoint}_sd']:.4f} |"
            )
    exact_max = max(
        row["oracle_kl_mean"] for row in forget if row["method"] == "exact_retrain"
    )
    lines.extend(
        [
            "",
            "## Failure modes and interpretation",
            "",
            "- Counterfactual fine-tuning produced the largest Pythia movement toward the oracle "
            "but damaged held-out utility; this is the central utility-collapse warning.",
            "- Several methods changed retain NLL unevenly across associations, so a favorable "
            "mean can conceal localized damage; all paired rows are retained in the CSV.",
            "- Mamba effects were smaller under this frozen protocol. This is observational and "
            "does not identify architecture as the cause.",
            f"- Exact retraining's maximum forget-set oracle KL was {exact_max:.3g}, as expected "
            "when the reference is compared with itself.",
            "",
            "Full group, example, and seed results are in `reports/tables/error_*.csv`. The "
            "synthetic task has only four independent forget associations and cannot establish "
            "behavior on natural memorization or legal deletion requests.",
            "",
        ]
    )
    (ROOT / "reports/ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cells, examples = load_cells()
    forget = forget_groups(cells, examples)
    damage = damage_rows(cells, examples)
    instability = seed_instability(cells)
    write_csv(ROOT / "reports/tables/error_forget_groups.csv", forget)
    write_csv(ROOT / "reports/tables/error_retain_utility_examples.csv", damage)
    write_csv(ROOT / "reports/tables/error_seed_instability.csv", instability)
    write_figure(forget)
    write_report(forget, damage, instability)
    print("Generated exploratory error analysis from all confirmatory Track A predictions")


if __name__ == "__main__":
    main()
