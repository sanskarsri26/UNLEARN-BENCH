#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/unlearn-bench-matplotlib")

from unlearn_bench.evaluation import stereoset_metrics  # noqa: E402
from unlearn_bench.utils.reproducibility import sha256_file  # noqa: E402

DATASET_SHA256 = "73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90"
MODELS = ("pythia-160m", "mamba-130m-hf")
METHODS = (
    "untouched",
    "trained_full",
    "exact_retrain",
    "continued_retain",
    "sham",
    "counterfactual",
    "gradient_ascent",
    "npo",
    "pcgu",
)
SEEDS = (11, 29, 47)
CATEGORIES = ("gender", "profession", "race", "religion")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_and_load() -> dict[tuple[str, str, int], dict]:
    cells = {}
    for path in sorted((ROOT / "results" / "track_b").glob("*/manifest.json")):
        manifest = load_json(path)
        if manifest["track"] != "external_bias_generalization":
            raise ValueError(f"Unexpected Track B label: {path}")
        if manifest["claim_status"] != "exploratory":
            raise ValueError(f"Track B cell is not exploratory: {path}")
        if manifest["dataset_sha256"] != DATASET_SHA256 or manifest["num_examples"] != 2106:
            raise ValueError(f"Track B dataset mismatch: {path}")
        predictions_path = ROOT / manifest["predictions_path"]
        metrics_path = ROOT / manifest["metrics_path"]
        if sha256_file(predictions_path) != manifest["predictions_sha256"]:
            raise ValueError(f"Track B prediction hash mismatch: {path}")
        if sha256_file(metrics_path) != manifest["metrics_sha256"]:
            raise ValueError(f"Track B metric hash mismatch: {path}")
        predictions = load_jsonl(predictions_path)
        if len(predictions) != 2106 or len({row["id"] for row in predictions}) != 2106:
            raise ValueError(f"Track B prediction coverage mismatch: {path}")
        metrics = load_json(metrics_path)
        if metrics != stereoset_metrics(predictions):
            raise ValueError(f"Track B metric recomputation mismatch: {path}")
        key = (manifest["model_name"], manifest["method"], manifest["random_seed"])
        if key in cells:
            raise ValueError(f"Duplicate Track B cell: {key}")
        cells[key] = {"manifest": manifest, "metrics": metrics}
    expected = {
        (model, method, seed) for model in MODELS for method in METHODS for seed in SEEDS
    }
    if set(cells) != expected:
        raise ValueError(f"Track B matrix mismatch; missing={sorted(expected - set(cells))}")
    return cells


def aggregate(cells: dict[tuple[str, str, int], dict]) -> tuple[list[dict], list[dict]]:
    overall = []
    categories = []
    for model in MODELS:
        for method in METHODS:
            groups = [cells[(model, method, seed)]["metrics"] for seed in SEEDS]
            values = {
                name: [group["overall"][name] for group in groups]
                for name in ("lms", "ss", "icat")
            }
            overall.append(
                {
                    "model": model,
                    "method": method,
                    "lms_mean": mean(values["lms"]),
                    "lms_sd": stdev(values["lms"]),
                    "ss_mean": mean(values["ss"]),
                    "ss_sd": stdev(values["ss"]),
                    "ss_distance_50_mean": mean(abs(value - 50.0) for value in values["ss"]),
                    "icat_mean": mean(values["icat"]),
                    "icat_sd": stdev(values["icat"]),
                    "runtime_seconds_total": sum(
                        cells[(model, method, seed)]["manifest"]["runtime_seconds"]
                        for seed in SEEDS
                    ),
                    "peak_vram_gib": max(
                        cells[(model, method, seed)]["manifest"]["peak_vram_bytes"]
                        for seed in SEEDS
                    )
                    / 2**30,
                }
            )
            for category in CATEGORIES:
                category_values = {
                    name: [group["by_category"][category][name] for group in groups]
                    for name in ("lms", "ss", "icat")
                }
                categories.append(
                    {
                        "model": model,
                        "method": method,
                        "category": category,
                        "count": groups[0]["by_category"][category]["count"],
                        "lms_mean": mean(category_values["lms"]),
                        "lms_sd": stdev(category_values["lms"]),
                        "ss_mean": mean(category_values["ss"]),
                        "ss_sd": stdev(category_values["ss"]),
                        "icat_mean": mean(category_values["icat"]),
                        "icat_sd": stdev(category_values["icat"]),
                    }
                )
    return overall, categories


def paired_deltas(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        for method in METHODS:
            if method == "trained_full":
                continue
            seed_rows = []
            for seed in SEEDS:
                current = cells[(model, method, seed)]["metrics"]["overall"]
                baseline = cells[(model, "trained_full", seed)]["metrics"]["overall"]
                seed_rows.append(
                    {
                        "lms": current["lms"] - baseline["lms"],
                        "ss": current["ss"] - baseline["ss"],
                        "neutrality": abs(current["ss"] - 50.0) - abs(baseline["ss"] - 50.0),
                        "icat": current["icat"] - baseline["icat"],
                    }
                )
            row = {"model": model, "method": method, "baseline": "trained_full"}
            for metric in ("lms", "ss", "neutrality", "icat"):
                values = [item[metric] for item in seed_rows]
                row[f"{metric}_delta_mean"] = mean(values)
                row[f"{metric}_delta_sd"] = stdev(values)
            rows.append(row)
    return rows


def paired_category_deltas(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        for method in METHODS:
            if method == "trained_full":
                continue
            for category in CATEGORIES:
                seed_rows = []
                for seed in SEEDS:
                    current = cells[(model, method, seed)]["metrics"]["by_category"][category]
                    baseline = cells[(model, "trained_full", seed)]["metrics"]["by_category"][
                        category
                    ]
                    seed_rows.append(
                        {
                            "lms": current["lms"] - baseline["lms"],
                            "ss": current["ss"] - baseline["ss"],
                            "neutrality": abs(current["ss"] - 50.0)
                            - abs(baseline["ss"] - 50.0),
                            "icat": current["icat"] - baseline["icat"],
                        }
                    )
                row = {
                    "model": model,
                    "method": method,
                    "category": category,
                    "baseline": "trained_full",
                }
                for metric in ("lms", "ss", "neutrality", "icat"):
                    values = [item[metric] for item in seed_rows]
                    row[f"{metric}_delta_mean"] = mean(values)
                    row[f"{metric}_delta_sd"] = stdev(values)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors = dict(zip(METHODS, plt.cm.tab10.colors, strict=False))
    labels = {
        "untouched": "Untouched",
        "trained_full": "Full trained",
        "exact_retrain": "Exact retrain",
        "continued_retain": "Continued retain",
        "sham": "Sham",
        "counterfactual": "Counterfactual",
        "gradient_ascent": "Gradient ascent",
        "npo": "NPO",
        "pcgu": "PCGU",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for axis, model in zip(axes, MODELS, strict=True):
        for row in (item for item in rows if item["model"] == model):
            axis.errorbar(
                row["ss_mean"],
                row["lms_mean"],
                xerr=row["ss_sd"],
                yerr=row["lms_sd"],
                marker="o",
                capsize=2,
                linestyle="none",
                color=colors[row["method"]],
                label=labels[row["method"]],
            )
        axis.axvline(50.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        axis.set_title(model)
        axis.set_xlabel("StereoSet SS (neutral at 50)")
        axis.set_ylabel("StereoSet LMS (higher is better)")
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle("Exploratory Track B external bias behavior (mean ± seed SD)")
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    output = ROOT / "reports" / "figures" / "track_b_stereoset.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(overall: list[dict], deltas: list[dict], category_deltas: list[dict]) -> None:
    lines = [
        "# Track B: StereoSet external generalization",
        "",
        "**EXPLORATORY. This is external bias behavior, not machine forgetting.**",
        "",
        "All 54 Track A checkpoints were evaluated on all 2,106 official StereoSet dev "
        "intrasentence examples. SS is neutral near 50; lower SS is not inherently better.",
        "",
        "| Model | Method | LMS mean ± SD | SS mean ± SD | |SS−50| | "
        "ICAT mean ± SD | Score s | Peak GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['lms_mean']:.2f} ± "
            f"{row['lms_sd']:.2f} | {row['ss_mean']:.2f} ± {row['ss_sd']:.2f} | "
            f"{row['ss_distance_50_mean']:.2f} | {row['icat_mean']:.2f} ± "
            f"{row['icat_sd']:.2f} | {row['runtime_seconds_total']:.1f} | "
            f"{row['peak_vram_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired changes from full-trained",
            "",
            "Negative Δ|SS−50| moves toward neutrality; positive ΔLMS and ΔICAT are improvements. "
            "Values are means across the three paired seeds, with no confirmatory "
            "hypothesis tests.",
            "",
            "| Model | Method | ΔLMS | ΔSS | Δ|SS−50| | ΔICAT |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['lms_delta_mean']:.3f} | "
            f"{row['ss_delta_mean']:.3f} | {row['neutrality_delta_mean']:.3f} | "
            f"{row['icat_delta_mean']:.3f} |"
        )
    interventions = {
        "continued_retain",
        "sham",
        "counterfactual",
        "gradient_ascent",
        "npo",
        "pcgu",
    }
    lines.extend(
        [
            "",
            "## Category heterogeneity",
            "",
            "The rows below are selected mechanically as the largest movement toward and away "
            "from SS neutrality within each model. Exact retraining and untouched are excluded "
            "from this intervention diagnostic.",
            "",
            "| Model | Direction | Method | Category | Δ|SS−50| | ΔLMS |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for model in MODELS:
        candidates = [
            row
            for row in category_deltas
            if row["model"] == model and row["method"] in interventions
        ]
        for direction, row in (
            ("toward neutrality", min(candidates, key=lambda item: item["neutrality_delta_mean"])),
            (
                "away from neutrality",
                max(candidates, key=lambda item: item["neutrality_delta_mean"]),
            ),
        ):
            lines.append(
                f"| {model} | {direction} | {row['method']} | {row['category']} | "
                f"{row['neutrality_delta_mean']:.3f} | {row['lms_delta_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Category-level values are in `reports/tables/track_b_categories.csv`. "
            "These exploratory "
            "results do not establish deletion, training-data membership, or causal architecture "
            "effects. Track A utility must be consulted alongside any apparent bias movement.",
            "",
        ]
    )
    (ROOT / "reports" / "TRACK_B_STEREOSET.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    cells = validate_and_load()
    overall, categories = aggregate(cells)
    deltas = paired_deltas(cells)
    category_deltas = paired_category_deltas(cells)
    write_csv(ROOT / "reports" / "tables" / "track_b_summary.csv", overall)
    write_csv(ROOT / "reports" / "tables" / "track_b_categories.csv", categories)
    write_csv(ROOT / "reports" / "tables" / "track_b_deltas.csv", deltas)
    write_csv(
        ROOT / "reports" / "tables" / "track_b_category_deltas.csv", category_deltas
    )
    write_figure(overall)
    write_report(overall, deltas, category_deltas)
    print("Validated and analyzed all 54 exploratory Track B cells")


if __name__ == "__main__":
    main()
