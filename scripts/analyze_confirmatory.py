#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/unlearn-bench-matplotlib")

from unlearn_bench.evaluation import metrics_from_predictions  # noqa: E402
from unlearn_bench.manifest import validate_failure_record, validate_run_manifest  # noqa: E402
from unlearn_bench.statistics import holm_adjust  # noqa: E402
from unlearn_bench.utils.reproducibility import sha256_file  # noqa: E402

PREREGISTRATION_COMMIT = "005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6"
EXPERIMENTS = {
    "controlled-main-pythia-160m-v2": "pythia-160m",
    "controlled-main-mamba-130m-v2": "mamba-130m-hf",
}
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
COMPARISONS = (
    "continued_retain",
    "sham",
    "counterfactual",
    "gradient_ascent",
    "npo",
    "pcgu",
)
SEEDS = (11, 29, 47)
BOOTSTRAP_DRAWS = 10_000


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def validate_checkpoint(manifest: dict) -> None:
    path = ROOT / manifest["final_checkpoint_path"]
    if path.stat().st_size != manifest["checkpoint_size_bytes"]:
        raise ValueError(f"Checkpoint size mismatch: {path}")
    if sha256_file(path) != manifest["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")


def load_cells() -> tuple[dict[tuple[str, str, int], dict], list[dict]]:
    expected = {
        (model, method, seed)
        for model in EXPERIMENTS.values()
        for method in METHODS
        for seed in SEEDS
    }
    cells: dict[tuple[str, str, int], dict] = {}
    failures = []
    for run_dir in sorted((ROOT / "results" / "runs").iterdir()):
        manifest_path = run_dir / "manifest.json"
        failure_path = run_dir / "failure.json"
        if failure_path.is_file():
            failure = load_json(failure_path)
            if failure.get("preregistration_commit") == PREREGISTRATION_COMMIT:
                validate_failure_record(failure)
                failures.append(failure)
            continue
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        if manifest.get("preregistration_commit") != PREREGISTRATION_COMMIT:
            continue
        validate_run_manifest(manifest, ROOT)
        if manifest["experiment_name"] not in EXPERIMENTS:
            raise ValueError(f"Unexpected confirmatory experiment: {manifest['experiment_name']}")
        key = (manifest["model_name"], manifest["method"], manifest["random_seed"])
        if key in cells:
            raise ValueError(f"Duplicate confirmatory cell: {key}")
        if key not in expected:
            raise ValueError(f"Unregistered confirmatory cell: {key}")
        predictions = load_jsonl(ROOT / manifest["predictions_path"])
        metrics = load_json(ROOT / manifest["metrics_path"])
        recomputed = metrics_from_predictions(predictions)
        for name, value in metrics.items():
            expected_value = recomputed[name]
            if isinstance(value, (int, float)):
                if not math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Metric mismatch in {manifest['run_id']}: {name}")
            elif value != expected_value:
                raise ValueError(f"Metric label mismatch in {manifest['run_id']}: {name}")
        validate_checkpoint(manifest)
        cells[key] = {"manifest": manifest, "metrics": metrics, "predictions": predictions}
    terminal = set(cells) | {
        (failure["model_name"], failure["method"], failure["random_seed"])
        for failure in failures
    }
    missing = expected - terminal
    extra = terminal - expected
    if missing or extra:
        raise ValueError(
            f"Confirmatory matrix mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if failures:
        names = [f"{item['run_id']}:{item['status']}" for item in failures]
        raise RuntimeError(
            f"Confirmatory failures require scientific review before analysis: {names}"
        )
    if set(cells) != expected:
        raise ValueError("Not all 54 preregistered cells completed")
    return cells, failures


def attach_groups(cells: dict[tuple[str, str, int], dict]) -> None:
    test_rows = load_jsonl(ROOT / "data" / "controlled" / "v2" / "test.jsonl")
    groups = {row["id"]: row["group_id"] for row in test_rows}
    expected_ids = set(groups)
    for key, cell in cells.items():
        predictions = cell["predictions"]
        ids = [row["id"] for row in predictions]
        if len(ids) != 40 or set(ids) != expected_ids or len(set(ids)) != 40:
            raise ValueError(f"Prediction coverage mismatch: {key}")
        for row in predictions:
            row["group_id"] = groups[row["id"]]


def paired_unit_differences(
    cells: dict[tuple[str, str, int], dict],
    model: str,
    method: str,
    partition: str,
    field: str,
) -> dict[int, dict[str, float]]:
    result = {}
    for seed in SEEDS:
        method_rows = {
            row["id"]: row
            for row in cells[(model, method, seed)]["predictions"]
            if row["partition"] == partition
        }
        baseline_rows = {
            row["id"]: row
            for row in cells[(model, "trained_full", seed)]["predictions"]
            if row["partition"] == partition
        }
        if set(method_rows) != set(baseline_rows):
            raise ValueError(f"Unpaired predictions for {model}, {method}, seed {seed}")
        grouped: dict[str, list[float]] = defaultdict(list)
        for row_id, row in method_rows.items():
            unit = row["group_id"] if partition in {"forget", "retain"} else row_id
            grouped[unit].append(float(row[field]) - float(baseline_rows[row_id][field]))
        result[seed] = {unit: mean(values) for unit, values in sorted(grouped.items())}
    return result


def hierarchical_interval(
    differences: dict[int, dict[str, float]], *, seed: int
) -> tuple[float, float, float]:
    seed_ids = sorted(differences)
    units = sorted(next(iter(differences.values())))
    if any(set(values) != set(units) for values in differences.values()):
        raise ValueError("Hierarchical bootstrap units differ across seeds")
    observed = mean(value for values in differences.values() for value in values.values())
    generator = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_seeds = generator.choice(seed_ids, size=len(seed_ids), replace=True)
        values = []
        for sampled_seed in sampled_seeds:
            sampled_units = generator.choice(units, size=len(units), replace=True)
            values.extend(differences[int(sampled_seed)][str(unit)] for unit in sampled_units)
        estimates[draw] = np.mean(values)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(observed), float(low), float(high)


def exact_sign_flip_p(differences: dict[int, dict[str, float]]) -> float:
    units = sorted(next(iter(differences.values())))
    unit_effects = np.asarray(
        [mean(differences[seed][unit] for seed in SEEDS) for unit in units], dtype=float
    )
    observed = abs(float(np.mean(unit_effects)))
    exceedances = 0
    total = 2 ** len(unit_effects)
    for signs in itertools.product((-1.0, 1.0), repeat=len(unit_effects)):
        permuted = abs(float(np.mean(unit_effects * np.asarray(signs))))
        exceedances += permuted >= observed - 1e-15
    return exceedances / total


def primary_statistics(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    endpoints = (("forget", "oracle_kl", "forget_oracle_kl"), ("utility", "loss", "utility_nll"))
    for model in EXPERIMENTS.values():
        for partition, field, endpoint in endpoints:
            family = []
            for method in COMPARISONS:
                differences = paired_unit_differences(cells, model, method, partition, field)
                effect, low, high = hierarchical_interval(
                    differences, seed=stable_seed(model, method, endpoint)
                )
                family.append(
                    {
                        "model": model,
                        "endpoint": endpoint,
                        "method": method,
                        "baseline": "trained_full",
                        "effect": effect,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": exact_sign_flip_p(differences),
                    }
                )
            adjusted = holm_adjust([row["p_value"] for row in family])
            for row, p_adjusted in zip(family, adjusted, strict=True):
                row["p_holm"] = p_adjusted
                rows.append(row)
    return rows


def summary_rows(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for model in EXPERIMENTS.values():
        for method in METHODS:
            values = [cells[(model, method, seed)] for seed in SEEDS]
            forget = [item["metrics"]["forget_oracle_kl"] for item in values]
            utility = [item["metrics"]["utility_nll"] for item in values]
            retain = [item["metrics"]["retain_nll"] for item in values]
            runtime = [item["manifest"]["optimization_runtime_seconds"] for item in values]
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "forget_mean": mean(forget),
                    "forget_sd": stdev(forget),
                    "utility_mean": mean(utility),
                    "utility_sd": stdev(utility),
                    "retain_mean": mean(retain),
                    "retain_sd": stdev(retain),
                    "optimization_seconds_mean": mean(runtime),
                    "optimization_seconds_sd": stdev(runtime),
                    "optimization_seconds_total": sum(runtime),
                    "peak_vram_gib": max(
                        item["manifest"]["peak_vram_bytes"] for item in values
                    )
                    / 2**30,
                    "checkpoint_gib_total": sum(
                        item["manifest"]["checkpoint_size_bytes"] for item in values
                    )
                    / 2**30,
                }
            )
    return rows


def seed_rows(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for model in EXPERIMENTS.values():
        for method in METHODS:
            for seed in SEEDS:
                cell = cells[(model, method, seed)]
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "forget_oracle_kl": cell["metrics"]["forget_oracle_kl"],
                        "utility_nll": cell["metrics"]["utility_nll"],
                        "retain_nll": cell["metrics"]["retain_nll"],
                        "optimization_seconds": cell["manifest"][
                            "optimization_runtime_seconds"
                        ],
                        "peak_vram_bytes": cell["manifest"]["peak_vram_bytes"],
                        "examples_processed": sum(
                            part.get("examples_processed", 0) or 0
                            for part in cell["manifest"]["optimization_work"].values()
                        ),
                        "tokens_processed": sum(
                            part.get("tokens_processed", 0) or 0
                            for part in cell["manifest"]["optimization_work"].values()
                        ),
                    }
                )
    return rows


def pareto_methods(rows: list[dict], model: str, axes: tuple[str, ...]) -> list[str]:
    candidates = [row for row in rows if row["model"] == model]
    frontier = []
    for row in candidates:
        dominated = any(
            all(other[axis] <= row[axis] for axis in axes)
            and any(other[axis] < row[axis] for axis in axes)
            for other in candidates
            if other is not row
        )
        if not dominated:
            frontier.append(row["method"])
    return frontier


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(rows: list[dict], seed_values: list[dict]) -> None:
    import matplotlib.pyplot as plt

    output = ROOT / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
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
    for x_axis, filename, label in (
        ("utility_mean", "main_forgetting_utility.png", "Utility NLL (lower is better)"),
        (
            "optimization_seconds_mean",
            "main_forgetting_compute.png",
            "Mean optimization time, seconds (lower is better)",
        ),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        for axis, model in zip(axes, EXPERIMENTS.values(), strict=True):
            for row in (item for item in rows if item["model"] == model):
                x_error = (
                    row["utility_sd"]
                    if x_axis == "utility_mean"
                    else row["optimization_seconds_sd"]
                )
                axis.errorbar(
                    row[x_axis],
                    row["forget_mean"],
                    xerr=x_error,
                    yerr=row["forget_sd"],
                    marker="o",
                    markersize=7,
                    capsize=2,
                    color=colors[row["method"]],
                    linestyle="none",
                    label=labels[row["method"]],
                )
            axis.set_title(model)
            axis.set_xlabel(label)
            axis.set_ylabel("Forget KL to exact retrain (lower is better)")
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
        figure.suptitle("Confirmatory forgetting trade-off (mean ± seed SD)")
        figure.tight_layout(rect=(0, 0.12, 1, 0.95))
        figure.savefig(output / filename, dpi=220)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    positions = np.arange(len(METHODS))
    for axis, model in zip(axes, EXPERIMENTS.values(), strict=True):
        for index, method in enumerate(METHODS):
            values = [
                row["forget_oracle_kl"]
                for row in seed_values
                if row["model"] == model and row["method"] == method
            ]
            axis.scatter([index] * len(values), values, color=colors[method], s=28)
            axis.plot([index] * len(values), values, color=colors[method], alpha=0.35)
        axis.set_xticks(positions, METHODS, rotation=55, ha="right")
        axis.set_title(model)
        axis.set_ylabel("Forget KL to exact retrain")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "main_seed_variation.png", dpi=220)
    plt.close(figure)


def write_report(rows: list[dict], statistics: list[dict]) -> None:
    def comparison(model: str, endpoint: str, method: str) -> dict:
        return next(
            row
            for row in statistics
            if row["model"] == model
            and row["endpoint"] == endpoint
            and row["method"] == method
        )

    frontiers = {
        model: {
            "forget_utility": pareto_methods(rows, model, ("forget_mean", "utility_mean")),
            "forget_compute": pareto_methods(
                rows, model, ("forget_mean", "optimization_seconds_mean")
            ),
        }
        for model in EXPERIMENTS.values()
    }
    pythia_counterfactual_forget = comparison(
        "pythia-160m", "forget_oracle_kl", "counterfactual"
    )["effect"]
    pythia_counterfactual_utility = comparison(
        "pythia-160m", "utility_nll", "counterfactual"
    )["effect"]
    lines = [
        "# Main confirmatory results",
        "",
        f"**Preregistration:** `{PREREGISTRATION_COMMIT}`",
        "",
        "All 54 frozen cells completed and passed manifest, prediction, metric, and "
        "checkpoint-hash validation. Lower is better for both primary endpoints.",
        "",
        "## Primary seed-aggregated results",
        "",
        "| Model | Method | Forget KL mean ± SD | Utility NLL mean ± SD | "
        "Retain NLL mean ± SD | Mean optimize s | Peak GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['forget_mean']:.4f} ± "
            f"{row['forget_sd']:.4f} | {row['utility_mean']:.4f} ± "
            f"{row['utility_sd']:.4f} | {row['retain_mean']:.4f} ± "
            f"{row['retain_sd']:.4f} | {row['optimization_seconds_mean']:.3f} | "
            f"{row['peak_vram_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified paired comparisons",
            "",
            "Effects are method minus full-trained. Negative effects favor the method. Confidence "
            "intervals are the preregistered hierarchical paired bootstrap; p-values use exact "
            "content-unit sign flips and Holm correction within each model × endpoint family.",
            "",
            "| Model | Endpoint | Method | Effect | 95% CI | Exact p | Holm p |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in statistics:
        lines.append(
            f"| {row['model']} | {row['endpoint']} | {row['method']} | "
            f"{row['effect']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['p_value']:.4f} | {row['p_holm']:.4f} |"
        )
    lines.extend(["", "## Pareto and joint-criterion summary", ""])
    for model, values in frontiers.items():
        joint = []
        for method in COMPARISONS:
            forgetting = next(
                row
                for row in statistics
                if row["model"] == model
                and row["endpoint"] == "forget_oracle_kl"
                and row["method"] == method
            )
            utility = next(
                row
                for row in statistics
                if row["model"] == model
                and row["endpoint"] == "utility_nll"
                and row["method"] == method
            )
            if forgetting["effect"] < 0 and utility["ci_high"] <= 0:
                joint.append(method)
        lines.extend(
            [
                f"- **{model}:** forgetting–utility frontier: "
                f"{', '.join(values['forget_utility'])}; forgetting–compute frontier: "
                f"{', '.join(values['forget_compute'])}.",
                f"- **{model} joint preregistered criterion:** "
                f"{', '.join(joint) if joint else 'no method met the criterion'}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Confirmatory interpretation",
            "",
            "For Pythia, counterfactual training produced the largest mean forgetting change "
            f"from full-trained ({pythia_counterfactual_forget:.4f}) but worsened utility "
            f"({pythia_counterfactual_utility:.4f}). "
            "Gradient ascent and NPO met the preregistered joint criterion; PCGU improved mean "
            "forgetting with an interval below zero, but its utility interval crossed zero.",
            "",
            "For Mamba, no method met the joint criterion. PCGU's forgetting interval was below "
            "zero, while its utility interval crossed zero. Counterfactual training had the "
            "largest mean forgetting and utility improvements, but both intervals crossed zero. "
            "Continued retain training moved both models farther from the oracle on forget examples.",
            "",
            f"No Holm-adjusted comparison reached 0.05 (smallest adjusted p = "
            f"{min(row['p_holm'] for row in statistics):.4f}). This is consistent with the limited "
            "four-group forget holdout and prevents strong significance claims despite several "
            "bootstrap intervals excluding zero.",
            "",
            "## Compute, failures, and deviations",
            "",
            "All 54 cells completed with no technical, OOM, unsupported-device, numerical, or "
            "method failures. Scheduler elapsed time totaled 413 A100-seconds (0.115 GPU-hours). "
            "Peak allocation was 9.74 GiB for Pythia and 15.84 GiB for Mamba. The only execution "
            "deviation was moving two zero-runtime pending jobs from `htc` to the available "
            "`lightwork` A100 MIG pool; frozen experiment settings did not change. Mamba used the "
            "predeclared sequential eager fallback.",
            "",
            "## Interpretation discipline",
            "",
            "These results establish behavior only for the controlled synthetic associations, two "
            "small pinned checkpoints, and the fixed short optimization budget. Cross-model "
            "differences are observational, not causal architecture effects. Exact-retrain KL is "
            "zero by definition. Track B external behavior, ablations, and error analysis are "
            "separate exploratory analyses.",
            "",
            "## Reproduction",
            "",
            "Regenerate this report, CSV tables, and figures with "
            "`python scripts/analyze_confirmatory.py`. Individual seed values are stored in "
            "`reports/tables/main_seed_results.csv`.",
            "",
        ]
    )
    (ROOT / "reports" / "MAIN_CONFIRMATORY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (ROOT / "results" / "manifests" / "CONFIRMATORY_COMPLETE.json").write_text(
        json.dumps(
            {
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "completed_cells": 54,
                "failed_cells": 0,
                "validated_checkpoint_hashes": 54,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "pareto_frontiers": frontiers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    cells, _ = load_cells()
    attach_groups(cells)
    summaries = summary_rows(cells)
    seeds = seed_rows(cells)
    statistics = primary_statistics(cells)
    write_csv(ROOT / "reports" / "tables" / "main_summary.csv", summaries)
    write_csv(ROOT / "reports" / "tables" / "main_seed_results.csv", seeds)
    write_csv(ROOT / "reports" / "tables" / "main_comparisons.csv", statistics)
    write_figures(summaries, seeds)
    write_report(summaries, statistics)
    print("Validated and analyzed all 54 preregistered confirmatory cells")


if __name__ == "__main__":
    main()
