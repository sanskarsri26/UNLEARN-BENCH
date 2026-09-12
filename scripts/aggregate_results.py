#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def aggregate(results_dir: Path) -> list[dict]:
    groups = defaultdict(list)
    for manifest_path in sorted(results_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads((manifest_path.parent / "metrics.json").read_text(encoding="utf-8"))
        key = (
            manifest.get("run_set_id", "legacy-run-set"),
            manifest.get("experiment_name", "unknown"),
            manifest["git_commit"],
            manifest["model_name"],
            manifest["method"],
        )
        groups[key].append((manifest, metrics))
    rows = []
    for (run_set, experiment, commit, model, method), values in sorted(groups.items()):
        seeds = [item[0]["random_seed"] for item in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in run set {run_set}, method {method}: {seeds}")
        forget = [item[1]["forget_oracle_kl"] for item in values]
        utility = [item[1]["utility_nll"] for item in values]
        runtime = [item[0]["runtime_seconds"] for item in values]
        rows.append(
            {
                "run_set": run_set,
                "experiment": experiment,
                "git_commit": commit,
                "model": model,
                "method": method,
                "seeds": len(seeds),
                "forget_oracle_kl_mean": mean(forget),
                "forget_oracle_kl_sd": stdev(forget) if len(forget) > 1 else 0.0,
                "utility_nll_mean": mean(utility),
                "utility_nll_sd": stdev(utility) if len(utility) > 1 else 0.0,
                "runtime_seconds_mean": mean(runtime),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run artifacts into reproducible tables")
    parser.add_argument("--results", default="results/runs")
    args = parser.parse_args()
    rows = aggregate(ROOT / args.results)
    output = ROOT / "reports" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "method", "seeds"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} aggregate rows to {output}")


if __name__ == "__main__":
    main()
