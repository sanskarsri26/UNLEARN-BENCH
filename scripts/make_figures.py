#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/unlearn-bench-matplotlib")


def main() -> None:
    import matplotlib.pyplot as plt

    points = []
    for manifest_path in sorted((ROOT / "results" / "runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads((manifest_path.parent / "metrics.json").read_text(encoding="utf-8"))
        points.append((manifest["method"], metrics["forget_oracle_kl"], metrics["utility_nll"]))
    if not points:
        raise SystemExit("No completed runs found under results/runs")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for method, forgetting, utility in points:
        axis.scatter(utility, forgetting, s=45)
        axis.annotate(
            method,
            (utility, forgetting),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Held-out utility NLL (lower is better)")
    axis.set_ylabel("Forget-set KL to exact retrain (lower is better)")
    axis.set_title("Forgetting–utility trade-off (smoke results are exploratory)")
    axis.grid(alpha=0.25)
    output = ROOT / "reports" / "figures" / "forgetting_utility_pareto.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
