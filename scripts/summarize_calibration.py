#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _calibration_rows() -> list[dict]:
    rows = []
    pattern = "controlled-calibration-*-v1-*/manifest.json"
    for manifest_path in sorted((ROOT / "results" / "runs").glob(pattern)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads((manifest_path.parent / "metrics.json").read_text(encoding="utf-8"))
        pcgu = manifest["method_metadata"].get("pcgu", {})
        rows.append(
            {
                "model": manifest["model_name"],
                "method": manifest["method"],
                "device": manifest["device"],
                "dtype": manifest["dtype"],
                "hardware": manifest["hardware"]["gpu"],
                "available_vram": manifest["hardware"]["device_memory_bytes"] / 2**30,
                "load": manifest["model_load_seconds"],
                "optimization": manifest["optimization_runtime_seconds"],
                "evaluation": manifest["evaluation_runtime_seconds"],
                "peak_gpu": manifest["peak_vram_bytes"] / 2**30,
                "peak_process": manifest["peak_process_memory_bytes"] / 2**30,
                "examples_per_second": manifest["examples_per_second"],
                "tokens_per_second": manifest["tokens_per_second"],
                "gradient_memory": pcgu.get("gradient_bytes", 0) / 2**30,
                "ranking_memory": pcgu.get("mask_bytes", 0) / 2**30,
                "forget_kl": metrics["forget_oracle_kl"],
                "utility_nll": metrics["utility_nll"],
                "git_commit": manifest["git_commit"],
            }
        )
    if not rows:
        raise SystemExit("No real-model calibration manifests found")
    return rows


def _value(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    rows = _calibration_rows()
    commits = sorted({row["git_commit"][:7] for row in rows})
    lines = [
        "# Device calibration",
        "",
        "**Status: EXPLORATORY COMPUTE CALIBRATION — NOT A CONFIRMATORY RESULT.**",
        "",
        "This report is regenerated from schema-v2 raw run artifacts by",
        "`python scripts/summarize_calibration.py`. Calibration commits: "
        + ", ".join(commits)
        + ".",
        "",
        "## Measured A100 results",
        "",
        "| Model | Method | Hardware / dtype | Load s | Optimize s | Eval s | Ex/s | Tok/s | "
        "Peak GPU GiB | Peak process GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['hardware']} / {row['dtype']} | "
            f"{row['load']:.2f} | {row['optimization']:.3f} | {row['evaluation']:.3f} | "
            f"{_value(row['examples_per_second'])} | {_value(row['tokens_per_second'])} | "
            f"{row['peak_gpu']:.2f} | {row['peak_process']:.2f} |"
        )
    pcgu_rows = [row for row in rows if row["method"] == "pcgu"]
    lines.extend(
        [
            "",
            "## PCGU memory",
            "",
            "| Model | Two gradient sets GiB | Ranking mask GiB |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in pcgu_rows:
        lines.append(
            f"| {row['model']} | {row['gradient_memory']:.3f} | {row['ranking_memory']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Compatibility and feasibility",
            "",
            "| Model | M3 MPS | M3 CPU | A100 | Recommended device |",
            "| --- | --- | --- | --- | --- |",
            "| Pythia-160M | NOT MEASURED | NOT MEASURED | SUPPORTED (FP32) | "
            "A100 until M3 calibration |",
            "| Mamba-130M | NOT MEASURED | NOT MEASURED | "
            "SUPPORTED WITH SEQUENTIAL EAGER KERNEL FALLBACK (FP32) | A100 |",
            "| Pythia-410M | NOT MEASURED | NOT MEASURED | NOT MEASURED | Undetermined |",
            "| Mamba-370M | NOT MEASURED | NOT MEASURED | NOT MEASURED | Undetermined |",
            "",
            "The Mamba job log explicitly reported that optional fused selective-scan and "
            "causal-convolution kernels were unavailable and that Transformers used its sequential "
            "implementation. This is a kernel implementation fallback on CUDA, not an MPS-to-CPU "
            "device fallback. No architecture was substituted.",
            "",
            "The current session is Linux/x86_64, not the target MacBook Air M3. MPS "
            "compatibility, "
            "CPU fallback behavior, sustained thermal throughput, and M3 memory therefore remain "
            "unmeasured. No local-feasibility classification is inferred from A100 data.",
            "",
            "## Scientific interpretation",
            "",
            "The calibration endpoints were inspected only to detect gross failures. Pythia's "
            "short "
            "calibration fit the controlled associations while catastrophically damaging held-out "
            "utility; Mamba did not show the same gross collapse under this exploratory setup. "
            "These "
            "observations motivate pre-confirmatory optimization calibration but are not method "
            "comparisons, architecture claims, or headline results. No calibration endpoint may be "
            "promoted into the confirmatory report.",
            "",
            "## Remaining measurements",
            "",
            "Checkpoint write time for a real saved checkpoint, sustained M3 throughput, MPS "
            "allocation, CPU fallback profiling, and large-tier scaling remain outstanding. The "
            "confirmatory matrix must not be authorized until those omissions are resolved or "
            "explicitly excluded before preregistration.",
            "",
        ]
    )
    destination = ROOT / "reports" / "device_calibration.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
