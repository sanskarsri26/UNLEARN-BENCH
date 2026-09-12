#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlearn_bench.evaluation import stereoset_metrics  # noqa: E402
from unlearn_bench.models import causal_batch, load_causal_lm  # noqa: E402
from unlearn_bench.models.causal import token_log_probabilities  # noqa: E402
from unlearn_bench.utils.device import (  # noqa: E402
    memory_metrics,
    reset_peak_memory,
    select_device,
)
from unlearn_bench.utils.reproducibility import (  # noqa: E402
    atomic_json,
    git_commit,
    set_seed,
    sha256_file,
    sha256_value,
    write_jsonl,
)

DATASET = ROOT / "Unlearning" / "data" / "stereoset_dev.json"
DATASET_SHA256 = "73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90"
SOURCE_COMMIT = "ead7d086a64a192a1eca88e0dd2fd163de375218"
PREREGISTRATION_COMMIT = "005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6"
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


def inserted_candidate(context: str, sentence: str) -> tuple[str, str]:
    if "BLANK" not in context:
        raise ValueError(f"StereoSet context has no BLANK: {context!r}")
    parts = context.split("BLANK")
    before = parts[0]
    anchor = parts[1]
    folded = sentence.casefold()
    if not folded.startswith(before.casefold()):
        raise ValueError("Candidate sentence does not match its BLANK context")
    if anchor:
        end = folded.find(anchor.casefold(), len(before))
        if end < 0:
            raise ValueError("Candidate sentence does not contain the post-BLANK context")
    else:
        end = len(sentence)
    candidate = sentence[len(before) : end].strip()
    if not candidate:
        raise ValueError("StereoSet candidate span is empty")
    if context.replace("BLANK", candidate).casefold() != folded:
        raise ValueError("Candidate does not reconstruct its BLANK context")
    return before.rstrip(), candidate


def prepare_items(limit: int | None = None) -> list[dict]:
    if sha256_file(DATASET) != DATASET_SHA256:
        raise ValueError("StereoSet artifact hash does not match the pinned official dev set")
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    source = payload["data"]["intrasentence"]
    if limit is not None:
        source = source[:limit]
    items = []
    for item in source:
        candidates = {}
        for sentence in item["sentences"]:
            label = sentence["gold_label"].replace("anti-stereotype", "anti_stereotype")
            if label in candidates:
                raise ValueError(f"Duplicate StereoSet label in {item['id']}")
            prompt, completion = inserted_candidate(item["context"], sentence["sentence"])
            candidates[label] = {
                "sentence_id": sentence["id"],
                "prompt": prompt,
                "completion": completion,
            }
        required = {"stereotype", "anti_stereotype", "unrelated"}
        if set(candidates) != required:
            raise ValueError(f"Malformed StereoSet labels in {item['id']}")
        items.append(
            {
                "id": item["id"],
                "category": item["bias_type"],
                "target": item["target"],
                "candidates": candidates,
            }
        )
    return items


def score_items(model, tokenizer, items: list[dict], batch_size: int) -> list[dict]:
    flat = []
    for item in items:
        for label, candidate in item["candidates"].items():
            flat.append({"item": item, "label": label, **candidate})
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            rows = flat[start : start + batch_size]
            batch = causal_batch(tokenizer, rows, next(model.parameters()).device)
            token_logp, active = token_log_probabilities(model, batch)
            values = (token_logp * active).sum(dim=1) / active.sum(dim=1)
            scores.extend(float(value) for value in values)
    output = {}
    for row, score in zip(flat, scores, strict=True):
        item_id = row["item"]["id"]
        if item_id not in output:
            output[item_id] = {
                "id": item_id,
                "category": row["item"]["category"],
                "target": row["item"]["target"],
            }
        output[item_id][row["label"]] = score
    return [output[item["id"]] for item in items]


def confirmatory_cells(
    model_name: str, methods: set[str], seeds: set[int]
) -> list[tuple[Path, dict]]:
    cells = []
    for path in sorted((ROOT / "results" / "runs").glob("controlled-main-*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("preregistration_commit") == PREREGISTRATION_COMMIT
            and manifest["model_name"] == model_name
            and manifest["method"] in methods
            and manifest["random_seed"] in seeds
        ):
            cells.append((path, manifest))
    expected = len(methods) * len(seeds)
    if len(cells) != expected:
        raise ValueError(f"Expected {expected} Track A source checkpoints, found {len(cells)}")
    return cells


def evaluate(
    model_name: str,
    methods: set[str],
    output_root: Path,
    *,
    seeds: set[int],
    limit: int | None,
    batch_size: int,
) -> None:
    items = prepare_items(limit)
    cells = confirmatory_cells(model_name, methods, seeds)
    context = select_device("cuda", "fp32", allow_mps_fallback=False)
    model_config = cells[0][1]["configuration"]["experiment"]["model_config"]
    set_seed(0, deterministic=True)
    model, tokenizer = load_causal_lm(model_config, context)
    for manifest_path, source in cells:
        run_id = f"track-b-stereoset-{source['run_id']}"
        destination = output_root / run_id
        if (destination / "manifest.json").is_file():
            print(f"existing: {run_id}")
            continue
        checkpoint = ROOT / source["final_checkpoint_path"]
        if sha256_file(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError(f"Source checkpoint hash mismatch: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        del payload
        gc.collect()
        reset_peak_memory(context)
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = score_items(model, tokenizer, items, batch_size)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        metrics = stereoset_metrics(predictions)
        if not all(
            math.isfinite(value)
            for group in [metrics["overall"], *metrics["by_category"].values()]
            for value in group.values()
        ):
            raise FloatingPointError(f"Non-finite Track B metric for {run_id}")
        destination.mkdir(parents=True, exist_ok=False)
        predictions_path = destination / "predictions.jsonl"
        metrics_path = destination / "metrics.json"
        write_jsonl(predictions_path, predictions)
        atomic_json(metrics_path, metrics)
        memory = memory_metrics(context)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(ROOT),
            "track": "external_bias_generalization",
            "claim_status": "exploratory",
            "source_track_a_run_id": source["run_id"],
            "source_track_a_checkpoint_sha256": source["checkpoint_sha256"],
            "source_track_a_preregistration_commit": source["preregistration_commit"],
            "model_name": model_name,
            "model_revision": source["model_revision"],
            "method": source["method"],
            "random_seed": source["random_seed"],
            "device": context.device,
            "dtype": context.precision,
            "hardware": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
                "device_memory_bytes": torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory,
            },
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "dataset": "StereoSet dev intrasentence",
            "dataset_sha256": DATASET_SHA256,
            "dataset_source_commit": SOURCE_COMMIT,
            "dataset_license": "CC-BY-SA-4.0",
            "num_examples": len(items),
            "batch_size": batch_size,
            "score_definition": "mean continuation-token conditional log probability",
            "runtime_seconds": runtime,
            "peak_vram_bytes": memory["peak_device_memory_bytes"],
            "predictions_path": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256_file(metrics_path),
            "configuration_sha256": sha256_value(
                {
                    "limit": limit,
                    "batch_size": batch_size,
                    "methods": sorted(methods),
                    "seeds": sorted(seeds),
                }
            ),
        }
        atomic_json(destination / "manifest.json", manifest)
        print(f"completed: {run_id} ({runtime:.3f}s, {len(items)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track A checkpoints on StereoSet")
    parser.add_argument("--model", choices=("pythia-160m", "mamba-130m-hf"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(11, 29, 47), default=[11, 29, 47])
    parser.add_argument("--output", default="results/track_b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    evaluate(
        args.model,
        set(args.methods),
        ROOT / args.output,
        seeds=set(args.seeds),
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
