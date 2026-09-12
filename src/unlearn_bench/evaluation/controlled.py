from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from unlearn_bench.models.tiny import TinyAssociationLM, Vocabulary


def evaluate_model(
    model: TinyAssociationLM,
    oracle: TinyAssociationLM,
    vocabulary: Vocabulary,
    records: list[dict],
) -> list[dict]:
    model.eval()
    oracle.eval()
    output = []
    with torch.no_grad():
        for row in records:
            prompt = [vocabulary.encode_prompt(row["prompt"])]
            logits = model(prompt)[0]
            oracle_logits = oracle(prompt)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            oracle_probs = F.softmax(oracle_logits, dim=-1)
            target = vocabulary.encode_completion(row["completion"])
            target_logp = float(log_probs[target])
            other = log_probs.clone()
            other[target] = -torch.inf
            predicted = int(log_probs.argmax())
            oracle_kl = float(
                torch.sum(
                    oracle_probs
                    * (torch.log(oracle_probs.clamp_min(1e-12)) - torch.log(probs.clamp_min(1e-12)))
                )
            )
            output.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "partition": row["partition"],
                    "category": row["category"],
                    "target": row["completion"],
                    "prediction": vocabulary.tokens[predicted],
                    "correct": predicted == target,
                    "loss": -target_logp,
                    "target_log_probability": target_logp,
                    "target_probability": float(probs[target]),
                    "conditional_log_probability_margin": target_logp - float(other.max()),
                    "oracle_kl": oracle_kl,
                }
            )
    return output


def metrics_from_predictions(predictions: list[dict]) -> dict[str, float | int | str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[row["partition"]].append(row)
    missing = {"forget", "retain", "utility"} - set(grouped)
    if missing:
        raise ValueError(f"Predictions are missing partitions: {sorted(missing)}")

    def mean(partition: str, field: str) -> float:
        return float(np.mean([row[field] for row in grouped[partition]]))

    return {
        "primary_forgetting_metric": "forget_oracle_kl",
        "primary_utility_metric": "utility_nll",
        "forget_oracle_kl": mean("forget", "oracle_kl"),
        "forget_nll": mean("forget", "loss"),
        "forget_target_probability": mean("forget", "target_probability"),
        "forget_log_probability_margin": mean("forget", "conditional_log_probability_margin"),
        "retain_nll": mean("retain", "loss"),
        "retain_accuracy": mean("retain", "correct"),
        "utility_nll": mean("utility", "loss"),
        "utility_perplexity": math.exp(min(20.0, mean("utility", "loss"))),
        "utility_accuracy": mean("utility", "correct"),
        "num_examples": len(predictions),
    }
