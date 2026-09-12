from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch.nn import functional as F

from unlearn_bench.methods.baselines import sham_update
from unlearn_bench.methods.pcgu import compute_pcgu_selection
from unlearn_bench.models.causal import causal_batch, causal_loss, sequence_log_probabilities


def clone_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


def train_causal_lm(
    model,
    tokenizer,
    records: list[dict],
    *,
    steps: int,
    learning_rate: float,
    target_field: str = "completion",
    batch_size: int | None = None,
    seed: int = 0,
) -> dict[str, int]:
    if batch_size is None:
        batch_size = len(records)
    if not 0 < batch_size <= len(records):
        raise ValueError("batch_size must be in [1, number of records]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order: list[int] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    examples_processed = 0
    tokens_processed = 0
    for _ in range(steps):
        while len(order) < batch_size:
            order.extend(torch.randperm(len(records), generator=generator).tolist())
        indices, order = order[:batch_size], order[batch_size:]
        selected = [records[index] for index in indices]
        batch = causal_batch(tokenizer, selected, next(model.parameters()).device, target_field)
        optimizer.zero_grad(set_to_none=True)
        causal_loss(model, batch).backward()
        optimizer.step()
        examples_processed += len(selected)
        tokens_processed += batch.completion_tokens
    return {
        "steps": steps,
        "examples_processed": examples_processed,
        "tokens_processed": tokens_processed,
        "batch_size": batch_size,
        "seed": seed,
    }


def _gradient_ascent(model, tokenizer, forget, retain, config) -> dict[str, int]:
    device = next(model.parameters()).device
    forget_batch = causal_batch(tokenizer, forget, device)
    retain_batch = causal_batch(tokenizer, retain, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    model.train()
    for _ in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = -causal_loss(model, forget_batch)
        loss = loss + config["retain_weight"] * causal_loss(model, retain_batch)
        loss.backward()
        optimizer.step()
    return {
        "steps": config["steps"],
        "examples_processed": config["steps"] * (len(forget) + len(retain)),
        "tokens_processed": config["steps"]
        * (forget_batch.completion_tokens + retain_batch.completion_tokens),
    }


def _npo(model, reference, tokenizer, forget, retain, config) -> dict[str, int]:
    device = next(model.parameters()).device
    forget_batch = causal_batch(tokenizer, forget, device)
    retain_batch = causal_batch(tokenizer, retain, device)
    with torch.no_grad():
        reference_logp = sequence_log_probabilities(reference, forget_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    beta = config["beta"]
    model.train()
    for _ in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        model_logp = sequence_log_probabilities(model, forget_batch)
        forget_loss = -(2.0 / beta) * F.logsigmoid(beta * (reference_logp - model_logp)).mean()
        loss = forget_loss + config["retain_weight"] * causal_loss(model, retain_batch)
        loss.backward()
        optimizer.step()
    return {
        "steps": config["steps"],
        "examples_processed": config["steps"] * (len(forget) + len(retain)),
        "tokens_processed": config["steps"]
        * (forget_batch.completion_tokens + retain_batch.completion_tokens),
    }


def _pcgu(model, tokenizer, forget, config) -> dict[str, Any]:
    device = next(model.parameters()).device
    original = causal_batch(tokenizer, forget, device, "completion")
    replacement = causal_batch(tokenizer, forget, device, "replacement")
    final = None
    model.train()
    for _ in range(config["steps"]):
        named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        parameters = [parameter for _, parameter in named_parameters]
        original_grads = torch.autograd.grad(
            causal_loss(model, original), parameters, allow_unused=True
        )
        replacement_grads = torch.autograd.grad(
            causal_loss(model, replacement), parameters, allow_unused=True
        )
        final = compute_pcgu_selection(
            named_parameters,
            original_grads,
            replacement_grads,
            config["selected_fraction"],
        )
        with torch.no_grad():
            for (name, parameter), gradient in zip(
                named_parameters, replacement_grads, strict=True
            ):
                if gradient is not None:
                    parameter.add_(gradient * final.masks[name], alpha=-config["learning_rate"])
    assert final is not None
    return {
        "steps": config["steps"],
        "examples_processed": config["steps"] * len(forget) * 2,
        "tokens_processed": config["steps"]
        * (original.completion_tokens + replacement.completion_tokens),
        "pcgu": {
            "selected_partitions": final.selected,
            "total_partitions": final.total,
            "selected_fraction_actual": final.selected / final.total,
            "similarity_min": final.similarity_min,
            "similarity_max": final.similarity_max,
            "gradient_bytes": final.gradient_bytes,
            "mask_bytes": final.mask_bytes,
            "ranking_seconds_last_step": final.ranking_seconds,
            "partition_aggregation_dimension": -1,
            "update_gradient": "replacement_target",
        },
    }


def apply_hf_method(
    name: str,
    full_model,
    base_model,
    exact_model,
    tokenizer,
    train_records: list[dict],
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    retain = [row for row in train_records if row["partition"] in {"retain", "utility"}]
    forget = [row for row in train_records if row["partition"] == "forget"]
    if name == "untouched":
        metadata = {"steps": 0, "examples_processed": 0, "tokens_processed": 0}
        return clone_causal_lm(base_model), metadata
    if name == "trained_full":
        metadata = {"steps": 0, "examples_processed": 0, "tokens_processed": 0}
        return clone_causal_lm(full_model), metadata
    if name == "exact_retrain":
        metadata = {"steps": 0, "examples_processed": 0, "tokens_processed": 0}
        return clone_causal_lm(exact_model), metadata
    model = clone_causal_lm(full_model)
    if name == "continued_retain":
        metadata = train_causal_lm(
            model,
            tokenizer,
            retain,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
            batch_size=config.get("batch_size"),
            seed=seed + 101,
        )
    elif name == "counterfactual":
        counterfactual = [{**row, "completion": row["replacement"]} for row in forget]
        metadata = train_causal_lm(
            model,
            tokenizer,
            retain + counterfactual,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
            batch_size=config.get("batch_size"),
            seed=seed + 211,
        )
    elif name == "gradient_ascent":
        metadata = _gradient_ascent(model, tokenizer, forget, retain, config)
    elif name == "npo":
        metadata = _npo(model, full_model, tokenizer, forget, retain, config)
    elif name == "sham":
        sham_update(model, scale=config["update_l2_norm"], seed=seed)
        metadata = {"steps": 1, "examples_processed": 0, "tokens_processed": 0}
    elif name == "pcgu":
        metadata = _pcgu(model, tokenizer, forget, config)
    else:
        raise ValueError(f"Unknown method: {name}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError(f"Method {name} produced non-finite parameters")
    metadata["parameter_l2_distance_from_full"] = math.sqrt(
        sum(
            torch.sum((left.detach().float() - right.detach().float()) ** 2).item()
            for left, right in zip(model.parameters(), full_model.parameters(), strict=True)
        )
    )
    return model, metadata
