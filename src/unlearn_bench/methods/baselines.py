from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from unlearn_bench.methods.common import clone_model, cross_entropy_loss
from unlearn_bench.methods.pcgu import pcgu_step
from unlearn_bench.models.tiny import TinyAssociationLM, Vocabulary


def supervised_train(
    model: TinyAssociationLM,
    vocabulary: Vocabulary,
    records: list[dict],
    *,
    steps: int,
    learning_rate: float,
) -> TinyAssociationLM:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model, vocabulary, records)
        loss.backward()
        optimizer.step()
    return model


def gradient_ascent(
    model: TinyAssociationLM,
    vocabulary: Vocabulary,
    forget: list[dict],
    retain: list[dict],
    *,
    steps: int,
    learning_rate: float,
    retain_weight: float,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = -cross_entropy_loss(model, vocabulary, forget)
        if retain_weight:
            loss = loss + retain_weight * cross_entropy_loss(model, vocabulary, retain)
        loss.backward()
        optimizer.step()


def npo(
    model: TinyAssociationLM,
    reference: TinyAssociationLM,
    vocabulary: Vocabulary,
    forget: list[dict],
    retain: list[dict],
    *,
    steps: int,
    learning_rate: float,
    beta: float,
    retain_weight: float,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    prompts = [vocabulary.encode_prompt(row["prompt"]) for row in forget]
    targets = torch.tensor(
        [vocabulary.encode_completion(row["completion"]) for row in forget],
        device=next(model.parameters()).device,
    )
    with torch.no_grad():
        reference_logp = (
            F.log_softmax(reference(prompts), dim=-1).gather(1, targets[:, None]).squeeze(1)
        )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model_logp = F.log_softmax(model(prompts), dim=-1).gather(1, targets[:, None]).squeeze(1)
        forget_loss = -(2.0 / beta) * F.logsigmoid(beta * (reference_logp - model_logp)).mean()
        loss = forget_loss + retain_weight * cross_entropy_loss(model, vocabulary, retain)
        loss.backward()
        optimizer.step()


def sham_update(model: TinyAssociationLM, *, scale: float, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed + 9109)
    noises = []
    parameters = []
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            noises.append(noise.to(parameter.device))
            parameters.append(parameter)
        global_norm = torch.sqrt(sum(torch.sum(noise.float() ** 2) for noise in noises))
        multiplier = scale / global_norm.clamp_min(1e-12).item()
        for parameter, noise in zip(parameters, noises, strict=True):
            parameter.add_(noise, alpha=multiplier)


def apply_method(
    name: str,
    full_model: TinyAssociationLM,
    base_model: TinyAssociationLM,
    exact_model: TinyAssociationLM,
    vocabulary: Vocabulary,
    train_records: list[dict],
    config: dict[str, Any],
    seed: int,
) -> tuple[TinyAssociationLM, dict[str, Any]]:
    retain = [row for row in train_records if row["partition"] in {"retain", "utility"}]
    forget = [row for row in train_records if row["partition"] == "forget"]
    metadata: dict[str, Any] = {}
    if name == "untouched":
        return clone_model(base_model), metadata
    if name == "trained_full":
        return clone_model(full_model), metadata
    if name == "exact_retrain":
        return clone_model(exact_model), metadata
    model = clone_model(full_model)
    if name == "continued_retain":
        supervised_train(
            model,
            vocabulary,
            retain,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
        )
    elif name == "counterfactual":
        counterfactual = [{**row, "completion": row["replacement"]} for row in forget]
        supervised_train(
            model,
            vocabulary,
            retain + counterfactual,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
        )
    elif name == "gradient_ascent":
        gradient_ascent(
            model,
            vocabulary,
            forget,
            retain,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
            retain_weight=config["retain_weight"],
        )
    elif name == "npo":
        npo(
            model,
            full_model,
            vocabulary,
            forget,
            retain,
            steps=config["steps"],
            learning_rate=config["learning_rate"],
            beta=config["beta"],
            retain_weight=config["retain_weight"],
        )
    elif name == "sham":
        sham_update(model, scale=config["update_l2_norm"], seed=seed)
    elif name == "pcgu":
        selections = []
        for _ in range(config["steps"]):
            selections.append(
                pcgu_step(
                    model,
                    vocabulary,
                    forget,
                    learning_rate=config["learning_rate"],
                    selected_fraction=config["selected_fraction"],
                )
            )
        final = selections[-1]
        metadata["pcgu"] = {
            "selected_partitions": final.selected,
            "total_partitions": final.total,
            "selected_fraction_actual": final.selected / final.total,
            "similarity_min": final.similarity_min,
            "similarity_max": final.similarity_max,
            "gradient_bytes": final.gradient_bytes,
            "mask_bytes": final.mask_bytes,
            "ranking_seconds": final.ranking_seconds,
            "partition_aggregation_dimension": -1,
            "update_gradient": "replacement_target",
        }
    else:
        raise ValueError(f"Unknown method: {name}")
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Method {name} produced non-finite parameters")
    metadata["parameter_l2_distance_from_full"] = math.sqrt(
        sum(
            torch.sum((left.detach() - right.detach()) ** 2).item()
            for left, right in zip(model.parameters(), full_model.parameters(), strict=True)
        )
    )
    return model, metadata
