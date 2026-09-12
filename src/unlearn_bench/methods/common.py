from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch.nn import functional as F

from unlearn_bench.models.tiny import TinyAssociationLM, Vocabulary


def clone_model(model: TinyAssociationLM) -> TinyAssociationLM:
    return copy.deepcopy(model)


def prompts_and_targets(
    records: Iterable[dict], vocabulary: Vocabulary, target_field: str = "completion"
) -> tuple[list[list[int]], torch.Tensor]:
    rows = list(records)
    prompts = [vocabulary.encode_prompt(row["prompt"]) for row in rows]
    targets = torch.tensor(
        [vocabulary.encode_completion(row[target_field]) for row in rows], dtype=torch.long
    )
    return prompts, targets


def cross_entropy_loss(
    model: TinyAssociationLM,
    vocabulary: Vocabulary,
    records: Iterable[dict],
    target_field: str = "completion",
) -> torch.Tensor:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot compute a loss for an empty record set")
    prompts, targets = prompts_and_targets(rows, vocabulary, target_field)
    targets = targets.to(next(model.parameters()).device)
    return F.cross_entropy(model(prompts), targets)


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
