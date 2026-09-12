from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from unlearn_bench.methods.common import cross_entropy_loss
from unlearn_bench.models.tiny import TinyAssociationLM, Vocabulary


@dataclass
class PCGUSelection:
    masks: dict[str, torch.Tensor]
    similarities: list[float]
    selected: int
    total: int


def _vector_partitions(name: str, tensor: torch.Tensor):
    """Yield the paper's input-aggregation vector partition (all but final index)."""
    if tensor.ndim == 0:
        raise ValueError(f"PCGU cannot partition scalar parameter {name}")
    if tensor.ndim == 1:
        yield name, None, tensor
        return
    prefix_shape = tensor.shape[:-1]
    for flat_index in range(math.prod(prefix_shape)):
        index = []
        remainder = flat_index
        for size in reversed(prefix_shape):
            index.append(remainder % size)
            remainder //= size
        prefix = tuple(reversed(index))
        yield name, prefix, tensor[prefix]


def compute_pcgu_selection(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients_a: tuple[torch.Tensor | None, ...],
    gradients_b: tuple[torch.Tensor | None, ...],
    selected_fraction: float,
) -> PCGUSelection:
    """Rank gradient-vector partitions by ascending cosine similarity.

    This follows Yu et al. (2023): input-aggregated parameter vectors with the lowest
    contrastive-gradient cosine similarity are selected. Ties are resolved stably by
    parameter order and vector index.
    """
    if not 0 < selected_fraction <= 1:
        raise ValueError("selected_fraction must be in (0, 1]")
    candidates = []
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named_parameters
    }
    for order, ((name, parameter), grad_a, grad_b) in enumerate(
        zip(named_parameters, gradients_a, gradients_b, strict=True)
    ):
        if grad_a is None or grad_b is None or not parameter.requires_grad:
            continue
        for vector_order, (_, prefix, vector_a) in enumerate(_vector_partitions(name, grad_a)):
            vector_b = grad_b if prefix is None else grad_b[prefix]
            similarity = F.cosine_similarity(
                vector_a.reshape(1, -1), vector_b.reshape(1, -1), dim=-1, eps=1e-12
            ).item()
            candidates.append((similarity, order, vector_order, name, prefix))
    if not candidates:
        raise ValueError("PCGU found no trainable gradient partitions")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    count = max(1, math.ceil(len(candidates) * selected_fraction))
    for _, _, _, name, prefix in candidates[:count]:
        if prefix is None:
            masks[name].fill_(True)
        else:
            masks[name][prefix] = True
    return PCGUSelection(
        masks=masks,
        similarities=[item[0] for item in candidates],
        selected=count,
        total=len(candidates),
    )


def pcgu_step(
    model: TinyAssociationLM,
    vocabulary: Vocabulary,
    contrastive_records: list[dict],
    learning_rate: float,
    selected_fraction: float,
) -> PCGUSelection:
    """Perform one PCGU update using original/replacement targets as the pair.

    Original PCGU targets masked-LM social-bias pairs. This controlled causal-LM
    adaptation uses the same partition/rank/mask/update algorithm, with the learned
    forget target as the first member and a versioned replacement as the second.
    """
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    original_loss = cross_entropy_loss(model, vocabulary, contrastive_records, "completion")
    original_grads = torch.autograd.grad(original_loss, parameters, retain_graph=False)
    replacement_loss = cross_entropy_loss(model, vocabulary, contrastive_records, "replacement")
    replacement_grads = torch.autograd.grad(replacement_loss, parameters, retain_graph=False)
    selection = compute_pcgu_selection(
        named_parameters, original_grads, replacement_grads, selected_fraction
    )
    with torch.no_grad():
        for (name, parameter), gradient in zip(named_parameters, replacement_grads, strict=True):
            parameter.add_(gradient * selection.masks[name], alpha=-learning_rate)
    return selection
