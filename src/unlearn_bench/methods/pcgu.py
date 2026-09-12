from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from unlearn_bench.methods.common import cross_entropy_loss
from unlearn_bench.models.tiny import TinyAssociationLM, Vocabulary


@dataclass
class PCGUSelection:
    masks: dict[str, torch.Tensor]
    selected: int
    total: int
    similarity_min: float
    similarity_max: float
    gradient_bytes: int
    mask_bytes: int
    ranking_seconds: float


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
    started = time.perf_counter()
    similarity_chunks: list[torch.Tensor] = []
    partitions: list[tuple[str, int, int]] = []
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named_parameters
    }
    offset = 0
    gradient_bytes = 0
    for (name, parameter), grad_a, grad_b in zip(
        named_parameters, gradients_a, gradients_b, strict=True
    ):
        if grad_a is None or grad_b is None or not parameter.requires_grad:
            continue
        width = grad_a.shape[-1] if grad_a.ndim > 1 else grad_a.numel()
        vectors_a = grad_a.reshape(-1, width)
        vectors_b = grad_b.reshape(-1, width)
        similarities = F.cosine_similarity(vectors_a, vectors_b, dim=-1, eps=1e-12)
        similarities = similarities.detach().float().cpu()
        similarity_chunks.append(similarities)
        partitions.append((name, offset, offset + similarities.numel()))
        offset += similarities.numel()
        gradient_bytes += grad_a.numel() * grad_a.element_size()
        gradient_bytes += grad_b.numel() * grad_b.element_size()
    if not similarity_chunks:
        raise ValueError("PCGU found no trainable gradient partitions")
    all_similarities = torch.cat(similarity_chunks)
    count = max(1, math.ceil(all_similarities.numel() * selected_fraction))
    selected_indices = torch.argsort(all_similarities, stable=True)[:count]
    for name, begin, end in partitions:
        local = selected_indices[(selected_indices >= begin) & (selected_indices < end)] - begin
        if local.numel():
            width = masks[name].shape[-1] if masks[name].ndim > 1 else masks[name].numel()
            mask_vectors = masks[name].reshape(-1, width)
            mask_vectors[local.to(mask_vectors.device)] = True
    return PCGUSelection(
        masks=masks,
        selected=count,
        total=all_similarities.numel(),
        similarity_min=float(all_similarities.min()),
        similarity_max=float(all_similarities.max()),
        gradient_bytes=gradient_bytes,
        mask_bytes=sum(mask.numel() * mask.element_size() for mask in masks.values()),
        ranking_seconds=time.perf_counter() - started,
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
