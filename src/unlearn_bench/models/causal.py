from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    completion_tokens: int

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def _encode_example(tokenizer, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_text = f"{prompt} {completion}"
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] == prompt_ids:
        first_completion = len(prompt_ids)
    else:
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoded["input_ids"]
        completion_start = len(prompt) + 1
        first_completion = next(
            (
                index
                for index, (_, end) in enumerate(encoded["offset_mapping"])
                if end > completion_start
            ),
            len(full_ids),
        )
    if first_completion == 0 or first_completion >= len(full_ids):
        raise ValueError(f"Could not identify completion tokens for prompt {prompt!r}")
    labels = [-100] * first_completion + full_ids[first_completion:]
    return full_ids, labels


def causal_batch(
    tokenizer,
    records: Iterable[dict],
    device: str | torch.device,
    target_field: str = "completion",
) -> CausalBatch:
    encoded = [_encode_example(tokenizer, row["prompt"], row[target_field]) for row in records]
    if not encoded:
        raise ValueError("Cannot build an empty causal-LM batch")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad or EOS token")
    width = max(len(input_ids) for input_ids, _ in encoded)
    input_rows = []
    label_rows = []
    masks = []
    for input_ids, labels in encoded:
        padding = width - len(input_ids)
        input_rows.append(input_ids + [pad_id] * padding)
        label_rows.append(labels + [-100] * padding)
        masks.append([1] * len(input_ids) + [0] * padding)
    label_tensor = torch.tensor(label_rows, dtype=torch.long, device=device)
    return CausalBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long, device=device),
        attention_mask=torch.tensor(masks, dtype=torch.long, device=device),
        labels=label_tensor,
        completion_tokens=int((label_tensor != -100).sum()),
    )


def token_log_probabilities(model, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(**batch.model_inputs(), use_cache=False).logits[:, :-1]
    labels = batch.labels[:, 1:]
    active = labels != -100
    safe_labels = labels.masked_fill(~active, 0)
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logp, active


def causal_loss(model, batch: CausalBatch) -> torch.Tensor:
    token_logp, active = token_log_probabilities(model, batch)
    return -token_logp[active].mean()


def sequence_log_probabilities(model, batch: CausalBatch) -> torch.Tensor:
    token_logp, active = token_log_probabilities(model, batch)
    return (token_logp * active).sum(dim=1)
