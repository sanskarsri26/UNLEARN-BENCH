from __future__ import annotations

import torch
from torch.nn import functional as F


def conditional_log_probability(model, tokenizer, context: str, continuation: str) -> float:
    """Score continuation tokens only for an autoregressive Hugging Face model."""
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("Continuation tokenization does not preserve the context prefix")
    if len(full_ids) <= len(context_ids):
        raise ValueError("Continuation must contain at least one token")
    input_ids = torch.tensor([full_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1]
    labels = input_ids[:, 1:]
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, len(context_ids) - 1)
    return float(token_logp[:, start:].sum())
