from __future__ import annotations

import torch
from torch.nn import functional as F

from unlearn_bench.models.causal import causal_batch


def evaluate_causal_lm(model, oracle, tokenizer, records: list[dict]) -> list[dict]:
    model.eval()
    oracle.eval()
    output = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for row in records:
            batch = causal_batch(tokenizer, [row], device)
            logits = model(**batch.model_inputs(), use_cache=False).logits[:, :-1]
            oracle_logits = oracle(**batch.model_inputs(), use_cache=False).logits[:, :-1]
            labels = batch.labels[:, 1:]
            active = labels != -100
            active_logits = logits[active].float()
            active_oracle_logits = oracle_logits[active].float()
            targets = labels[active]
            log_probs = F.log_softmax(active_logits, dim=-1)
            oracle_log_probs = F.log_softmax(active_oracle_logits, dim=-1)
            target_logp = log_probs.gather(1, targets[:, None]).squeeze(1)
            target_mask = F.one_hot(targets, num_classes=log_probs.shape[-1]).bool()
            other_max = log_probs.masked_fill(target_mask, -torch.inf).max(dim=-1).values
            predictions = log_probs.argmax(dim=-1)
            oracle_probs = oracle_log_probs.exp()
            oracle_kl = torch.sum(oracle_probs * (oracle_log_probs - log_probs), dim=-1).mean()
            mean_logp = target_logp.mean()
            output.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "partition": row["partition"],
                    "category": row["category"],
                    "target": row["completion"],
                    "prediction": tokenizer.decode([int(predictions[0])]),
                    "correct": bool(torch.equal(predictions, targets)),
                    "loss": float(-mean_logp),
                    "target_log_probability": float(mean_logp),
                    "target_probability": float(mean_logp.exp()),
                    "conditional_log_probability_margin": float((target_logp - other_max).mean()),
                    "oracle_kl": float(oracle_kl),
                    "completion_token_count": targets.numel(),
                }
            )
    return output
