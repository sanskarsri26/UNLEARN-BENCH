from __future__ import annotations

from typing import Any


def load_causal_lm(config: dict[str, Any]):
    """Load a revision-pinned base LM and verify tokenizer/model compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = config["repository"]
    revision = config["revision"]
    tokenizer_revision = config.get("tokenizer_revision", revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision)
    model_vocab = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > model_vocab:
        raise ValueError(f"Tokenizer has {len(tokenizer)} tokens but model supports {model_vocab}")
    return model, tokenizer
