from __future__ import annotations

from typing import Any

from unlearn_bench.utils.device import DeviceContext


def load_causal_lm(config: dict[str, Any], context: DeviceContext):
    """Load a revision-pinned base LM and verify tokenizer/model compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = config["repository"]
    revision = config["revision"]
    tokenizer_revision = config.get("tokenizer_revision", revision)
    tokenizer_name = config.get("tokenizer_repository", name)
    trust_remote_code = config.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": context.dtype,
    }
    if config.get("attention_implementation"):
        model_kwargs["attn_implementation"] = config["attention_implementation"]
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs).to(context.device)
    model_vocab = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > model_vocab:
        raise ValueError(f"Tokenizer has {len(tokenizer)} tokens but model supports {model_vocab}")
    return model, tokenizer
