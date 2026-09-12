import unittest
from unittest.mock import MagicMock, patch

import torch

from unlearn_bench.models import TinyAssociationLM, build_vocabulary
from unlearn_bench.models.hf import load_causal_lm
from unlearn_bench.utils.device import DeviceContext


class ModelTests(unittest.TestCase):
    def test_tiny_model_vocabulary_compatibility(self):
        rows = [{"prompt": "A small prompt", "completion": "answer", "replacement": None}]
        vocabulary = build_vocabulary(rows)
        model = TinyAssociationLM(len(vocabulary.tokens), hidden_size=4)
        logits = model([vocabulary.encode_prompt(rows[0]["prompt"])])
        self.assertEqual(tuple(logits.shape), (1, len(vocabulary.tokens)))

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_hf_loader_records_pinned_configuration(self, tokenizer_loader, model_loader):
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 10
        tokenizer_loader.return_value = tokenizer
        model = MagicMock()
        model.get_input_embeddings.return_value.num_embeddings = 10
        model.to.return_value = model
        model_loader.return_value = model
        context = DeviceContext("cpu", "cpu", "cpu", "fp32", "fp32", torch.float32, False)
        config = {
            "repository": "owner/model",
            "revision": "model-sha",
            "tokenizer_repository": "owner/tokenizer",
            "tokenizer_revision": "tokenizer-sha",
            "trust_remote_code": False,
            "attention_implementation": "eager",
        }
        loaded_model, loaded_tokenizer = load_causal_lm(config, context)
        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        tokenizer_loader.assert_called_once_with(
            "owner/tokenizer", revision="tokenizer-sha", trust_remote_code=False
        )
        model_loader.assert_called_once_with(
            "owner/model",
            revision="model-sha",
            trust_remote_code=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
