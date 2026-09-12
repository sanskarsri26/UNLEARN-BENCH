import unittest
from types import SimpleNamespace

import torch

from unlearn_bench.evaluation.hf_controlled import evaluate_causal_lm
from unlearn_bench.methods.hf import apply_hf_method, clone_causal_lm, train_causal_lm
from unlearn_bench.models.causal import causal_batch, causal_loss


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.tokens = {"<pad>": 0}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        pieces = text.split()
        ids = []
        offsets = []
        cursor = 0
        for piece in pieces:
            start = text.index(piece, cursor)
            cursor = start + len(piece)
            if piece not in self.tokens:
                self.tokens[piece] = len(self.tokens)
            ids.append(self.tokens[piece])
            offsets.append((start, cursor))
        output = {"input_ids": ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output

    def decode(self, ids):
        inverse = {index: token for token, index in self.tokens.items()}
        return " ".join(inverse.get(index, f"<token-{index}>") for index in ids)


class FakeCausalLM(torch.nn.Module):
    def __init__(self, vocabulary_size=32, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


class HuggingFaceBackendTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.records = [
            {
                "id": "forget",
                "split": "test",
                "partition": "forget",
                "category": "fact",
                "prompt": "code for alder is",
                "completion": "blue",
                "replacement": "red",
            },
            {
                "id": "retain",
                "split": "test",
                "partition": "retain",
                "category": "fact",
                "prompt": "code for birch is",
                "completion": "green",
                "replacement": None,
            },
            {
                "id": "utility",
                "split": "test",
                "partition": "utility",
                "category": "utility",
                "prompt": "opposite of hot is",
                "completion": "cold",
                "replacement": None,
            },
        ]
        for row in self.records:
            self.tokenizer(row["prompt"] + " " + row["completion"])
            if row["replacement"]:
                self.tokenizer(row["prompt"] + " " + row["replacement"])

    def test_completion_only_batch_and_loss(self):
        model = FakeCausalLM()
        batch = causal_batch(self.tokenizer, self.records, "cpu")
        self.assertEqual(batch.completion_tokens, 3)
        self.assertEqual(int((batch.labels != -100).sum()), 3)
        self.assertTrue(torch.isfinite(causal_loss(model, batch)))

    def test_training_evaluation_and_pcgu(self):
        base = FakeCausalLM()
        full = clone_causal_lm(base)
        exact = clone_causal_lm(base)
        train_causal_lm(full, self.tokenizer, self.records, steps=1, learning_rate=0.01)
        train_causal_lm(exact, self.tokenizer, self.records[1:], steps=1, learning_rate=0.01)
        model, metadata = apply_hf_method(
            "pcgu",
            full,
            base,
            exact,
            self.tokenizer,
            self.records,
            {"steps": 1, "learning_rate": 0.01, "selected_fraction": 0.1},
            11,
        )
        predictions = evaluate_causal_lm(model, exact, self.tokenizer, self.records)
        self.assertEqual(len(predictions), 3)
        self.assertGreater(metadata["pcgu"]["selected_partitions"], 0)
        self.assertTrue(all("oracle_kl" in row for row in predictions))


if __name__ == "__main__":
    unittest.main()
