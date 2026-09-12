import unittest

from unlearn_bench.models import TinyAssociationLM, build_vocabulary


class ModelTests(unittest.TestCase):
    def test_tiny_model_vocabulary_compatibility(self):
        rows = [{"prompt": "A small prompt", "completion": "answer", "replacement": None}]
        vocabulary = build_vocabulary(rows)
        model = TinyAssociationLM(len(vocabulary.tokens), hidden_size=4)
        logits = model([vocabulary.encode_prompt(rows[0]["prompt"])])
        self.assertEqual(tuple(logits.shape), (1, len(vocabulary.tokens)))


if __name__ == "__main__":
    unittest.main()
