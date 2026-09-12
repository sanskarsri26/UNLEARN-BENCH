import unittest

from unlearn_bench.evaluation import evaluate_model
from unlearn_bench.methods import supervised_train
from unlearn_bench.models import TinyAssociationLM, build_vocabulary
from unlearn_bench.utils.reproducibility import set_seed


class ReproducibilityTests(unittest.TestCase):
    def _run(self):
        rows = [
            {
                "id": "x",
                "split": "test",
                "partition": "forget",
                "category": "fact",
                "prompt": "key x",
                "completion": "one",
                "replacement": "two",
            },
            {
                "id": "y",
                "split": "test",
                "partition": "retain",
                "category": "fact",
                "prompt": "key y",
                "completion": "two",
                "replacement": None,
            },
        ]
        vocabulary = build_vocabulary(rows)
        set_seed(19)
        model = TinyAssociationLM(len(vocabulary.tokens), 4)
        supervised_train(model, vocabulary, rows, steps=3, learning_rate=0.1)
        return evaluate_model(model, model, vocabulary, rows)

    def test_fixed_seed_has_identical_predictions(self):
        self.assertEqual(self._run(), self._run())


if __name__ == "__main__":
    unittest.main()
