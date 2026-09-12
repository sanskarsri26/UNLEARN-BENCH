import unittest
from pathlib import Path

from unlearn_bench.config import resolve_experiment, validate_experiment
from unlearn_bench.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_experiment_configs_resolve(self):
        for path in (ROOT / "configs" / "experiments").glob("*.yaml"):
            with self.subTest(path=path.name):
                resolved = resolve_experiment(path)
                self.assertEqual(set(resolved["methods"]), set(resolved["method_configs"]))

    def test_claim_status_is_explicit_and_validated(self):
        minimal = {
            "name": "test",
            "track": "controlled_unlearning",
            "claim_status": "exploratory",
            "model": "model",
            "dataset": "dataset",
            "methods": ["method"],
            "seeds": [1],
        }
        validate_experiment(minimal)
        with self.assertRaisesRegex(ValueError, "claim_status"):
            validate_experiment({**minimal, "claim_status": "confirm-ish"})

    def test_confirmatory_matrix_is_frozen(self):
        expected = {
            "main.yaml": ("pythia_160m", 0.00001),
            "main_mamba.yaml": ("mamba_130m", 0.00005),
        }
        for filename, (model, learning_rate) in expected.items():
            with self.subTest(filename=filename):
                config = resolve_experiment(ROOT / "configs" / "experiments" / filename)
                self.assertEqual(config["claim_status"], "confirmatory")
                self.assertEqual(config["model"], model)
                self.assertEqual(config["dataset"], "controlled_v2")
                self.assertEqual(config["device"], "cuda")
                self.assertEqual(config["precision"], "fp32")
                self.assertEqual(config["seeds"], [11, 29, 47])
                self.assertEqual(len(config["methods"]), 9)
                self.assertEqual(config["training"]["steps"], 20)
                self.assertEqual(config["training"]["learning_rate"], learning_rate)
                self.assertEqual(config["method_overrides"]["pcgu"]["steps"], 6)

    def test_confirmatory_overrides_cannot_change_frozen_protocol(self):
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            run_experiment(
                ROOT / "configs" / "experiments" / "main.yaml",
                ROOT,
                overrides={"device": "cpu"},
            )


if __name__ == "__main__":
    unittest.main()
