import unittest
from pathlib import Path

from unlearn_bench.config import resolve_experiment, validate_experiment

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


if __name__ == "__main__":
    unittest.main()
