import unittest
from pathlib import Path

from unlearn_bench.config import resolve_experiment

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_experiment_configs_resolve(self):
        for path in (ROOT / "configs" / "experiments").glob("*.yaml"):
            with self.subTest(path=path.name):
                resolved = resolve_experiment(path)
                self.assertEqual(set(resolved["methods"]), set(resolved["method_configs"]))


if __name__ == "__main__":
    unittest.main()
