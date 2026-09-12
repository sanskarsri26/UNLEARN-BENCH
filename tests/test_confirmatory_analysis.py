import unittest

from scripts.analyze_confirmatory import (
    exact_sign_flip_p,
    hierarchical_interval,
    pareto_methods,
)


class ConfirmatoryAnalysisTests(unittest.TestCase):
    def test_hierarchical_interval_is_deterministic(self):
        differences = {
            11: {"a": -2.0, "b": -1.0},
            29: {"a": -1.5, "b": -0.5},
            47: {"a": -1.0, "b": 0.0},
        }
        first = hierarchical_interval(differences, seed=123)
        second = hierarchical_interval(differences, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first[0], -1.0)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_exact_sign_flip_enumerates_content_units(self):
        differences = {
            seed: {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
            for seed in (11, 29, 47)
        }
        self.assertEqual(exact_sign_flip_p(differences), 0.125)

    def test_pareto_frontier_uses_strict_dominance(self):
        rows = [
            {"model": "m", "method": "a", "forget": 1.0, "utility": 2.0},
            {"model": "m", "method": "b", "forget": 2.0, "utility": 1.0},
            {"model": "m", "method": "c", "forget": 2.0, "utility": 2.0},
        ]
        self.assertEqual(pareto_methods(rows, "m", ("forget", "utility")), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
