import unittest

from unlearn_bench.statistics import holm_adjust, paired_bootstrap_ci


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible(self):
        first = paired_bootstrap_ci([1, 2, 3], [0, 1, 1], samples=200, seed=4)
        second = paired_bootstrap_ci([1, 2, 3], [0, 1, 1], samples=200, seed=4)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_holm_is_monotonic_in_rank_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
