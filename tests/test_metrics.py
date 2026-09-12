import unittest

from unlearn_bench.evaluation.controlled import metrics_from_predictions
from unlearn_bench.evaluation.stereoset import stereoset_metrics


class MetricTests(unittest.TestCase):
    def test_stereoset_ideal_fixture(self):
        rows = [
            {"category": "gender", "stereotype": 3.0, "anti_stereotype": 2.0, "unrelated": 0.0},
            {"category": "gender", "stereotype": 2.0, "anti_stereotype": 3.0, "unrelated": 0.0},
        ]
        metrics = stereoset_metrics(rows)["overall"]
        self.assertEqual(metrics["lms"], 100.0)
        self.assertEqual(metrics["ss"], 50.0)
        self.assertEqual(metrics["icat"], 100.0)

    def test_destroyed_model_is_penalized_by_utility(self):
        rows = []
        for partition in ("forget", "retain", "utility"):
            rows.append(
                {
                    "partition": partition,
                    "oracle_kl": 0.2,
                    "loss": 5.0 if partition == "utility" else 1.0,
                    "target_probability": 0.1,
                    "conditional_log_probability_margin": -1.0,
                    "correct": False,
                }
            )
        metrics = metrics_from_predictions(rows)
        self.assertEqual(metrics["utility_nll"], 5.0)
        self.assertGreater(metrics["utility_perplexity"], 100)


if __name__ == "__main__":
    unittest.main()
