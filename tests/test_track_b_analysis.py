import unittest

from scripts.analyze_track_b import paired_deltas


class TrackBAnalysisTests(unittest.TestCase):
    def test_neutrality_delta_uses_distance_from_50(self):
        cells = {}
        for model in ("pythia-160m", "mamba-130m-hf"):
            for method in (
                "untouched",
                "trained_full",
                "exact_retrain",
                "continued_retain",
                "sham",
                "counterfactual",
                "gradient_ascent",
                "npo",
                "pcgu",
            ):
                for seed in (11, 29, 47):
                    ss = 55.0 if method == "trained_full" else 52.0
                    cells[(model, method, seed)] = {
                        "metrics": {"overall": {"lms": 80.0, "ss": ss, "icat": 72.0}}
                    }
        rows = paired_deltas(cells)
        pcgu = next(
            row for row in rows if row["model"] == "pythia-160m" and row["method"] == "pcgu"
        )
        self.assertEqual(pcgu["neutrality_delta_mean"], -3.0)


if __name__ == "__main__":
    unittest.main()
