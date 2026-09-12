import unittest

from scripts.evaluate_stereoset_track_b import inserted_candidate, prepare_items


class TrackBTests(unittest.TestCase):
    def test_extracts_end_and_middle_insertions(self):
        self.assertEqual(
            inserted_candidate("The player was BLANK.", "The player was patient."),
            ("The player was", "patient"),
        )
        self.assertEqual(
            inserted_candidate("They are BLANK at heart.", "They are kind at heart."),
            ("They are", "kind"),
        )
        self.assertEqual(
            inserted_candidate("They are BLANK and very BLANK.", "they are kind and very kind."),
            ("They are", "kind"),
        )

    def test_official_dev_artifact_is_complete(self):
        items = prepare_items()
        self.assertEqual(len(items), 2106)
        self.assertEqual(
            set(items[0]["candidates"]), {"stereotype", "anti_stereotype", "unrelated"}
        )


if __name__ == "__main__":
    unittest.main()
