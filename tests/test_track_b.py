import unittest

from scripts.evaluate_stereoset_track_b import causal_prompt, inserted_candidate, prepare_items


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
        empty = [
            candidate
            for item in items
            for candidate in item["candidates"].values()
            if not candidate["prompt"]
        ]
        self.assertEqual(len(empty), 57)

    def test_empty_context_uses_pinned_special_token(self):
        class Tokenizer:
            bos_token = "<bos>"
            eos_token = "<eos>"

        self.assertEqual(causal_prompt(Tokenizer(), ""), "<bos>")
        self.assertEqual(causal_prompt(Tokenizer(), "context"), "context")


if __name__ == "__main__":
    unittest.main()
