import unittest

from unlearn_bench.preregistration import validate_marker_structure


class PreregistrationTests(unittest.TestCase):
    def test_complete_marker_structure(self):
        marker = {
            "schema_version": 1,
            "reviewer": "test reviewer",
            "reviewed_at_utc": "2026-09-11T12:00:00+00:00",
            "preregistration_commit": "a" * 40,
            "preregistration_path": "docs/preregistration_main.md",
            "authorized_experiments": ["main"],
            "frozen_files_sha256": {"docs/preregistration_main.md": "b" * 64},
            "compute_budget_gpu_hours": 1.0,
        }
        validate_marker_structure(marker)

    def test_marker_rejects_short_commit(self):
        marker = {
            "schema_version": 1,
            "reviewer": "test reviewer",
            "reviewed_at_utc": "2026-09-11T12:00:00+00:00",
            "preregistration_commit": "abc123",
            "preregistration_path": "docs/preregistration_main.md",
            "authorized_experiments": ["main"],
            "frozen_files_sha256": {"docs/preregistration_main.md": "b" * 64},
            "compute_budget_gpu_hours": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "full Git SHA"):
            validate_marker_structure(marker)


if __name__ == "__main__":
    unittest.main()
