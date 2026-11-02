"""Ensure SFT acquisition summaries distinguish asking from receiving new pixels."""
from copy import deepcopy
import unittest

from scripts.score_sft_comparison import acquisition_behavior


class AcquisitionBehaviorTests(unittest.TestCase):
    def fixture(self):
        cases, rows = {}, []
        for variant in ("Sufficient", "Obtainable", "Missing", "Unavailable"):
            case_id = "family-" + variant
            initial = ["a", "b"] if variant == "Sufficient" else ["a"]
            cases[case_id] = {"initial": initial}
            rows.append({
                "case_id": case_id, "family_id": "family", "variant": variant,
                "arm": "interactive_agent", "task": "identity", "termination": "finished",
                "requests": 0 if variant == "Sufficient" else 1,
                "released": ["a", "b"] if variant in ("Sufficient", "Obtainable") else ["a"],
                "verdict": "Supported" if variant in ("Sufficient", "Obtainable") else "Need more evidence",
                "tool_calls": 0 if variant == "Sufficient" else 1,
            })
        return cases, rows

    def test_attempts_and_new_pixels_are_separate(self):
        cases, rows = self.fixture()
        result = acquisition_behavior(rows, cases, {"family"})["by_variant"]
        self.assertEqual(result["Obtainable"]["episodes_with_new_photo"], 1)
        for variant in ("Missing", "Unavailable"):
            self.assertEqual(result[variant]["episodes_with_request"], 1)
            self.assertEqual(result[variant]["episodes_with_new_photo"], 0)
            self.assertEqual(result[variant]["abstained"], 1)
        # Asking for already available material is not a new observation.
        rows[0]["requests"] = 1
        self.assertEqual(acquisition_behavior(rows, cases, {"family"})["by_variant"]
                         ["Sufficient"]["episodes_with_new_photo"], 0)

    def test_incomplete_duplicate_or_unrequested_release_is_rejected(self):
        cases, rows = self.fixture()
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                acquisition_behavior(invalid, cases, {"family"})
        altered = deepcopy(rows)
        altered[1]["requests"] = 0
        with self.assertRaises(ValueError):
            acquisition_behavior(altered, cases, {"family"})


if __name__ == "__main__":
    unittest.main()
