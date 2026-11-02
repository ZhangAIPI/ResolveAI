"""Source-label diagnostics keep variants and unknown identities out of the denominator."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from scripts.score_candidate_labels import score


class CandidateScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / "data"
        data.mkdir()
        cases = [
            {"family_id": f, "variant": "Sufficient", "provenance_truth": label,
             "task": "identity" if label is None else "state"}
            for f, label in [("f1", "Supported"), ("f2", None), ("f3", "Refuted")]
        ]
        self.payload = json.dumps(cases).encode()
        (data / "cases.json").write_bytes(self.payload)
        (self.root / "plan.json").write_text(json.dumps({
            "data": str(data), "family_ids": ["f1", "f2", "f3"],
            "dataset_sha256": hashlib.sha256(self.payload).hexdigest(),
        }))
        models = [{"name": n, "model_id": n, "revision": "r"} for n in ["a", "b"]]
        (self.root / "models.json").write_text(json.dumps(models))
        for model in models:
            folder = self.root / model["name"]
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"source_commit": "c"}))
            rows = []
            for family, label in [("f1", "Supported"), ("f2", None), ("f3", "Refuted")]:
                for arm in ["initial_text", "full_available_text", "interactive_agent"]:
                    if model["name"] == "b" and family == "f3" and arm != "initial_text":
                        continue
                    rows.append({
                        "family_id": family, "variant": "Sufficient", "arm": arm,
                        "source_label": label,
                        "verdict": None if model["name"] == "b" and arm == "interactive_agent" else label,
                    })
            rows.append({"family_id": "f1", "variant": "Obtainable", "arm": "initial_text",
                         "source_label": "Supported", "verdict": "Refuted"})
            with sqlite3.connect(folder / "episodes.sqlite") as db:
                db.execute("CREATE TABLE episodes (result TEXT)")
                db.executemany("INSERT INTO episodes VALUES (?)",
                               [(json.dumps(row),) for row in rows])

    def tearDown(self):
        self.temp.cleanup()

    def test_common_cohort_excludes_unknown_incomplete_and_sibling_versions(self):
        result = score(self.root)
        self.assertEqual(result["family_ids"], ["f1"])
        self.assertEqual(result["unknown_identity_families_excluded"], 1)
        self.assertEqual(result["eligible_families"], 2)
        self.assertIsNone(result["formal_grounded_accuracy"])
        second = result["workers"][1]["arms"]["interactive_agent"]
        self.assertEqual(second["episodes"], 1)
        self.assertEqual(second["submitted"], 0)
        self.assertEqual(second["label_agreement"], 0)

    def test_changed_dataset_is_not_silently_scored(self):
        (self.root / "data/cases.json").write_bytes(self.payload + b" ")
        with self.assertRaisesRegex(ValueError, "Frozen dataset changed"):
            score(self.root)


if __name__ == "__main__":
    unittest.main()
