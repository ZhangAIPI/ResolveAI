"""Publication requires every fixed-cohort case, including failed decisions."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from scripts.refresh_candidate_results import completed_worker


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = {"name":"model","model_id":"m","revision":"r"}
        self.ref = {"dataset_sha256":"d","family_ids":["a","b"],
                    "task_counts":{"state":1,"identity":1}}
        self.cases = {"a":{"task":"state","provenance_truth":"Supported"},
                      "b":{"task":"identity","provenance_truth":"Refuted"}}
        (self.root/"manifest.json").write_text(json.dumps({
            "configuration":{"model_id":"m","revision":"r","dataset_sha256":"d"},
            "source_commit":"c"}))
        self.rows = [{"family_id":family,"variant":"Sufficient","arm":arm,
                      "source_label":case["provenance_truth"],
                      "verdict":None if family=="a" else "Refuted"}
                     for family,case in self.cases.items()
                     for arm in ("initial_text","full_available_text","interactive_agent")]

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self,rows):
        with sqlite3.connect(self.root/"episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes (result TEXT)")
            db.executemany("INSERT INTO episodes VALUES (?)",[(json.dumps(r),) for r in rows])

    def test_incomplete_model_is_not_scored_on_a_smaller_subset(self):
        self.write_rows(self.rows[:-1])
        worker,status = completed_worker(self.spec,self.root,self.ref,self.cases,24)
        self.assertIsNone(worker)
        self.assertEqual(status["status"],"paper_cohort_incomplete")
        self.assertEqual(status["paper_cases_by_setting"]["interactive_agent"],1)

    def test_complete_cohort_can_be_published_before_rest_of_pool_finishes(self):
        self.write_rows(self.rows)
        worker,status = completed_worker(self.spec,self.root,self.ref,self.cases,24)
        self.assertTrue(worker["paper_cohort_complete"])
        self.assertFalse(worker["full_study_complete"])
        self.assertEqual(worker["arms"]["interactive_agent"]["episodes"],2)
        self.assertEqual(worker["arms"]["interactive_agent"]["label_matches"],1)
        self.assertEqual(worker["arms"]["interactive_agent"]["submitted"],1)

    def test_source_label_change_is_rejected(self):
        self.rows[0]["source_label"] = "Refuted"
        self.write_rows(self.rows)
        with self.assertRaisesRegex(ValueError,"source label"):
            completed_worker(self.spec,self.root,self.ref,self.cases,24)


if __name__ == "__main__":
    unittest.main()
