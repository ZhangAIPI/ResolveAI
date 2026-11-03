"""Complete-cohort admission and direct-only API conditions, without API requests."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.score_api_results import collect,api_usage
from scripts.paper_result_table import render


class ApiResultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.directory=Path(self.temp.name)
        self.reference={"dataset_sha256":"data","family_ids":["a","b"]}
        self.cases={"a":{"task":"state","provenance_truth":"Supported"},
                    "b":{"task":"identity","provenance_truth":"Refuted"}}
        config={"dataset_sha256":"data","family_ids":["a","b"],"model_id":"gpt-6-luna",
                "api_config":{"model_id":"gpt-6-luna"}}
        (self.directory/"manifest.json").write_text(json.dumps({"configuration":config,"source_commit":"code"}))
        self.rows=[]
        for family,case in self.cases.items():
            for arm in ("initial_text","full_available_text","interactive_agent"):
                row={"family_id":family,"arm":arm,"variant":"Sufficient",
                     "source_label":case["provenance_truth"],"task":case["task"],
                     "verdict":None if family=="a" else "Refuted",
                     "termination":"max_turns" if family=="a" else "finished",
                     "latency_s":.3,"tool_calls":0,"requests":0}
                trace={"messages":[],"wall_latency_s":.3,
                       "generation":[{"input_tokens":100,"output_tokens":10,"latency_s":.2}]}
                self.rows.append((row,trace))

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        with sqlite3.connect(self.directory/"episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes (result TEXT,trace TEXT)")
            db.executemany("INSERT INTO episodes VALUES (?,?)",
                          [(json.dumps(r),json.dumps(t)) for r,t in rows])

    def test_partial_cohort_is_not_scored(self):
        self.write_rows(self.rows[:-1])
        worker,status=collect(self.directory,self.reference,self.cases)
        self.assertIsNone(worker)
        self.assertFalse(status["ready"])

    def test_failed_answers_remain_in_complete_cohort(self):
        self.write_rows(self.rows)
        worker,status=collect(self.directory,self.reference,self.cases)
        self.assertTrue(status["ready"])
        self.assertEqual(worker["arms"]["interactive_agent"]["episodes"],2)
        self.assertEqual(worker["arms"]["interactive_agent"]["label_matches"],1)

    def test_replayed_generation_does_not_look_like_free_inference(self):
        row,trace=deepcopy(self.rows[0])
        trace["generation"][0].update(response_reused=True,replay_wall_s=.01)
        cost=api_usage(row,trace)
        self.assertAlmostEqual(cost["wall_s"],.49)

    def test_decisions_tool_columns_are_inapplicable(self):
        root=Path(__file__).resolve().parents[1]
        reference=json.loads((root/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
        costs=json.loads((root/"docs/results/cvpr_efficiency.json").read_text())
        source=reference["workers"][0]
        efficiency=next(c for c in costs["workers"] if c["model_id"]==source["model_id"])
        closed={"dataset_sha256":reference["dataset_sha256"],"family_ids":reference["family_ids"],
                "workers":[{"name":"luna_decisions","endpoint":"decisions",
                            "arms":{"initial_text":source["arms"]["initial_text"]},
                            "costs":{"initial_text":efficiency["arms"]["initial_text"]}}]}
        table=render(reference,costs,closed)
        row=next(line for line in table.splitlines() if line.startswith("GPT-6 Luna (Decisions)"))
        self.assertTrue(row.endswith("-- & -- & -- & -- & --"+r"\\"))
        self.assertNotIn(r"\textbf{--}",row)
        with self.assertRaises(ValueError):
            render(reference,costs,{**closed,"family_ids":[]})


if __name__=="__main__":
    unittest.main()
