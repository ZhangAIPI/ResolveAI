"""The production rollout schema must survive diagnostic persistence."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from scripts.evaluate_study import normalize_agent_record, Records

class EvaluationStorageTests(unittest.TestCase):
    def test_native_metadata_is_normalized_without_changing_the_decision(self):
        decision={"verdict":"Need more evidence","citations":[]}
        trace={"metadata":{"termination":"finished","errors":0},"decision":decision}
        result=normalize_agent_record(trace,SimpleNamespace(requests=2,calls=4))
        self.assertEqual(result["termination"],"finished")
        self.assertIs(result["decision"],decision)
        self.assertEqual(result["requests"],2)

    def test_resume_finds_committed_result_and_matching_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            records=Records(Path(directory))
            row={"case_id":"c","arm":"initial_text","termination":"finished","errors":0,
                 "requests":0,"source_label":None,"source_label_match":None}
            trace={"messages":[],"decision":{"verdict":"Need more evidence"}}
            records.save(row,trace)
            self.assertTrue(records.has("c","initial_text"))
            self.assertFalse(records.has("c","interactive_agent"))
            self.assertEqual(records.summary()["episodes"],1)
            self.assertFalse(records.store.output.exists())
            records.db.close()
            reopened=Records(Path(directory))
            self.assertTrue(reopened.has("c","initial_text"))
            reopened.db.close()

if __name__=="__main__": unittest.main()
