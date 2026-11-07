"""CPU orchestration checks for the procedural runner; no model or real dataset.

The small fake grader exercises data flow and accounting. Certificate and region
semantics are covered separately by test_multistep.py.
"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from resolveai.conversation import TrajectoryStore
from resolveai.multistep import observation_token
from scripts import evaluate_multistep as runner


def answer(name, arguments, text=""):
    call = {"name": name, "arguments": arguments}
    return {
        "parse_error": False,
        "tool_call": call,
        "raw": "<tool_call>" + json.dumps(call) + "</tool_call>",
        "text": text,
        "input_tokens": 100,
        "output_tokens": 20,
    }


def request(view, text=""):
    return answer("request_photo", {
        "query": {"object": "inspection scene", "time": "inspection", "view": view},
    }, text)


def finish(verdict="Need more evidence"):
    citations = []
    if verdict != "Need more evidence":
        citations = [
            {"image_id": view, "bbox": [0, 0, 8, 8],
             "time": "reference" if view == "reference" else "inspection"}
            for view in ("reference", "front_survey", "rear_left")
        ]
    return answer("finish", {"verdict": verdict, "citations": citations, "links": []})


def fake_score(case, decision, tokens):
    views = {token["view"] for token in tokens}
    sufficient = {"front_survey", "rear_left"} <= views
    correct = bool(decision and decision["verdict"] == "Supported")
    grounded = correct and sufficient
    return {
        "source_label_correct": correct,
        "schema_valid_finish": decision is not None,
        "procedural_evidence_sufficient": sufficient,
        "simulator_grounded_correct": grounded,
        "chain_grounded_correct": grounded,
        "localized_grounded_correct": grounded,
        "unsupported_decision": bool(
            decision and decision["verdict"] != "Need more evidence" and not grounded),
        "minimum_request_count": 2,
    }


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.inputs = []

    def ask_conversation(self, payload, max_context_tokens):
        if max_context_tokens != 8192 or payload["max_new_tokens"] != 768:
            raise AssertionError("The common inference limits changed")
        self.inputs.append(deepcopy(payload))
        return self.answers.pop(0)


class MultistepRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        evidence = []
        for index, view in enumerate(("reference", "initial", *runner.REQUEST_VIEWS)):
            color = ((40, 40, 40) if view in ("initial", "original")
                     else (index * 20, 30, 100))
            Image.new("RGB", (8, 8), color).save(self.root / (view + ".png"))
            reference = view == "reference"
            evidence.append({
                "id": view, "path": view + ".png", "source_id": view,
                "party": "A" if reference else "B",
                "object": "reference crate" if reference else "inspection scene",
                "time": "reference" if reference else "inspection",
                "view": view, "camera_view": view, "available": True,
            })
        self.case = {
            "case_id": "cpu-fixture-Obtainable",
            "family_id": "cpu-fixture",
            "paired_history_id": "cpu-pair",
            "claim": "Synthetic fixture only.",
            "initial": ["reference", "initial"],
            "evidence": evidence,
            "annotation": {
                "verdict": "Supported",
                "minimal_evidence_sets": {"Supported": [["front_survey", "rear_left"]]},
            },
            "request_options": {
                "objects": ["inspection scene"], "times": ["inspection"],
                "views": list(runner.REQUEST_VIEWS),
            },
            "multistep_certificate": {"target_rear_view": "rear_left"},
        }
        self.args = SimpleNamespace(
            budget=6, max_turns=8, max_new_tokens=768,
            max_context_tokens=8192, seed=20261107)

    def tearDown(self):
        self.temp.cleanup()

    def run_arm(self, policy, answers):
        client = FakeClient(answers)
        row, trace = runner.run_episode(
            self.case, self.root, client, policy, self.args,
            observation_token, fake_score)
        return client, row, trace

    def test_adaptive_two_steps_and_common_terminal(self):
        client, row, trace = self.run_arm("adaptive_acquisition", [
            request("front_survey", "PRIVATE_SELECTOR_PROSE"),
            request("rear_left", "SECOND_SELECTOR_PROSE"),
            finish("Supported"),
        ])
        self.assertEqual((row["requests"], row["request_cost"], row["model_calls"]), (2, 6, 3))
        self.assertTrue(row["finished"])
        self.assertTrue(row["multistep_evaluation"]["chain_grounded_correct"])
        self.assertEqual(row["new_image_requests"], 2)
        self.assertNotIn("SELECTOR_PROSE", json.dumps(client.inputs[-1]))
        self.assertIn("PRIVATE_SELECTOR_PROSE", json.dumps(trace["generation"]))
        self.assertEqual(
            [tool["function"]["name"] for tool in client.inputs[-1]["tools"]], ["finish"])
        fixed_client, fixed_row, _ = self.run_arm("fixed_restoration", [finish()])
        self.assertEqual(row["initial_public_sha256"], fixed_row["initial_public_sha256"])
        self.assertEqual(client.inputs[-1]["tools"], fixed_client.inputs[-1]["tools"])
        self.assertEqual(
            client.inputs[-1]["messages"][0], fixed_client.inputs[-1]["messages"][0])

    def test_fixed_restoration_repeats_without_new_pixels(self):
        _, row, trace = self.run_arm("fixed_restoration", [finish()])
        self.assertEqual(
            (row["requests"], row["new_image_requests"], row["model_calls"]), (2, 0, 1))
        self.assertTrue(row["finished"])
        self.assertEqual([
            item["call"]["arguments"]["query"]["view"]
            for item in trace["acquisition"]["executed_requests"]
        ], ["original", "original"])

    def test_fixed_and_oracle_use_real_public_calls(self):
        for policy in ("fixed_view_order", "oracle"):
            client, row, _ = self.run_arm(policy, [finish("Supported")])
            self.assertTrue(row["selection_complete"])
            self.assertEqual(row["new_image_requests"], 2)
            self.assertNotIn("multistep_certificate", json.dumps(client.inputs[-1]))

    def test_paired_random_has_no_replacement(self):
        first = runner.scheduled_views(self.case, "random_request", 7)
        other = runner.scheduled_views(
            {**self.case, "family_id": "other-family"}, "random_request", 7)
        self.assertEqual(first, other)
        self.assertEqual(len(set(first)), 2)

    def test_invalid_selector_is_retained_without_retry_or_prose(self):
        invalid = {
            "parse_error": True, "raw": "INVALID_SELECTOR_PROSE",
            "input_tokens": 100, "output_tokens": 4,
        }
        client, row, trace = self.run_arm("adaptive_acquisition", [invalid, finish()])
        self.assertEqual(len(client.inputs), 2)
        self.assertFalse(row["selection_complete"])
        self.assertFalse(row["multistep_evaluation"]["source_label_correct"])
        self.assertNotIn("INVALID_SELECTOR_PROSE", json.dumps(client.inputs[-1]))
        self.assertIn("INVALID_SELECTOR_PROSE", json.dumps(trace["generation"]))

    def test_invalid_terminal_type_override_fails_without_retry(self):
        client, row, _ = self.run_arm("fixed_restoration", [
            answer("finish", {"type": "inspect", "image_id": "initial"}),
        ])
        self.assertEqual(len(client.inputs), 1)
        self.assertFalse(row["finished"])
        self.assertEqual(row["termination"], "invalid_terminal")
        self.assertFalse(row["multistep_evaluation"]["chain_grounded_correct"])

    def test_saved_resume_and_separate_metric_summary(self):
        _, row, trace = self.run_arm("fixed_restoration", [finish()])
        output = self.root / "output"
        output.mkdir()
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE episodes (case_id TEXT, policy TEXT, result TEXT, "
                       "trace TEXT, PRIMARY KEY(case_id,policy))")
            runner.save_record(
                db, TrajectoryStore(output / "pending-trace.jsonl"), row, trace)
            expected = {(self.case["case_id"], "fixed_restoration")}
            saved, _ = runner.validate_saved(
                db, expected, {self.case["case_id"]: self.case}, output, fake_score)
            self.assertEqual(saved, expected)
            summary = runner.summarize(db, [self.case], ["fixed_restoration"])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["metric_scope"], "procedural")
        self.assertFalse(summary["independent_human_review"])
        counts = summary["policies"]["fixed_restoration"]["counts"]
        for key in ("chain_grounded_correct", "localized_grounded_correct",
                    "simulator_grounded_correct", "unsupported_decisions"):
            self.assertIn(key, counts)


if __name__ == "__main__":
    unittest.main()
