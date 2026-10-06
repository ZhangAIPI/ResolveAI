"""Diagnostics must isolate choices/access without inventing missing decisions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from resolveai.diagnostic import direct, menu_actions, menu_rollout, outcome, request_query, text_direct
from resolveai.environment import Environment


class NoOCR:
    available = False


class ChoiceClient:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.prompts = []

    def ask_conversation(self, payload, max_context_tokens):
        self.prompts.append(payload)
        choice = next(self.choices)
        return {"choice": choice, "raw": choice}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (24, 24)).save(self.root/"preview.png")
        Image.new("RGB", (96, 96)).save(self.root/"original.png")
        self.case = {"case_id": "f-Obtainable", "family_id": "f", "variant": "Obtainable",
            "diagnostic_source": "mvtec", "diagnostic_truth": "Supported", "claim": "A defect is visible.",
            "initial": ["preview"], "request_options": {
                "objects": ["bottle"], "times": ["capture"], "views": ["original"]},
            "evidence": [{"id": name, "source_id": "private-origin", "path": name+".png",
                "object": "bottle", "time": "capture", "party": "A",
                "view": "overview" if name=="preview" else "original", "available": True}
                for name in ["preview", "original"]],
            "annotation": {"verdict": "Supported", "minimal_evidence_sets": {"Supported": [["original"]]}}}

    def tearDown(self):
        self.temp.cleanup()

    def env(self, case=None):
        return Environment(case or self.case, self.root, ocr_backend=NoOCR())

    def test_menu_cannot_cite_unreleased_images_or_private_annotation(self):
        actions = menu_actions(self.env())
        for call in actions.values():
            if call["name"] == "finish":
                self.assertEqual([c["image_id"] for c in call["arguments"]["citations"]], ["preview"])
        self.assertNotIn("private-origin", json.dumps(actions))
        self.assertNotIn("minimal_evidence_sets", json.dumps(actions))

    def test_actual_request_then_decision_is_distinct_from_localization_score(self):
        env = self.env()
        client = ChoiceClient(["D", "A"])
        record = menu_rollout(env, self.case, client, 6)
        row = outcome(self.case, "menu_agent", record, env._released)
        self.assertTrue(row["label_correct"])
        self.assertTrue(row["received_target_original"])
        self.assertEqual(row["requests"], 1)
        self.assertEqual(row["request_events"][0]["status"], "provided")
        self.assertNotIn("evaluation", record)
        self.assertEqual(record["messages"][2]["tool_calls"][0]["function"]["name"], "request_photo")

    def test_unavailable_never_fabricates_pixels_or_rewrites_physical_label(self):
        case = deepcopy(self.case)
        case["evidence"][1]["available"] = False
        env = self.env(case)
        client = ChoiceClient(["D", "C"])
        record = menu_rollout(env, case, client, 6)
        row = outcome(case, "menu_unavailable", record, env._released)
        self.assertEqual(row["verdict"], "Need more evidence")
        self.assertFalse(row["received_target_original"])
        self.assertFalse(row["label_correct"])
        self.assertEqual(row["request_events"][0]["status"], "unable_to_provide")
        self.assertEqual(env.budget, 9)

    def test_direct_has_no_defect_specific_prompt_or_hidden_label(self):
        client = ChoiceClient(["A"])
        case = {**self.case, "claim": "These chairs are the same item.",
                "diagnostic_public_targets": [{"initial_image_id":"preview","source_region":[0,0,24,24]}]}
        record = direct(self.env(case), case, client)
        self.assertEqual(record["decision"]["verdict"], "Supported")
        prompt = json.dumps(client.prompts)
        self.assertNotIn("Supported means a defect", prompt)
        self.assertNotIn("minimal_evidence_sets", prompt)
        self.assertIn("target_regions", prompt)

    def test_text_format_failure_is_not_an_abstention(self):
        class TextClient:
            def __init__(self, raw):
                self.raw = raw
            def ask_conversation(self, payload, max_context_tokens=8192):
                return {"raw": self.raw}
        record = text_direct(self.env(), self.case, TextClient("I do not follow the format."))
        self.assertIsNone(record["decision"])
        self.assertEqual(record["termination"], "invalid_decision")
        record = text_direct(self.env(), self.case, TextClient("Need more evidence"))
        self.assertEqual(record["decision"]["verdict"], "Need more evidence")
        self.assertEqual(record["errors"], 0)

    def test_context_halt_and_turn_exhaustion_are_not_abstentions(self):
        class Halt:
            def ask_conversation(self, *args, **kwargs):
                return {"halt": "context_limit"}
        record = direct(self.env(), self.case, Halt())
        self.assertIsNone(record["decision"])
        self.assertEqual(record["termination"], "context_limit")
        record = menu_rollout(self.env(), self.case, ChoiceClient(["D"]), 1)
        self.assertIsNone(record["decision"])
        self.assertEqual(record["termination"], "max_turns")
        self.assertEqual(request_query(self.case)["view"], "original")


if __name__ == "__main__":
    unittest.main()
