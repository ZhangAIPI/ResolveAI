"""Exercise acquisition decisions with a non-oracle fixture policy."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from resolveai.pilot import episode
from resolveai.vlm_worker import parse_decision


class PixelClient:
    def __init__(self):
        self.observations = []

    def ask(self, observation, allowed=False, status="not_requested"):
        self.observations.append(observation)
        assert "annotation" not in observation
        assert "provenance" not in observation
        ids = [r["image_id"] for r in observation["images"]]
        verdict = "Supported" if "original" in ids else "Need more evidence"
        return {"decision": {"verdict": verdict, "image_ids": ids,
                    "request_original": allowed, "confidence": .5},
                "parse_error": False, "raw": "fixture only", "input_tokens": 1,
                "output_tokens": 1, "latency_s": 0, "peak_memory_gb": 0}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("original", "preview"):
            Image.new("RGB", (8, 8), "gray").save(self.root / (name + ".png"))
        self.case = {"case_id": "f-Obtainable", "family_id": "f", "variant": "Obtainable",
            "claim": "fixture claim", "initial": ["preview"], "evidence": [
                {"id": name, "path": name + ".png", "source_id": "same-source",
                 "party": "public", "time": "capture", "object": "bottle",
                 "view": name, "available": True} for name in ("original", "preview")],
            "request_spec": {"object": "bottle", "time": "capture", "view": "original"},
            "annotation": {"verdict": "Supported", "visual_verdict": "Supported",
                           "minimal_evidence_sets": {"Supported": [["original"]]}},
            "provenance": {"category": "bottle"}}

    def test_agent_gets_only_public_pixels_then_one_request(self):
        client = PixelClient()
        row = episode(self.case, self.root, client, "agent", 6)
        self.assertEqual(row["requests"], 1)
        self.assertEqual(row["model_calls"], 2)
        self.assertTrue(row["grounded_correct"])
        self.assertEqual(row["remaining_budget"], 3)
        self.assertEqual(len(client.observations[0]["images"]), 1)
        self.assertEqual(len(client.observations[1]["images"]), 2)

    def test_unavailable_cannot_be_cited_as_seen(self):
        self.case["evidence"][0]["available"] = False
        row = episode(self.case, self.root, PixelClient(), "fixed_request", 6)
        self.assertEqual(row["request_status"], "unable_to_provide")
        self.assertFalse(row["coverage"])
        self.assertFalse(row["correct"])
        self.assertFalse(row["unsupported_decision"])

    def test_no_request_baseline_and_privileged_oracle(self):
        initial = episode(self.case, self.root, PixelClient(), "initial", 6)
        oracle = episode(self.case, self.root, PixelClient(), "oracle", 6)
        self.assertEqual(initial["requests"], 0)
        self.assertFalse(initial["correct"])
        self.assertTrue(oracle["correct"])
        self.assertEqual(oracle["requests"], 0)

    def test_strict_model_response_parsing(self):
        with self.assertRaises(ValueError):
            parse_decision("no json")
        with self.assertRaises(ValueError):
            parse_decision('{"verdict":"Supported","image_ids":[],"request_original":"yes","confidence":0.8}')
