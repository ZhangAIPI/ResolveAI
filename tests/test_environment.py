"""Software fixtures only; these pixels are not research examples."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from resolveai.environment import Environment
from resolveai.splits import split_families


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i in ("front", "side", "blocked"):
            Image.new("RGB", (16, 16), "gray").save(self.root / (i + ".png"))
        self.case = {"claim": "The table is scratched", "initial": ["front"],
            "evidence": [{"id": i, "path": i + ".png", "source_id": i,
                "party": "A", "object": "table", "time": "after", "view": i,
                "available": i != "blocked"} for i in ("front", "side", "blocked")],
            "annotation": {"verdict": "Supported", "minimal_evidence_sets":
                {"Supported": [["side"]], "Refuted": [["front"]]}}}
        self.env = Environment(self.case, self.root)

    def request(self, view):
        return {"type": "request_photo", "query":
                {"object": "table", "time": "after", "view": view}}

    def test_no_oracle_and_unreleased_rejection(self):
        obs = self.env.observation()
        self.assertNotIn("annotation", obs)
        self.assertEqual(set(obs["images"][0]), {"image_id", "source_id", "party",
            "time", "source_bbox", "source_size", "image_png", "view_id", "display_size", "object", "camera_view"})
        with self.assertRaises(ValueError):
            self.env.step({"type": "inspect", "image_id": "side"})
        self.assertEqual(self.env.budget, 12)

    def test_fork_request_and_budget(self):
        branch = self.env.fork()
        result = branch.step(self.request("side"))
        self.assertEqual(result["images"][0]["image_id"], "side")
        self.assertEqual(branch.budget, 9)
        self.assertEqual(len(self.env.observation()["images"]), 1)
        self.assertEqual(branch.requests, 1)

    def test_missing_and_unavailable_do_not_reveal_existence(self):
        self.assertEqual(self.env.step(self.request("blocked"))["status"],
                         self.env.step(self.request("absent"))["status"])

    def test_crop_provenance_and_atomic_validation(self):
        result = self.env.step({"type": "crop", "image_id": "front", "bbox": [2, 3, 8, 9]})
        self.assertEqual(result["images"][0]["source_bbox"], [2, 3, 8, 9])
        budget = self.env.budget
        with self.assertRaises(ValueError):
            self.env.step({"type": "crop", "image_id": "front", "bbox": [-1, 0, 9, 9]})
        self.assertEqual(self.env.budget, budget)

    def test_grounded_decision_and_decisive_refutation(self):
        citation = {"image_id": "front", "bbox": [0, 0, 16, 16], "time": "after"}
        decision = {"type": "finish", "verdict": "Supported", "citations": [citation]}
        self.assertTrue(self.env.evaluate(decision)["unsupported_decision"])
        self.env.step(self.request("side"))
        decision["citations"] = [{"image_id": "side", "bbox": [0, 0, 16, 16], "time": "after"}]
        self.assertTrue(self.env.evaluate(decision)["grounded_correct"])
        self.case["annotation"]["verdict"] = "Refuted"
        env = Environment(self.case, self.root)
        self.assertTrue(env.evaluate({"verdict": "Refuted", "citations": [citation]})["grounded_correct"])

    def test_path_escape_and_exhausted_budget(self):
        self.case["evidence"][0]["path"] = "../outside.png"
        with self.assertRaises(ValueError):
            Environment(self.case, self.root).observation()
        env = Environment(self.env._case, self.root, budget=0)
        with self.assertRaises(ValueError):
            env.step(self.request("side"))

    def test_shared_asset_component_stays_in_one_split(self):
        families = [{"family_id": "a", "group_ids": ["object-1"]},
                    {"family_id": "b", "group_ids": ["object-1", "scene-2"]},
                    {"family_id": "c", "group_ids": ["scene-2"]}]
        splits = split_families(families)
        self.assertEqual(len(set(splits.values())), 1)
        self.assertEqual(splits, split_families(list(reversed(families))))


if __name__ == "__main__":
    unittest.main()
