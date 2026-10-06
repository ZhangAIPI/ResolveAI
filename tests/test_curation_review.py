"""Evidence review must distinguish availability, visual uncertainty and source truth."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from resolveai.curation import identity_families, variants
from resolveai.environment import Environment
from resolveai.review import merge_votes, require_admitted
from resolveai.vlm_worker import parse_tool_call


class NoOCR:
    available = False


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ["a", "b", "b-side", "b-preview"]:
            Image.new("RGB", (32, 32), "white").save(self.root / (name + ".png"))
        evidence = [{"id": name, "source_id": name, "path": name + ".png",
                     "object": "subject-A" if name == "a" else "subject-B",
                     "party": "submitted", "time": "capture", "available": True,
                     "view": "view-01" if name in {"a", "b-side"} else "view-02",
                     "target_bbox": [2, 2, 28, 28]}
                    for name in ["a", "b-side", "b", "b-preview"]]
        self.endpoint = lambda name: {"image_id": name, "bbox": [2, 2, 28, 28], "time": "capture"}
        edge = lambda name: {"relation": "same_object", "left": self.endpoint("a"), "right": self.endpoint(name)}
        self.annotation = {"protocol": "evidence-chain-v1", "verdict": "Supported",
                           "subclaims": [{"id": "identity", "kind": "identity", "truth": "Supported",
                                         "minimal_evidence_sets": {"Supported": [[edge("b")], [edge("b-side")]]}}]}
        self.family = list(variants({"family_id": "f", "asset_root": str(self.root),
            "initial": ["a", "b-preview"], "evidence": evidence, "claim": "The two targets are the same item.",
            "annotation": self.annotation,
            "request_options": {"objects": ["subject-A", "subject-B"], "times": ["capture"],
                                "views": ["view-01", "view-02"]}}, "b"))
        self.vote = {"protocol": "visual-review-v1", "dataset_sha256": "d"*64,
            "family_id": "f", "reviewer_type": "human", "reviewer_id": "reviewer-one",
            "claim_clear": True, "alternatives_checked": True, "reason": "Matching unique marks in alternative views.",
            "initial_verdicts": {v: "Supported" if v == "Sufficient" else "Need more evidence" for v in
                                 ["Sufficient", "Obtainable", "Missing", "Unavailable"]},
            "pool_verdicts": {v: "Supported" for v in ["Sufficient", "Obtainable", "Missing", "Unavailable"]},
            "annotation": deepcopy(self.annotation)}

    def tearDown(self):
        self.temp.cleanup()

    def votes(self):
        other = deepcopy(self.vote)
        other["reviewer_id"] = "reviewer-two"
        return [deepcopy(self.vote), other]

    def test_missing_designated_photo_does_not_erase_alternative_evidence(self):
        annotation = merge_votes(self.votes(), self.family)
        self.assertEqual(annotation["status"], "reviewed")
        missing = next(c for c in self.family if c["variant"] == "Missing")
        env = Environment(missing, self.root, ocr_backend=NoOCR())
        state = env._world.fingerprint
        response = env.step({"type": "request_photo", "query":
                             {"object": "subject-B", "time": "capture", "view": "view-01"}})
        self.assertEqual(response["status"], "provided")
        self.assertEqual(response["images"][0]["image_id"], "b-side")
        self.assertEqual(state, env._world.fingerprint)

    def test_votes_must_explain_preview_and_alternatives(self):
        votes = self.votes()
        for vote in votes:
            vote["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"] = [
                vote["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0]]
        with self.assertRaisesRegex(ValueError, "do not explain"):
            merge_votes(votes, self.family)

    def test_independent_review_and_disagreement_are_not_fabricated(self):
        for votes in [[self.vote], [self.vote, self.vote]]:
            with self.assertRaises(ValueError):
                merge_votes(votes, self.family)
        votes = self.votes()
        votes[1]["pool_verdicts"]["Missing"] = "Need more evidence"
        with self.assertRaisesRegex(ValueError, "disagreement"):
            merge_votes(votes, self.family)
        votes = self.votes()
        votes[1]["reviewer_type"] = "model"
        with self.assertRaises(ValueError):
            merge_votes(votes, self.family)

    def test_oversized_regions_and_hidden_assets_cannot_be_promoted(self):
        for field, value in [("image_id", "hidden"), ("bbox", [0, 0, 100, 100])]:
            votes = self.votes()
            for vote in votes:
                vote["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0][0]["right"][field] = value
            with self.assertRaises(ValueError):
                merge_votes(votes, self.family)

    def test_formal_evaluation_gate_and_public_target_separation(self):
        with self.assertRaises(ValueError):
            require_admitted({"protocol": "benchmark-v0.4", "formal_evaluation_ready": False}, self.family)
        cases = deepcopy(self.family)
        annotation = merge_votes(self.votes(), cases)
        for case in cases:
            case["annotation"] = deepcopy(annotation)
            case["review"] = {"protocol": "visual-review-v1", "reviewers": ["one", "two"],
                              "initial_verdict": self.vote["initial_verdicts"][case["variant"]],
                              "pool_verdict": self.vote["pool_verdicts"][case["variant"]]}
        require_admitted({"protocol": "benchmark-v0.4", "formal_evaluation_ready": True}, cases)
        observation = Environment(cases[1], self.root, ocr_backend=NoOCR()).observation()
        self.assertEqual(observation["images"][0]["target_bbox"], [2, 2, 28, 28])
        self.assertNotIn("annotation", observation)
        self.assertNotIn("review", observation)

    def test_plain_json_is_accepted_without_inventing_missing_arguments(self):
        raw = '{"name":"inspect","arguments":{"image_id":"a"}}'
        call, text = parse_tool_call(raw, allow_bare=True)
        self.assertEqual(call, {"name": "inspect", "arguments": {"image_id": "a"}})
        self.assertEqual(text, "")
        for raw in ['{"name":"inspect"}', '{"name":"inspect","arguments":{}} trailing',
                    '{"name":"inspect","arguments":{}} {"name":"finish","arguments":{}}']:
            with self.assertRaises(ValueError):
                parse_tool_call(raw, allow_bare=True)

    def test_same_category_other_sequence_is_unknown_not_negative(self):
        co3d = self.root / "co3d"
        co3d.mkdir()
        cases = []
        for category in ["chair", "cup"]:
            for seq in ["one", "two"]:
                ids = [category + seq + str(i) for i in range(4)]
                rows = [{"id": image_id, "source_id": image_id, "path": image_id + ".png",
                         "party": "submitted", "object": "item", "time": "capture",
                         "view": f"view-{i+1:02d}", "available": True} for i, image_id in enumerate(ids)]
                rows.append({**rows[-1], "id": ids[-1] + "-preview", "view": "preview"})
                edge = {"relation": "same_object", "left": self.endpoint(ids[0]), "right": self.endpoint(ids[-1])}
                cases.append({"family_id": category+seq, "variant": "Obtainable", "category": category,
                    "evidence": rows, "scene": {"source_sequence": seq+category},
                    "annotation": {"subclaims": [{"minimal_evidence_sets": {"Supported": [[edge]]}}]}})
        (co3d / "cases.json").write_text(json.dumps(cases))
        families, excluded = identity_families(co3d, self.root)
        unknown = [f for f, _ in families if f["provenance"]["construction"] == "same-category-candidate"]
        self.assertEqual(len(unknown), 4)
        self.assertTrue(all(f["provenance_truth"] is None for f in unknown))
        self.assertTrue(all(f["annotation"]["subclaims"][0]["truth"] == "uncertain" for f in unknown))
        self.assertTrue(all(len(f["evidence"]) == 5 for f, _ in families))


if __name__ == "__main__":
    unittest.main()
