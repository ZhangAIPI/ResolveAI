"""CPU-only reviewer-control tests; loads the isolated candidate, never a GPU."""
import base64
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.tools import ActionError

path = Path(__file__).with_name("evaluate_evidence_controls.py")
if not path.exists():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_evidence_controls.py"
spec = importlib.util.spec_from_file_location("reviewer_staged_controls", path)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


class NoOCR:
    available = False


class Client:
    ready = {"gpu": "mock-no-GPU"}

    def __init__(self):
        self.payloads = []

    def close(self):
        pass

    def ask_conversation(self, payload, max_context_tokens=8192):
        self.payloads.append(deepcopy(payload))
        return {"parse_error": False, "raw": "", "text": "", "output_tokens": 7,
                "tool_call": {"name": "finish", "arguments": {
                    "verdict": "Need more evidence", "citations": [], "links": []}}}


class RequestThenFinish(Client):
    def ask_conversation(self, payload, max_context_tokens=8192):
        if not self.payloads:
            self.payloads.append(deepcopy(payload))
            return {"parse_error": False, "raw": "", "text": "", "output_tokens": 10,
                    "tool_call": {"name": "request_photo", "arguments": {
                        "query": {"object": "cup", "time": "capture", "view": "original"}}}}
        return super().ask_conversation(payload, max_context_tokens)


class ReviewerInterventionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name, size, color in [("preview", 16, "red"), ("original", 64, "blue"),
                                   ("reference", 24, "green")]:
            Image.new("RGB", (size, size), color).save(self.root / (name + ".png"))
        self.case = {
            "case_id": "f-Obtainable", "family_id": "f", "task": "state",
            "variant": "Obtainable", "claim": "The cup is cracked.",
            "official_split": "test", "asset_root": str(self.root),
            "provenance_truth": "Supported", "initial": ["reference", "preview"],
            "request_options": {"objects": ["cup"], "times": ["capture"],
                                "views": ["overview", "original"]},
            "evidence": [{
                "id": name, "source_id": name, "party": "submitted",
                "path": name + ".png", "object": obj, "time": "capture",
                "view": view, "available": True,
            } for name, view, obj in [("preview", "overview", "cup"),
                                      ("original", "original", "cup"),
                                      ("reference", "reference", "normal-reference")]],
            "annotation": {"verdict": "Supported", "audit_secret": "PRIVATE_REVIEW_SENTINEL",
                           "minimal_evidence_sets": {"Supported": [["reference", "original"]]}},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def env(self, masked=False, budget=3, case=None):
        cls = candidate.NoNewPhotoEnvironment if masked else Environment
        return cls(case or self.case, self.root, budget, ocr_backend=NoOCR())

    def test_mask_preserves_initial_public_prompt_tools_and_physics(self):
        left, right = self.env(), self.env(masked=True)
        self.assertEqual(Conversation(left).public(), Conversation(right).public())
        self.assertEqual(left._world.fingerprint, right._world.fingerprint)
        a, b = Client(), Client()
        candidate.controlled_episode(left, a, "adaptive")
        candidate.controlled_episode(right, b, "adaptive_no_new_photo")
        self.assertEqual(a.payloads[0], b.payloads[0])
        self.assertEqual(a.payloads[0]["max_new_tokens"], 384)

    def test_mask_denies_new_record_with_normal_cost_and_failure_response(self):
        env = self.env(masked=True)
        before = env._world.fingerprint
        out = env.step({"type": "request_photo", "query": {
            "object": "cup", "time": "capture", "view": "original"}})
        self.assertEqual(out, {"status": "unable_to_provide", "images": [],
                               "budget": 0, "finished": False})
        self.assertEqual(env._released, {"preview", "reference"})
        self.assertEqual((env.calls, env.requests, env.request_cost), (1, 1, 3))
        self.assertEqual(env._world.fingerprint, before)

    def test_mask_still_returns_actual_repeated_preview(self):
        env = self.env(masked=True)
        out = env.step({"type": "request_photo", "query": {
            "object": "cup", "time": "capture", "view": "overview"}})
        self.assertEqual(out["status"], "provided")
        self.assertEqual(out["images"][0]["image_id"], "preview")
        image = Image.open(BytesIO(out["images"][0]["image_png"]))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(env.budget, 0)

    def test_unaffordable_request_does_not_spend_resources(self):
        env = self.env(masked=True, budget=2)
        with self.assertRaises(ActionError):
            env.step({"type": "request_photo", "query": {
                "object": "cup", "time": "capture", "view": "original"}})
        self.assertEqual((env.budget, env.calls, env.requests), (2, 0, 0))

    def test_adaptive_mask_pair_uses_actual_feedback_and_shared_turn_limit(self):
        normal, masked = self.env(), self.env(masked=True)
        a, b = RequestThenFinish(), RequestThenFinish()
        before = masked._world.fingerprint
        left = candidate.controlled_episode(normal, a, "adaptive", max_turns=8)
        right = candidate.controlled_episode(masked, b, "adaptive_no_new_photo", max_turns=8)
        self.assertEqual(a.payloads[0], b.payloads[0])
        self.assertEqual(left["metadata"]["turns"], 2)
        self.assertEqual(right["metadata"]["turns"], 2)
        self.assertEqual(normal.budget, masked.budget)
        self.assertIn("original", normal._released)
        self.assertNotIn("original", masked._released)
        self.assertEqual(masked._world.fingerprint, before)

    def test_provisional_supported_conjunction_combines_all_parts(self):
        annotation = {"verdict": "Supported", "subclaims": [
            {"minimal_evidence_sets": {"Supported": [["a"], ["b"]]}},
            {"minimal_evidence_sets": {"Supported": [["c"]]}}
        ]}
        self.assertEqual(set(candidate.annotation_image_sets(annotation)),
                         {frozenset({"a", "c"}), frozenset({"b", "c"})})

    def test_provisional_refutation_accepts_one_annotated_counterexample(self):
        annotation = {"verdict": "Refuted", "subclaims": [
            {"minimal_evidence_sets": {"Supported": [["a"]]}},
            {"minimal_evidence_sets": {"Refuted": [[{
                "relation": "different_object",
                "left": {"image_id": "x", "bbox": [0, 0, 2, 2], "time": "capture"},
                "right": {"image_id": "y", "bbox": [0, 0, 2, 2], "time": "capture"},
            }]]}}
        ]}
        self.assertEqual(candidate.annotation_image_sets(annotation), [frozenset({"x", "y"})])

    def test_annotation_controller_selects_shortest_real_public_request_path(self):
        env = self.env()
        initial = set(env._released)
        path, metadata = candidate.annotation_oracle_schedule(env, Conversation(env).public(), 8)
        self.assertEqual(path, [{"name": "request_photo", "arguments": {
            "query": {"object": "cup", "time": "capture", "view": "original"}}}])
        self.assertEqual(metadata["minimum_request_count"], 1)
        self.assertFalse(metadata["pixel_sufficiency_reviewed"])
        self.assertEqual(env._released, initial, "BFS must not execute an environment call")
        self.assertEqual(env.budget, 3)

    def test_annotation_controller_respects_budget_and_finish_turn(self):
        for budget, turns in [(2, 8), (3, 1)]:
            env = self.env(budget=budget)
            calls, info = candidate.annotation_oracle_schedule(env, Conversation(env).public(), turns)
            self.assertEqual(calls, [])
            self.assertFalse(info["annotation_coverage_reachable"])

    def test_annotation_reference_never_supplies_unavailable_or_fabricated_pixels(self):
        case = deepcopy(self.case)
        next(e for e in case["evidence"] if e["id"] == "original")["available"] = False
        env, client = self.env(case=case), Client()
        trace = candidate.controlled_episode(env, client, "annotation_oracle")
        self.assertFalse(trace["annotation_oracle"]["annotation_coverage_reachable"])
        self.assertEqual(env._released, {"preview", "reference"})
        self.assertEqual(env.requests, 0)
        self.assertNotIn("PRIVATE_REVIEW_SENTINEL", json.dumps(client.payloads))
        self.assertNotIn("target_record_sets", json.dumps(client.payloads))

    def test_annotation_controller_returns_actual_original_and_private_metadata_stays_off_prompt(self):
        env, client = self.env(), Client()
        trace = candidate.controlled_episode(env, client, "annotation_oracle")
        self.assertTrue(trace["annotation_oracle"]["annotation_coverage_reachable"])
        self.assertEqual(trace["metadata"]["turns"], 2)
        self.assertEqual(env.request_cost, 3)
        self.assertEqual(env.budget, 0)
        self.assertNotIn("PRIVATE_REVIEW_SENTINEL", json.dumps(client.payloads))
        self.assertNotIn("annotation_coverage_reachable", json.dumps(client.payloads))
        tool = next(m for m in client.payloads[0]["messages"] if m["role"] == "tool")
        pixels = next(b["image_png"] for b in tool["content"] if b["type"] == "image")
        self.assertEqual(Image.open(BytesIO(base64.b64decode(pixels))).getpixel((0, 0)),
                         (0, 0, 255))

    def test_full_available_finish_is_privileged_access_and_keeps_native_finish_schema(self):
        env, client = self.env(), Client()
        fingerprint = env._world.fingerprint
        candidate.controlled_episode(env, client, "full_available_finish")
        self.assertEqual(env._released, {"preview", "reference", "original"})
        self.assertEqual(env.requests, 0)
        self.assertEqual(env.budget, 3)
        self.assertEqual(env._world.fingerprint, fingerprint)
        self.assertEqual([t["function"]["name"] for t in client.payloads[0]["tools"]], ["finish"])

    def test_staged_runner_writes_new_arm_matrix_and_resumes_without_inference(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        variants = ("Sufficient", "Obtainable", "Missing", "Unavailable")
        cases = [dict(deepcopy(self.case), variant=v, case_id="f-" + v) for v in variants]
        raw = json.dumps(cases).encode()
        (self.root / "cases.json").write_bytes(raw)
        plan = {"data": str(self.root), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "family_ids": ["f"], "budget": 3, "max_turns": 8}
        cohort = {"dataset_sha256": plan["dataset_sha256"], "family_ids": ["f"]}
        (self.root / "plan.json").write_text(json.dumps(plan))
        (self.root / "cohort.json").write_text(json.dumps(cohort))
        ground = self.root / "grounding"
        ground.mkdir()
        (ground / "manifest.json").write_text('{"test":true}')
        output = self.root / "evaluation"
        argv = ["staged", str(self.root / "plan.json"), str(model), str(output),
                "--cohort", str(self.root / "cohort.json"), "--model-id", "mock/model",
                "--revision", "0" * 40, "--grounding-root", str(ground)]
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test-no-GPU"}), \
             patch("sys.argv", argv), patch.object(candidate, "RapidOCRBackend", NoOCR), \
             patch.object(candidate, "FrozenGrounding",
                          return_value=SimpleNamespace(available=lambda _: False)), \
             patch.object(candidate, "ModelClient", return_value=Client()), redirect_stdout(StringIO()):
            candidate.main()
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["episodes"], 16)
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test-no-GPU"}), \
             patch("sys.argv", argv), patch.object(candidate, "ModelClient") as factory, \
             redirect_stdout(StringIO()):
            candidate.main()
            factory.assert_not_called()


    def test_restore_preview_state_uses_public_original_option(self):
        case = deepcopy(self.case)
        for e in case["evidence"]:
            if e["id"] == "preview":
                e["source_size"] = [64, 64]
                e["source_bbox"] = [0, 0, 64, 64]
        env = self.env(case=case)
        call = candidate.request_schedule(Conversation(env).public(), "restore_preview", 1, "f")
        self.assertEqual(call, [{"name": "request_photo", "arguments": {
            "query": {"object": "cup", "time": "capture", "view": "original"}}}])

    def test_restore_preview_identity_uses_public_camera_and_slot_without_original_option(self):
        case = deepcopy(self.case)
        case["task"] = "identity"
        case["request_options"] = {"objects": ["subject-A", "subject-B"],
                                   "times": ["capture"], "views": ["view-01", "view-02"]}
        for e in case["evidence"]:
            e["object"] = "subject-A" if e["id"] == "reference" else "subject-B"
            e["view"] = "view-01" if e["id"] == "reference" else "view-02"
            if e["id"] == "preview":
                e.update(view="preview", camera_view="view-02",
                         source_size=[64, 64], source_bbox=[0, 0, 64, 64])
        env, client = self.env(case=case), Client()
        schedule = candidate.request_schedule(Conversation(env).public(), "restore_preview", 1, "f")
        self.assertEqual(schedule, [{"name": "request_photo", "arguments": {
            "query": {"object": "subject-B", "time": "capture", "view": "view-02"}}}])
        trace = candidate.controlled_episode(env, client, "restore_preview")
        self.assertEqual(trace["scripted_prefix"]["executed_calls"][0]["image_ids"], ["original"])
        self.assertEqual(env.request_cost, 3)
        self.assertNotIn("PRIVATE_REVIEW_SENTINEL", json.dumps(client.payloads))

    def test_restore_preview_schedule_is_independent_of_private_annotations_and_labels(self):
        case = deepcopy(self.case)
        for e in case["evidence"]:
            if e["id"] == "preview":
                e.update(source_size=[64, 64], source_bbox=[0, 0, 64, 64])
        changed = deepcopy(case)
        changed["provenance_truth"] = "Refuted"
        changed["annotation"] = {"verdict": "Refuted",
                                "minimal_evidence_sets": {"Refuted": [["reference"]]}}
        a = candidate.request_schedule(Conversation(self.env(case=case)).public(), "restore_preview", 1, "f")
        b = candidate.request_schedule(Conversation(self.env(case=changed)).public(), "restore_preview", 1, "f")
        self.assertEqual(a, b)

    def test_restore_preview_skips_full_resolution_initial_evidence(self):
        case = dict(deepcopy(self.case), initial=["reference", "original"])
        calls = candidate.request_schedule(Conversation(self.env(case=case)).public(), "restore_preview", 1, "f")
        self.assertEqual(calls, [])

    def test_wrong_environment_cannot_silently_run_no_new_photo_arm(self):
        with self.assertRaisesRegex(ValueError, "restricted environment"):
            candidate.controlled_episode(self.env(), Client(), "adaptive_no_new_photo")


if __name__ == "__main__":
    unittest.main()
