"""Request controls must use public choices and real, budgeted tool results."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO, StringIO
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

from scripts.evaluate_evidence_controls import (
    ARMS, controlled_episode, request_schedule, select_cases, validate_records, main,
)
from scripts.evaluate_study import Records
from resolveai.conversation import Conversation
from resolveai.environment import Environment


class NoOCR:
    available = False


class Client:
    ready = {"gpu": "test-no-GPU"}

    def close(self):
        pass

    def __init__(self):
        self.payloads = []

    def ask_conversation(self, payload, max_context_tokens=8192):
        self.payloads.append(deepcopy(payload))
        return {"parse_error": False, "raw": "", "text": "",
                "tool_call": {"name": "finish", "arguments": {
                    "verdict": "Need more evidence", "citations": [], "links": []}},
                "output_tokens": 7}


class EvidenceControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name, size, color in [("preview", 16, "red"), ("original", 64, "blue")]:
            Image.new("RGB", (size, size), color).save(self.root / (name + ".png"))
        self.case = {
            "case_id": "f-Obtainable", "family_id": "f", "task": "state",
            "variant": "Obtainable", "claim": "The cup is cracked.",
            "official_split": "test", "provenance_truth": "Supported",
            "initial": ["preview"],
            "request_options": {"objects": ["cup"], "times": ["capture"],
                                "views": ["overview", "original"]},
            "evidence": [{
                "id": name, "source_id": "source", "party": "submitted",
                "path": name + ".png", "object": "cup", "time": "capture",
                "view": view, "available": True,
            } for name, view in [("preview", "overview"), ("original", "original")]],
            "annotation": {"verdict": "Supported",
                           "minimal_evidence_sets": {"Supported": [["original"]]}},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def env(self, case=None, budget=12):
        return Environment(case or self.case, self.root, budget, ocr_backend=NoOCR())

    def test_schedule_is_unchanged_by_private_labels_or_evidence_order(self):
        first = self.env()
        changed = deepcopy(self.case)
        changed["provenance_truth"] = "Refuted"
        changed["annotation"] = {"verdict": "Refuted",
                                "minimal_evidence_sets": {"Refuted": [["preview"]]}}
        changed["evidence"].reverse()
        second = self.env(changed)
        for arm in ARMS:
            self.assertEqual(
                request_schedule(Conversation(first).public(), arm, 11, "f"),
                request_schedule(Conversation(second).public(), arm, 11, "f"))

    def test_original_uses_only_publicly_offered_queries(self):
        public = Conversation(self.env()).public()
        expected = [{"name": "request_photo", "arguments": {
            "query": {"object": "cup", "time": "capture", "view": "original"}}}]
        self.assertEqual(request_schedule(public, "fixed_original", 1, "f"), expected)
        changed = deepcopy(self.case)
        changed["request_options"]["views"] = ["view-01", "view-02"]
        self.assertEqual(request_schedule(Conversation(self.env(changed)).public(),
                                         "fixed_original", 1, "f"), [])

    def test_random_order_is_seeded_and_preserves_the_public_query_pool(self):
        case = deepcopy(self.case)
        case["request_options"]["objects"] = ["subject-A", "subject-B"]
        case["request_options"]["views"] = ["view-01", "view-02", "view-03"]
        public = Conversation(self.env(case)).public()
        fixed = request_schedule(public, "fixed_order", 42, "f")
        a = request_schedule(public, "random_order", 42, "f")
        b = request_schedule(public, "random_order", 42, "f")
        self.assertEqual(a, b)
        self.assertEqual(sorted(map(json.dumps, fixed)), sorted(map(json.dumps, a)))
        self.assertNotEqual(a, request_schedule(public, "random_order", 43, "f"))
        for variant in ("Sufficient", "Obtainable", "Missing", "Unavailable"):
            self.assertEqual(a, request_schedule(public, "random_order", 42, "f-" + variant))

    def test_prefix_spends_real_budget_and_reserves_a_model_turn(self):
        client = Client()
        env = self.env(budget=3)
        trace = controlled_episode(env, client, "fixed_order", max_turns=8)
        self.assertEqual(env.requests, 1)
        self.assertEqual(env.request_cost, 3)
        self.assertEqual(env.budget, 0)
        self.assertEqual(trace["metadata"]["turns"], 2)
        self.assertEqual(trace["scripted_prefix"]["assistant_turns"], 1)
        self.assertEqual(len(trace["generation"]), 1)
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(client.payloads[0]["max_new_tokens"], 384)
        self.assertEqual([x["function"]["name"] for x in client.payloads[0]["tools"]],
                         ["finish"])
        self.assertEqual([m["name"] for m in client.payloads[0]["messages"]
                          if m["role"] == "tool"], ["request_photo"])
        env = self.env()
        trace = controlled_episode(env, Client(), "fixed_order", max_turns=2)
        self.assertEqual(env.requests, 1)
        self.assertEqual(trace["metadata"]["turns"], 2)

    def test_finish_only_never_releases_the_original(self):
        env, client = self.env(), Client()
        trace = controlled_episode(env, client, "finish_only")
        self.assertEqual(env._released, {"preview"})
        self.assertEqual(env.requests, 0)
        self.assertEqual(trace["scripted_prefix"]["executed_calls"], [])
        pixels = [b for m in client.payloads[0]["messages"]
                  if isinstance(m["content"], list) for b in m["content"]
                  if b["type"] == "image"]
        self.assertEqual(len(pixels), 1)
        image = Image.open(BytesIO(base64.b64decode(pixels[0]["image_png"])))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_requested_pixels_are_actual_files_and_the_world_is_immutable(self):
        for arm in ARMS:
            env = self.env()
            fingerprint = env._world.fingerprint
            before = {p.name: p.read_bytes() for p in self.root.glob("*.png")}
            client = Client()
            trace = controlled_episode(env, client, arm, case_id=self.case["case_id"])
            self.assertEqual(env._world.fingerprint, fingerprint)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.glob("*.png")})
            if arm == "fixed_original":
                tool = next(m for m in client.payloads[0]["messages"] if m["role"] == "tool")
                pixels = next(b["image_png"] for b in tool["content"] if b["type"] == "image")
                image = Image.open(BytesIO(base64.b64decode(pixels)))
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(trace["scripted_prefix"]["executed_calls"][0]["image_ids"],
                                 ["original"])

    def test_unavailable_original_does_not_inject_oracle_pixels(self):
        case = deepcopy(self.case)
        next(e for e in case["evidence"] if e["id"] == "original")["available"] = False
        env, client = self.env(case), Client()
        trace = controlled_episode(env, client, "fixed_original")
        self.assertEqual(env._released, {"preview"})
        result = trace["scripted_prefix"]["executed_calls"][0]
        self.assertEqual(result["status"], "unable_to_provide")
        self.assertEqual(result["image_ids"], [])
        tools = [m for m in client.payloads[0]["messages"] if m["role"] == "tool"]
        self.assertFalse(any(b["type"] == "image" for m in tools for b in m["content"]))

    def test_cohort_requires_exact_family_version_coverage_and_dataset_hash(self):
        cases = [dict(self.case, variant=v, case_id="f-" + v)
                 for v in ("Sufficient", "Obtainable", "Missing", "Unavailable")]
        raw = json.dumps(cases).encode()
        digest = hashlib.sha256(raw).hexdigest()
        plan = {"dataset_sha256": digest, "family_ids": ["f", "other"]}
        cohort = {"dataset_sha256": digest, "family_ids": ["f"]}
        family, selected = select_cases(plan, cohort, raw, 1, 0,
                                       ["Sufficient", "Obtainable", "Missing", "Unavailable"])
        self.assertEqual(family, ["f"])
        self.assertEqual(len(selected), 4)
        with self.assertRaises(ValueError):
            select_cases(plan, cohort, raw + b" ", 1, 0, ["Sufficient"])
        with self.assertRaises(ValueError):
            select_cases(plan, cohort, raw, 1, 0, ["Nonexistent"])

    def test_control_storage_resumes_and_rejects_unexpected_case_keys(self):
        output = self.root / "output"
        output.mkdir()
        arm = "fixed_original"
        records = Records(output, arms=ARMS)
        trace = {"messages": [], "metadata": {"case_id": self.case["case_id"], "arm": arm}}
        row = {
            "case_id": self.case["case_id"], "family_id": "f",
            "variant": "Obtainable", "arm": arm, "termination": "finished",
            "errors": 0, "requests": 1, "source_label": "Supported",
            "source_label_match": False,
        }
        records.save(row, trace)
        validate_records(records.db, [self.case], [arm], complete=True)
        with self.assertRaises(ValueError):
            validate_records(records.db, [self.case], ["finish_only"], complete=True)
        records.db.close()
        records = Records(output, arms=ARMS)
        self.assertTrue(records.has(self.case["case_id"], arm))
        self.assertEqual(records.summary()["episodes"], 1)
        records.db.close()


    def test_remaining_finish_retries_do_not_exceed_the_global_turn_limit(self):
        class InvalidClient(Client):
            def ask_conversation(self, payload, max_context_tokens=8192):
                self.payloads.append(deepcopy(payload))
                return {"parse_error": True, "raw": "invalid", "output_tokens": 1}
        env, client = self.env(), InvalidClient()
        trace = controlled_episode(env, client, "fixed_original", max_turns=8)
        self.assertEqual(trace["metadata"]["turns"], 8)
        self.assertEqual(trace["scripted_prefix"]["assistant_turns"], 1)
        self.assertEqual(len(trace["generation"]), 7)
        self.assertEqual(trace["termination"], "max_turns")
        self.assertIsNone(trace["decision"])

    def test_runner_persists_all_expected_keys_and_resumes_without_loading_a_model(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        cases = []
        for variant in ("Sufficient", "Obtainable", "Missing", "Unavailable"):
            case = dict(deepcopy(self.case), variant=variant, case_id="f-" + variant,
                        asset_root=str(self.root))
            if variant in ("Missing", "Unavailable"):
                next(e for e in case["evidence"] if e["id"] == "original")["available"] = False
            cases.append(case)
        payload = json.dumps(cases).encode()
        (self.root / "cases.json").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        plan = {"data": str(self.root), "dataset_sha256": digest,
                "family_ids": ["f"], "budget": 12, "max_turns": 8}
        cohort = {"dataset_sha256": digest, "family_ids": ["f"]}
        plan_path, cohort_path = self.root / "plan.json", self.root / "cohort.json"
        plan_path.write_text(json.dumps(plan))
        cohort_path.write_text(json.dumps(cohort))
        output = self.root / "evaluation"
        grounding = self.root / "grounding"
        grounding.mkdir()
        (grounding / "manifest.json").write_text('{"test":true}')
        argv = ["controls", str(plan_path), str(model), str(output),
                "--cohort", str(cohort_path), "--model-id", "test/model",
                "--revision", "0" * 40, "--grounding-root", str(grounding)]
        target = "scripts.evaluate_evidence_controls."
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test"}), \
             patch("sys.argv", argv), patch(target + "RapidOCRBackend", NoOCR), \
             patch(target + "FrozenGrounding",
                   return_value=SimpleNamespace(available=lambda kind: True)) as backend, \
             patch(target + "ModelClient", return_value=Client()) as factory, \
             redirect_stdout(StringIO()):
            main()
            factory.assert_called_once()
            backend.assert_called_once_with(grounding, device="cuda:0")
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["episodes"], 20)
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test"}), \
             patch("sys.argv", argv), patch(target + "ModelClient") as factory, \
             redirect_stdout(StringIO()):
            main()
            factory.assert_not_called()
        records = Records(output, arms=ARMS)
        validate_records(records.db, cases, ARMS, complete=True)
        records.db.close()


    def test_adaptive_keeps_original_prompt_and_all_tools_without_scripted_actions(self):
        env = Environment(self.case, self.root, 3, ocr_backend=NoOCR(),
                          grounding_backend=SimpleNamespace(available=lambda kind: True))
        client = Client()
        original = Conversation(env)
        trace = controlled_episode(env, client, "adaptive", max_turns=8)
        payload = client.payloads[0]
        self.assertEqual(payload["tools"], original.tools)
        self.assertIn("ground_image_to_image",
                      {tool["function"]["name"] for tool in payload["tools"]})
        self.assertTrue(payload["messages"][0]["content"].startswith(
            original.messages[0]["content"]))
        self.assertIn("Episode limit: 8 assistant turns", payload["messages"][0]["content"])
        self.assertEqual(payload["max_new_tokens"], 384)
        self.assertEqual(trace["scripted_prefix"]["assistant_turns"], 0)
        self.assertEqual(env.budget, 3)

    def test_adaptive_executed_requests_spend_the_same_real_budget(self):
        class RequestThenFinish(Client):
            def ask_conversation(self, payload, max_context_tokens=8192):
                if not self.payloads:
                    self.payloads.append(deepcopy(payload))
                    return {"parse_error": False, "raw": "", "text": "",
                            "tool_call": {"name": "request_photo", "arguments": {
                                "query": {"object": "cup", "time": "capture",
                                          "view": "original"}}},
                            "output_tokens": 10}
                return super().ask_conversation(payload, max_context_tokens)
        env, client = self.env(budget=3), RequestThenFinish()
        trace = controlled_episode(env, client, "adaptive", max_turns=8)
        self.assertEqual(env.budget, 0)
        self.assertEqual(env.requests, 1)
        self.assertEqual(trace["metadata"]["turns"], 2)
        self.assertEqual(trace["scripted_prefix"]["assistant_turns"], 0)
        self.assertEqual(env._released, {"preview", "original"})
        self.assertEqual(len(trace["generation"]), 2)


if __name__ == "__main__":
    unittest.main()
