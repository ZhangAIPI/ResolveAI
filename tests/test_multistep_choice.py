"""CPU tests for the interface-assisted finite-choice diagnostic."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image, ImageDraw

from scripts import evaluate_multistep_choice as runner
from resolveai.environment import Environment
from resolveai.multistep import observation_token


def score(case, decision, tokens):
    views = {token["view"] for token in tokens}
    sufficient = {"front_survey", "rear_left"} <= views
    correct = bool(decision and decision["verdict"] == "Supported")
    return {
        "source_label_correct": correct,
        "procedural_evidence_sufficient": sufficient,
        "chain_grounded_correct": correct and sufficient,
        "localized_grounded_correct": False,
        "simulator_grounded_correct": False,
        "unsupported_decision": bool(decision and decision["verdict"] != "Need more evidence"
                                     and not (correct and sufficient)),
    }


class FakeClient:
    def __init__(self, choices):
        self.choices = list(choices)
        self.inputs = []

    def ask_conversation(self, payload, max_context_tokens):
        self.inputs.append(deepcopy(payload))
        if (payload["mode"] != "choice" or payload["max_new_tokens"] != 1
                or payload["assistant_prefix"] is not False or max_context_tokens != 8192):
            raise AssertionError("Choice interface limits changed")
        letter = self.choices.pop(0)
        return {"choice": letter, "raw": letter, "input_tokens": 100, "output_tokens": 1}


class ChoiceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        evidence = []
        views = ("reference", "initial", *runner.native.REQUEST_VIEWS)
        for view in views:
            size = (180, 100) if view in ("front_survey", "initial", "original") else (80, 80)
            image = Image.new("RGB", size, (130, 130, 130))
            draw = ImageDraw.Draw(image)
            if view == "front_survey":
                for x in (10, 70, 130):
                    draw.rectangle((x, 25, x + 39, 84), fill=(120, 75, 35))
            elif view not in ("initial", "original"):
                draw.rectangle((10, 10, 69, 69), fill=(120, 75, 35))
            image.save(self.root / (view + ".png"))
            reference = view == "reference"
            evidence.append({
                "id": view, "path": view + ".png", "source_id": "public-" + view,
                "party": "A" if reference else "B",
                "object": "reference crate" if reference else "inspection scene",
                "time": "reference" if reference else "inspection",
                "view": view, "camera_view": view, "available": True,
            })
        self.case = {
            "case_id": "choice-fixture-Obtainable", "family_id": "choice-fixture",
            "paired_history_id": "choice-pair", "claim": "Synthetic fixture only.",
            "initial": ["reference", "initial"], "evidence": evidence,
            "annotation": {"verdict": "Supported",
                           "minimal_evidence_sets": {"Supported": [["front_survey", "rear_left"]]}},
            "request_options": {"objects": ["inspection scene"], "times": ["inspection"],
                                "views": list(runner.native.REQUEST_VIEWS)},
            "multistep_certificate": {"target_rear_view": "rear_left", "SECRET": "DO_NOT_DISCLOSE"},
            "multistep_regions": {"SECRET_REGION": [99, 99, 100, 100]},
        }
        self.args = SimpleNamespace(budget=6, max_context_tokens=8192, seed=20261107)

    def tearDown(self):
        self.temp.cleanup()

    def environment(self, views=()):
        env = Environment(self.case, self.root, 6, ocr_backend=runner.native.NoOCR())
        for view in views:
            env.step({"type": "request_photo",
                      "query": {"object": "inspection scene", "time": "inspection", "view": view}})
        return env

    def run_arm(self, policy, choices):
        client = FakeClient(choices)
        row, trace = runner.run_episode(
            self.case, self.root, client, policy, self.args, observation_token, score)
        return client, row, trace

    def test_rgb_components_follow_visible_pixels_and_source_coordinates(self):
        env = self.environment(["front_survey"])
        images = {i["image_id"]: i for i in env.observation()["images"]}
        self.assertEqual(runner.rgb_boxes(images["front_survey"]),
                         [[10, 25, 50, 85], [70, 25, 110, 85], [130, 25, 170, 85]])
        self.assertEqual(runner.rgb_boxes(images["reference"]), [[10, 10, 70, 70]])
        remapped = {**images["reference"], "source_bbox": [100, 200, 260, 360]}
        self.assertEqual(runner.rgb_boxes(remapped), [[120, 220, 240, 340]])

    def test_candidates_use_only_released_images_and_rgb_boxes(self):
        observation = self.environment(["front_survey", "rear_left"]).observation()
        candidates = runner.terminal_candidates(observation)
        self.assertEqual(list(candidates), ["A", "B", "C"])
        self.assertEqual(candidates["B"]["verdict"], "Supported")
        self.assertEqual(candidates["C"]["verdict"], "Refuted")
        self.assertEqual(candidates["B"]["front_slot"], "left")
        self.assertEqual(len(candidates["B"]["links"]), 2)
        self.assertEqual(candidates["B"]["links"][0]["right"]["bbox"], [10, 25, 50, 85])
        self.assertEqual(candidates["B"]["links"][1]["right"]["bbox"], [10, 10, 70, 70])
        self.assertNotIn("rear_center", json.dumps(candidates))
        self.assertNotIn("SECRET", json.dumps(candidates))

    def test_missing_front_or_rear_does_not_force_abstention(self):
        for views in (["rear_left"], ["original", "original"]):
            with self.subTest(views=views):
                candidates = runner.terminal_candidates(self.environment(views).observation())
                self.assertEqual({v["verdict"] for v in candidates.values()},
                                 {"Supported", "Refuted", "Need more evidence"})
                self.assertTrue(all(v["links"] == [] for v in candidates.values()))

    def test_adaptive_two_requests_and_assisted_terminal(self):
        client, row, trace = self.run_arm("adaptive_acquisition", ["B", "C", "B"])
        self.assertEqual((row["requests"], row["request_cost"], row["model_calls"]), (2, 6, 3))
        self.assertTrue(row["choice_valid"])
        self.assertTrue(row["front_slot_assessable"])
        self.assertTrue(row["selected_chain_target_correct"])
        self.assertTrue(row["multistep_evaluation"]["chain_grounded_correct"])
        self.assertFalse(row["multistep_evaluation"]["localized_grounded_correct"])
        self.assertEqual(row["interface_scope"], runner.INTERFACE)
        self.assertFalse(trace["native_grounding_claim"])
        self.assertEqual(trace["candidate_generation"], runner.CANDIDATE_GENERATION)
        self.assertNotIn("SECRET", json.dumps(client.inputs))
        self.assertNotIn("multistep_certificate", json.dumps(client.inputs))
        self.assertNotIn("multistep_regions", json.dumps(client.inputs))

    def test_five_arms_share_initial_input_and_terminal_interface(self):
        hashes, systems = [], []
        for policy in runner.native.POLICIES:
            choices = ["B", "C", "A"] if policy == "adaptive_acquisition" else ["A"]
            client, row, trace = self.run_arm(policy, choices)
            hashes.append(row["initial_public_sha256"])
            systems.append(client.inputs[-1]["messages"][0])
            self.assertEqual(row["requests"], 2)
            self.assertEqual(row["request_cost"], 6)
            self.assertEqual(row["output_tokens"], len(client.inputs))
            self.assertEqual([g["phase"] for g in trace["generation"]].count("terminal"), 1)
            self.assertNotIn("tools", client.inputs[-1])
        self.assertEqual(len(set(hashes)), 1)
        self.assertTrue(all(system == systems[0] for system in systems))

    def test_invalid_selector_is_retained_without_retry(self):
        client, row, trace = self.run_arm("adaptive_acquisition", ["Z", "A"])
        self.assertEqual(len(client.inputs), 2)
        self.assertFalse(row["selection_complete"])
        self.assertEqual(row["requests"], 0)
        self.assertEqual(row["selection_error"], "invalid_acquisition_choice")
        self.assertTrue(row["choice_valid"])
        self.assertFalse(row["multistep_evaluation"]["source_label_correct"])
        self.assertEqual(trace["generation"][0]["raw"], "Z")

    def test_invalid_terminal_is_failure_and_not_retried(self):
        client, row, _ = self.run_arm("fixed_restoration", ["Z"])
        self.assertEqual(len(client.inputs), 1)
        self.assertFalse(row["finished"])
        self.assertFalse(row["choice_valid"])
        self.assertFalse(row["multistep_evaluation"]["source_label_correct"])

    def test_resume_preserves_raw_assisted_scores_but_summary_has_no_grounding_topline(self):
        _, row, trace = self.run_arm("fixed_restoration", ["B"])
        output = self.root / "output"
        output.mkdir()
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE episodes (case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
            runner.native.save_record(db, runner.native.TrajectoryStore(output / "pending.jsonl"), row, trace)
            saved, _ = runner.native.validate_saved(
                db, {(self.case["case_id"], "fixed_restoration")},
                {self.case["case_id"]: self.case}, output, score)
            self.assertEqual(len(saved), 1)
            summary = runner.summarize(db, [self.case], ["fixed_restoration"])
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["native_grounding_claim"])
        counts = summary["policies"]["fixed_restoration"]["counts"]
        self.assertIn("source_label_correct", counts)
        self.assertIn("evidence_sufficient", counts)
        self.assertIn("valid_choice", counts)
        self.assertNotIn("chain_grounded_correct", counts)
        self.assertNotIn("localized_grounded_correct", counts)

    def test_absent_front_is_never_reconstructed_from_case_secrets(self):
        _, _, trace = self.run_arm("fixed_restoration", ["B"])
        self.assertTrue(all(not candidate["links"] for candidate in trace["choice_candidates"].values()))
        self.assertTrue(all(candidate["front_slot"] is None for candidate in trace["choice_candidates"].values()))


if __name__ == "__main__":
    unittest.main()
