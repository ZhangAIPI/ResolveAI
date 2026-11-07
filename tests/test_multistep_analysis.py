"""Small synthetic storage fixtures for complete-only paired-bank statistics."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.analyze_multistep import (
    POLICIES, analyze, bank_bootstrap, canonical, digest, markdown,
)


def hashed(text):
    return sha256(text.encode()).hexdigest()


def dataset(root):
    cases = []
    for bank in range(2):
        for member, target in enumerate(("rear_left", "rear_center")):
            cases.append({
                "case_id": f"case-{bank}-{member}-Obtainable",
                "family_id": f"family-{bank}-{member}", "paired_history_id": f"bank-{bank}",
                "variant": "Obtainable", "task": "identity_condition",
                "provenance_truth": "Supported" if member == 0 else "Refuted",
                "request_options": {"objects": ["inspection scene"], "times": ["inspection"],
                                    "views": ["original", "front_survey", "rear_left", "rear_center", "rear_right"]},
                "multistep_certificate": {
                    "bank_sha256": hashed(f"bank{bank}"), "target_rear_view": target,
                    "required_links": [
                        {"relation": "same_object",
                         "left": {"image_id": "reference", "time": "reference", "bbox": [0, 0, 10, 10]},
                         "right": {"image_id": "front_survey", "time": "inspection", "bbox": [0, 0, 10, 10]}},
                        {"relation": "same_object",
                         "left": {"image_id": "front_survey", "time": "inspection", "bbox": [0, 0, 10, 10]},
                         "right": {"image_id": target, "time": "inspection", "bbox": [0, 0, 10, 10]}}],
                },
            })
    root.mkdir()
    path = root / "cases.json"
    path.write_text(json.dumps(cases))
    (root / "dataset_manifest.json").write_text(json.dumps({
        "dataset_sha256": digest(path), "annotation_status": "procedural",
        "paired_scene_banks": 2, "renderer_source_sha256": hashed("renderer")}))
    return cases


def write_run(root, cases, source_hash, *, model="model-a", missing=(), failures=(), shards=1,
              changed_config=None, accounting_failures=()):
    for shard in range(shards):
        directory = root / f"shard-{shard}"
        directory.mkdir(parents=True)
        families = sorted(c["family_id"] for c in cases)[shard::shards]
        selected = [c for c in cases if c["family_id"] in families]
        config = {
            "experiment": "procedural-multistep-two-request-v2",
            "dataset_sha256": source_hash, "model": {"checkpoint": model},
            "model_id": model, "revision": "frozen",
            "source_sha256": hashed("source"), "script_sha256": hashed("runner"),
            "assets": {"/fixture/asset.png": {"sha256": hashed("asset"), "bytes": 100}},
            "shard": shard, "shards": shards, "families": families,
            "case_ids": [c["case_id"] for c in selected],
            "policies": list(POLICIES), "variants": ["Obtainable"],
            "budget": 6, "max_turns": 8, "max_new_tokens": 768, "max_context_tokens": 8192,
            "seed": 20261107, "do_sample": False, "request_slots": 2,
            "request_views": ["original", "front_survey", "rear_left", "rear_center", "rear_right"],
            "random_seed_group": "paired_history_id", "random_rule": "frozen",
            "acquisition_instruction": "select", "terminal_instruction": "finish",
            "terminal_input": "fresh", "terminal_calls": 1,
            "acquisition_assistant_text": "withheld-for-all-policies",
        }
        config.update(changed_config or {})
        (directory / "manifest.json").write_text(json.dumps({"configuration": config, "source_commit": "fixture"}))
        with sqlite3.connect(directory / "episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes(case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
            for case in selected:
                for policy in POLICIES:
                    key = case["case_id"], policy
                    if key in missing:
                        continue
                    target = case["multistep_certificate"]["target_rear_view"]
                    views = (["front_survey", target] if policy in ("adaptive_acquisition", "oracle")
                             else ["original", "original"] if policy == "fixed_restoration"
                             else ["front_survey", "rear_left"] if policy == "fixed_view_order"
                             else ["rear_left", "front_survey"])
                    found = "front_survey" in views and target in views
                    finished = key not in failures
                    correct = found and finished
                    strict = correct and target == "rear_left"
                    score = {
                        "procedural_evidence_sufficient": found, "chain_grounded_correct": correct,
                        "localized_grounded_correct": strict, "simulator_grounded_correct": strict,
                        "source_label_correct": correct, "unsupported_decision": False,
                    }
                    trace = {"decision": ({
                        "type": "finish", "verdict": case["provenance_truth"] if found else "Need more evidence",
                        "citations": [], "links": deepcopy(case["multistep_certificate"]["required_links"]) if found else []
                    } if finished else None),
                             "multistep_evaluation": deepcopy(score),
                             "acquisition": {"executed_requests": [
                                 {"call": {"name": "request_photo", "arguments": {"query": {
                                     "object": "inspection scene", "time": "inspection", "view": view}}},
                                  "status": "provided", "error": None} for view in views]}}
                    row = {
                        "case_id": case["case_id"], "family_id": case["family_id"], "policy": policy,
                        "multistep_evaluation": score, "finished": finished,
                        "requests": 2, "selection_complete": True,
                        "initial_public_sha256": hashed(case["paired_history_id"]),
                        "input_tokens": 100, "output_tokens": 20,
                        "wall_latency_s": 2.0, "model_calls": 3 if policy == "adaptive_acquisition" else 1,
                        "new_image_requests": int(found) + 1 if policy != "fixed_restoration" else 0,
                        "token_accounting_complete": key not in accounting_failures,
                        "trace_sha256": sha256(canonical(trace).encode()).hexdigest(),
                    }
                    db.execute("INSERT INTO episodes VALUES(?,?,?,?)",
                               (case["case_id"], policy, json.dumps(row), canonical(trace)))


def assisted_run(root):
    for path in root.glob("shard-*/manifest.json"):
        manifest = json.loads(path.read_text())
        manifest["configuration"].update(
            experiment="procedural-multistep-interface-assisted-choice-v1",
            interface_scope="interface-assisted-finite-choice-diagnostic",
            native_grounding_claim=False, automatic_identity_links=True)
        path.write_text(json.dumps(manifest))
        with sqlite3.connect(path.parent / "episodes.sqlite") as db:
            rows = list(db.execute("SELECT case_id,policy,result,trace FROM episodes"))
            for case_id, policy, raw, trace_raw in rows:
                row, trace = json.loads(raw), json.loads(trace_raw)
                views = [e["call"]["arguments"]["query"]["view"]
                         for e in trace["acquisition"]["executed_requests"]]
                assessable = "front_survey" in views
                row.update(
                    interface_scope="interface-assisted-finite-choice-diagnostic",
                    choice_valid=bool(row["finished"]), front_slot_assessable=assessable,
                    selected_chain_target_correct=assessable,
                    selected_front_slot="selected-public-slot" if assessable else None)
                db.execute("UPDATE episodes SET result=? WHERE case_id=? AND policy=?",
                           (json.dumps(row), case_id, policy))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.cases = dataset(self.data)
        self.source_hash = digest(self.data / "cases.json")

    def run_fixture(self, name, **kwargs):
        root = self.root / name
        write_run(root, self.cases, self.source_hash, model=name, **kwargs)
        return root

    def test_complete_two_models_and_shards_preserve_failures_in_denominator(self):
        failure = (self.cases[0]["case_id"], "adaptive_acquisition")
        a = self.run_fixture("a", shards=2, failures=[failure])
        b = self.run_fixture("b", shards=1)
        report = analyze(self.data, {"a": a, "b": b}, replicates=200)
        self.assertTrue(report["complete"])
        values = report["models"]["a"]["policies"]["adaptive_acquisition"]
        self.assertEqual(values["episodes"], 4)
        self.assertEqual(values["rates_pct"]["procedural_evidence_sufficient"], 100)
        self.assertEqual(values["rates_pct"]["chain_grounded_correct"], 75)
        self.assertEqual(values["rates_pct"]["finished"], 75)
        self.assertEqual(report["cohort"]["paired_banks"], 2)
        self.assertEqual(values["decision_when_evidence_found"]["source_label_accuracy_pct"], 75)
        self.assertEqual(values["identity_links"]["valid_pct_given_definite_finish"], 100)

    def test_incomplete_model_has_no_scores_but_complete_model_can_report(self):
        a = self.run_fixture("a")
        b = self.run_fixture("b", missing=[(self.cases[0]["case_id"], "oracle")])
        report = analyze(self.data, {"a": a, "b": b}, replicates=100)
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"]["b"]["completed_episodes"], 19)
        self.assertIn("policies", report["models"]["a"])
        self.assertNotIn("adaptive_minus_baseline", report["models"]["b"])
        self.assertFalse(report["cross_model_comparison_ready"])
        self.assertIn("未完整模型仅列完成数量", markdown(report))

    def test_duplicate_restoration_and_paired_target_switch_are_distinct(self):
        a = self.run_fixture("a")
        report = analyze(self.data, {"a": a}, replicates=200)
        policies = report["models"]["a"]["policies"]
        restore = policies["fixed_restoration"]
        self.assertEqual(restore["mean"]["duplicate_requests"], 1)
        self.assertEqual(restore["rates_pct"]["episodes_with_duplicate_requests"], 100)
        self.assertEqual(restore["rates_pct"]["procedural_evidence_sufficient"], 0)
        self.assertEqual(policies["adaptive_acquisition"]["paired_history_behavior"]["correct_switch_pct_all_banks"], 100)
        self.assertEqual(policies["fixed_view_order"]["paired_history_behavior"]["correct_switch_pct_all_banks"], 0)
        self.assertEqual(policies["random_request"]["rates_pct"]["front_then_target_rear"], 0)
        self.assertEqual(policies["random_request"]["rates_pct"]["procedural_evidence_sufficient"], 50)

    def test_paired_found_requires_both_members_not_only_one_lucky_target(self):
        a = self.run_fixture("paired-found")
        report = analyze(self.data, {"a": a}, replicates=100)
        policies = report["models"]["a"]["policies"]
        adaptive = policies["adaptive_acquisition"]["paired_history_behavior"]
        fixed = policies["fixed_view_order"]["paired_history_behavior"]
        self.assertEqual(adaptive["counts"]["both_members_evidence_found"], 2)
        self.assertEqual(adaptive["both_members_evidence_found_pct"], 100)
        self.assertEqual(fixed["counts"]["both_members_evidence_found"], 0)
        self.assertEqual(fixed["both_members_evidence_found_pct"], 0)
        self.assertEqual(policies["fixed_view_order"]["rates_pct"]["procedural_evidence_sufficient"], 50)

    def test_cluster_bootstrap_is_seeded_and_weights_all_cases(self):
        values = {"a": 1, "b": 1, "c": 1, "d": -1}
        groups = {"a": "bank1", "b": "bank1", "c": "bank1", "d": "bank2"}
        first = bank_bootstrap(values, groups, seed=20261107, replicates=2000)
        self.assertEqual(first, bank_bootstrap(values, groups, seed=20261107, replicates=2000))
        self.assertEqual(first["banks"], 2)
        self.assertEqual(first["difference_pp"], 50)
        self.assertEqual(first["paired_bank_bootstrap_ci95_pp"], [-100, 100])
        with self.assertRaises(ValueError):
            bank_bootstrap(values, {"a": "bank1"})

    def test_mixed_settings_or_dataset_are_rejected(self):
        a = self.run_fixture("a")
        b = self.run_fixture("b", changed_config={"max_new_tokens": 384})
        with self.assertRaisesRegex(ValueError, "inference settings"):
            analyze(self.data, {"a": a, "b": b}, replicates=100)
        manifest = b / "shard-0/manifest.json"
        doc = json.loads(manifest.read_text())
        doc["configuration"]["dataset_sha256"] = hashed("wrong-data")
        manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            analyze(self.data, {"b": b}, replicates=100)

    def test_bad_trace_or_unplanned_keys_cannot_enter_final_table(self):
        a = self.run_fixture("a")
        database = a / "shard-0/episodes.sqlite"
        with sqlite3.connect(database) as db:
            db.execute("UPDATE episodes SET trace='{}' WHERE case_id=? AND policy=?",
                       (self.cases[0]["case_id"], "oracle"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            analyze(self.data, {"a": a}, replicates=100)

    def test_incomplete_transport_accounting_yields_no_token_average(self):
        key = (self.cases[0]["case_id"], "adaptive_acquisition")
        a = self.run_fixture("a", accounting_failures=[key])
        report = analyze(self.data, {"a": a}, replicates=100)
        values = report["models"]["a"]["policies"]["adaptive_acquisition"]
        self.assertIsNone(values["mean"]["input_tokens"])
        self.assertIsNone(values["mean"]["output_tokens"])
        self.assertFalse(values["token_accounting_complete"])
        self.assertIn("—", markdown(report))

    def test_dataset_manifest_tamper_and_pair_group_tamper_are_rejected(self):
        a = self.run_fixture("a")
        source = self.data / "cases.json"
        source.write_text(source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analyze(self.data, {"a": a}, replicates=100)

    def test_assisted_choice_cannot_mix_with_native_models(self):
        native = self.run_fixture("native")
        choice = self.run_fixture("choice")
        assisted_run(choice)
        with self.assertRaisesRegex(ValueError, "interfaces"):
            analyze(self.data, {"native": native, "choice": choice}, replicates=100)

    def test_assisted_choice_has_no_native_grounding_topline(self):
        choice = self.run_fixture("choice")
        assisted_run(choice)
        report = analyze(self.data, {"choice": choice}, replicates=100)
        self.assertFalse(report["native_grounding_claim"])
        self.assertEqual(report["interface_track"], "assisted_finite_choice")
        result = report["models"]["choice"]["policies"]["adaptive_acquisition"]
        self.assertEqual(result["rates_pct"]["choice_valid"], 100)
        self.assertEqual(result["rates_pct"]["selected_chain_target_correct"], 100)
        self.assertNotIn("chain_grounded_correct", result["rates_pct"])
        self.assertIn("chain_grounded_correct", result["assisted_scoring_raw_counts"])
        self.assertNotIn("chain_grounded_correct",
                         report["models"]["choice"]["adaptive_minus_baseline"]["fixed_view_order"])
        self.assertIn("界面辅助", markdown(report))
        self.assertNotIn("| 证据链正确%", markdown(report))

    def test_pending_model_directory_has_no_scores(self):
        report = analyze(self.data, {"not_started": self.root / "pending"}, replicates=100)
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"]["not_started"]["completed_episodes"], 0)
        self.assertNotIn("policies", report["models"]["not_started"])


if __name__ == "__main__":
    unittest.main()
