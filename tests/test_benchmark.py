"""Public evaluation invariants and lossless cross-checkpoint tool serialization."""
import json
from pathlib import Path
import tempfile
import unittest

from resolveai.benchmark import locked_manifest
from resolveai.benchmark_report import model_summary, interval
from resolveai.vlm_worker import portable_tools, parse_tool_call


class BenchmarkTests(unittest.TestCase):
    def test_portable_history_retains_schemas_calls_results_and_images(self):
        messages = [{"role": "system", "content": "verify"},
            {"role": "assistant", "content": "inspect", "tool_calls": [{"id": "c1",
             "function": {"name": "inspect", "arguments": {"image_id": "photo"}}}]},
            {"role": "tool", "name": "inspect", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "actual pixels"}, {"type": "image", "image_png": "abc"}]}]
        result = portable_tools(messages, [{"function": {"name": "inspect"}}])
        self.assertIn('"name": "inspect"', result[0]["content"])
        self.assertIn('<tool_call>', result[1]["content"])
        self.assertEqual(result[2]["role"], "user")
        self.assertIn('"tool_call_id": "c1"', result[2]["content"][0]["text"])
        self.assertEqual(result[2]["content"][2], messages[2]["content"][1])
        self.assertEqual(messages[2]["role"], "tool")

    def test_resume_rejects_a_changed_evaluation_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"configuration": {"dataset_sha256": "first"}}))
            self.assertEqual(locked_manifest(path, {"dataset_sha256": "first"})["configuration"]["dataset_sha256"], "first")
            with self.assertRaises(ValueError):
                locked_manifest(path, {"dataset_sha256": "different"})

    def test_final_summary_rejects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"configuration": {
                "limit_families": None, "dataset_manifest": {"families": 2},
                "policies": ["initial", "agent"]}}))
            (root / "results.jsonl").write_text(json.dumps({"case_id": "family-Sufficient",
                "family_id": "family", "policy": "initial"}) + "\n")
            with self.assertRaises(ValueError):
                model_summary(root)

    def test_interval_for_two_independent_source_units(self):
        self.assertEqual(interval([0., 1.])["ci95"], [0., 1.])

    def test_correlated_variants_share_one_statistical_unit(self):
        from resolveai.benchmark_report import paired
        rows = []
        for family, improvement in (("one", True), ("two", False)):
            for variant in ("Sufficient", "Obtainable", "Missing", "Unavailable"):
                for policy in ("initial", "agent"):
                    rows.append({"family_id": family, "case_id": family + variant,
                        "policy": policy, "grounded_correct": improvement and policy == "agent"})
        result = paired(rows)
        self.assertEqual(result["paired_source_families"], 2)
        self.assertEqual(result["grounded_proxy_delta"], .5)
        self.assertEqual(result["family_ci95"], [0.0, 1.0])

    def test_portable_transport_accepts_only_complete_unclosed_json(self):
        raw = '<tool_call>{"name":"finish","arguments":{"verdict":"Need more evidence","citations":[]}}'
        self.assertEqual(parse_tool_call(raw, allow_unclosed=True)[0]["name"], "finish")
        with self.assertRaises(ValueError):
            parse_tool_call(raw[:-1], allow_unclosed=True)
        with self.assertRaises(ValueError):
            parse_tool_call(raw, allow_unclosed=False)
