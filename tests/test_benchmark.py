"""Public evaluation invariants and lossless cross-checkpoint tool serialization."""
import json
from pathlib import Path
import tempfile
import unittest

from resolveai.benchmark import locked_manifest
from resolveai.vlm_worker import portable_tools


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
