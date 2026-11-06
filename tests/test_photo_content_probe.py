"""CPU tests for the selected-history content probe; no model inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "evaluate_photo_content_probe.py"
if not SCRIPT.exists():
    SCRIPT = HERE.parent / "scripts/evaluate_photo_content_probe.py"
SPEC = importlib.util.spec_from_file_location("content_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ContentProbeTests(unittest.TestCase):
    def fixture(self, root):
        from resolveai.tools import TOOLS
        pixels = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        paths = [probe.blob(pixels[::-1].copy(), root / "images"),
                 probe.blob(pixels[:2, :2].copy(), root / "images"),
                 probe.blob(pixels, root / "images")]
        image = lambda p: {"type": "image", "image_ref": str(p.relative_to(root))}
        trace = {"messages": [
            {"role": "system", "content": "native public instructions"},
            {"role": "user", "content": [{"type": "text", "text": "public claim"},
                image(paths[0]), image(paths[1])]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "request_photo", "arguments": {"query": {
                    "object": "item", "time": "capture", "view": "original"}}}}]},
            {"role": "tool", "name": "request_photo", "content": [
                {"type": "text", "text": '{"status":"provided","budget":0}'}, image(paths[2])]},
            {"role": "assistant", "content": "NEVER_INCLUDE_FUTURE_ANSWER"}],
            "tools": deepcopy(TOOLS), "generation": [{}, {"input_tokens": 1234}],
            "decision": {"private_truth": "NEVER_INCLUDE_PRIVATE_LABEL"}}
        return {"trace": trace, "directory": root, "result": {"family_id": "family-x"}}

    def fake_audit(self, processor, payload):
        hashes = [probe.rgb_sha(probe.rgb(p["image_ref"]))
                  for _, _, p in probe.image_parts(payload["messages"])]
        return {"input_tokens": 1234, "nonpixel_tensors": {"input_ids": "fixed"},
                "pixel_values_shape": [10, 3], "pixel_values_dtype": "float32",
                "pixel_values_sha256": probe.digest("".join(hashes).encode()),
                "pixel_blocks_sha256": hashes, "image_grid_thw": [[1, 2, 2]] * len(hashes)}

    def test_joint_rgb_permutation_and_seed(self):
        pixels = np.arange(11 * 13 * 3, dtype=np.uint8).reshape(11, 13, 3)
        result = probe.transformed(pixels, "shuffled", probe.family_seed("f"))
        self.assertEqual(result.shape, pixels.shape)
        self.assertEqual(sorted(map(tuple, result.reshape(-1, 3))),
                         sorted(map(tuple, pixels.reshape(-1, 3))))
        np.testing.assert_array_equal(result, probe.transformed(pixels, "shuffled", probe.family_seed("f")))
        self.assertFalse(np.array_equal(result, probe.transformed(pixels, "shuffled", probe.family_seed("g"))))
        self.assertTrue((probe.transformed(pixels, "gray", 0) == 127).all())

    def test_prefix_truncates_future_and_private_information(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(Path(d))
            prefix, slot = probe.selected_prefix(source["trace"], source["directory"])
            self.assertEqual(len(prefix["messages"]), 4)
            self.assertEqual(slot, (3, 1))
            self.assertNotIn("NEVER_INCLUDE", probe.canonical(prefix))
            self.assertEqual([t["function"]["name"] for t in prefix["tools"]], ["finish"])
            self.assertEqual(prefix["messages"][-1]["content"][0]["text"], '{"status":"provided","budget":0}')

    def test_only_returned_pixels_change_and_originals_remain_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = self.fixture(root)
            before = {p: probe.file_sha(p) for p in (root / "images").glob("*.png")}
            with patch.object(probe, "encode_audit", self.fake_audit):
                trio = probe.prepare(source, root / "derived", 42, None)
            self.assertEqual(len({probe.nonpixel_hash(x["payload"]) for x in trio.values()}), 1)
            self.assertEqual(len({x["input_sha256"] for x in trio.values()}), 3)
            self.assertEqual(len(list((root / "derived").glob("*.png"))), 2)
            for p, h in before.items():
                self.assertEqual(probe.file_sha(p), h)
            fixed = [[p["image_ref"] for _, _, p in probe.image_parts(x["payload"]["messages"])][:2]
                     for x in trio.values()]
            self.assertTrue(all(x == fixed[0] for x in fixed))
            for x in trio.values():
                wire = probe.transport(x["payload"])
                self.assertTrue(all("image_png" in p and "image_ref" not in p
                                    for _, _, p in probe.image_parts(wire["messages"])))
                self.assertNotIn("shuffled", probe.canonical(wire))
                self.assertNotIn("gray", probe.canonical(wire))

    def test_tensor_grid_and_initial_pixel_guards(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(Path(d))
            prefix, _ = probe.selected_prefix(source["trace"], source["directory"])
            a = self.fake_audit(None, prefix)
            b = deepcopy(a);b["image_grid_thw"][0] = [1, 9, 9]
            with self.assertRaises(ValueError):
                probe.matched([a, b], 2)
            b = deepcopy(a);b["pixel_blocks_sha256"][0] = "changed-reference"
            with self.assertRaises(ValueError):
                probe.matched([a, b], 2)
            b = deepcopy(a);b["pixel_values_sha256"] = "changed";b["pixel_blocks_sha256"][2] = "allowed-target"
            probe.matched([a, b], 2)

    def test_native_source_processor_guard(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(Path(d));source["trace"]["generation"][1]["input_tokens"] = 1
            with patch.object(probe, "encode_audit", self.fake_audit), self.assertRaises(ValueError):
                probe.prepare(source, Path(d) / "derived", 42, None)

    def test_single_model_call_invalid_outputs_are_not_retried(self):
        class Client:
            def __init__(self, answer):
                self.answer, self.calls = answer, 0
            def ask_conversation(self, payload, context):
                self.calls += 1
                self.payload, self.context = payload, context
                return self.answer
        valid = {"name": "finish", "arguments": {"verdict": "Need more evidence", "citations": []}}
        raw = "<tool_call>" + probe.canonical(valid) + "</tool_call>"
        for parse_error, call, output_raw, expected in [
                (False, valid, raw, True), (True, None, "bad", False),
                (False, {"name": "request_photo", "arguments": {}}, raw, False),
                (False, valid, raw + raw, False),
                (False, {"name": "finish", "arguments": {"type": "inspect", "image_id": "x"}}, raw, False),
                (False, {"name": "finish", "arguments": {"type": "finish", "verdict": "Supported", "citations": []}}, raw, False),
                (False, {"name": "finish", "arguments": {"citations": []}}, raw, False),
                (False, {"name": "finish", "arguments": {"verdict": "invented", "citations": []}}, raw, False)]:
            client = Client({"input_tokens": 1234, "parse_error": parse_error,
                             "tool_call": call, "raw": output_raw})
            payload = {"messages": [], "tools": [], "max_new_tokens": 384}
            _, passed, _ = probe.ask_once(client, payload, {"input_tokens": 1234})
            self.assertEqual(passed, expected)
            self.assertEqual(client.calls, 1)
            self.assertEqual(client.context, 8192)
            self.assertEqual(client.payload["max_new_tokens"], 384)

    def test_no_partial_scores_and_failed_denominator(self):
        import json
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE episodes (family_id TEXT, condition TEXT, result TEXT)")
        expected = {("f", c) for c in probe.CONDITIONS}
        for c in probe.CONDITIONS:
            row = {"family_id": "f", "condition": c, "source_label_match": c == "real",
                   "schema_valid_finish": c != "gray", "verdict": None,
                   "generation": {"latency_s": 1, "input_tokens": 10, "output_tokens": 3}}
            db.execute("INSERT INTO episodes VALUES (?,?,?)", ("f", c, json.dumps(row)))
            result = probe.summary(db, expected)
            if c != "shuffled":
                self.assertFalse(result["complete"])
                self.assertNotIn("conditions", result)
        self.assertTrue(result["complete"])
        self.assertEqual(result["conditions"]["gray"]["attempts"], 1)
        self.assertEqual(result["conditions"]["gray"]["schema_valid_finish"], 0)

    def test_completed_resume_repairs_missing_summary_without_model_loading(self):
        import json
        import sys
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);directory = root / "out/shard-0";directory.mkdir(parents=True)
            payload = {"messages": [], "tools": [], "max_new_tokens": 384}
            db = sqlite3.connect(directory / "episodes.sqlite")
            db.execute("CREATE TABLE episodes (family_id TEXT, condition TEXT, result TEXT, trace TEXT, PRIMARY KEY(family_id,condition))")
            for condition in probe.CONDITIONS:
                row = {"family_id": "f", "condition": condition, "source_record_sha256": "source",
                       "input_sha256": probe.digest(probe.canonical(probe.transport(payload)).encode()),
                       "nonpixel_public_sha256": probe.nonpixel_hash(payload),
                       "source_label_match": False, "schema_valid_finish": False, "verdict": None, "generation": {}}
                db.execute("INSERT INTO episodes VALUES (?,?,?,?)",
                           ("f", condition, json.dumps(row), json.dumps(payload)))
            db.commit();db.close()
            argv = ["probe", str(root), str(root), str(root / "out")]
            with patch.object(sys, "argv", argv), patch.object(probe, "sources",
                    return_value=({"f": {"record_sha256": "source"}}, {})):
                probe.main()
            self.assertTrue(json.loads((directory / "summary.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
