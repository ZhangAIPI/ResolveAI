"""Controls preserve pixels and choices; queue failures retain the allocation."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from PIL import Image
from scripts.evaluate_controls import OutputBudget, finish_direct
from resolveai.diagnostic import text_direct
from resolveai.environment import Environment


class NoOCR:
    available = False


class Client:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    def ask_conversation(self, payload, max_context_tokens=8192):
        self.payloads.append(payload)
        return self.response


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (24, 24)).save(self.root / "photo.png")
        self.case = {
            "case_id": "c", "family_id": "f", "claim": "A defect is visible.",
            "initial": ["photo"], "evidence": [
                {"id": "photo", "source_id": "private-source", "party": "submitted",
                 "path": "photo.png", "object": "cup",
                 "time": "capture", "view": "front", "available": True}
            ],
            "annotation": {"verdict": "Supported",
                           "minimal_evidence_sets": {"Supported": [["photo"]]}},
        }

    def tearDown(self):
        self.temp.cleanup()

    def env(self):
        return Environment(self.case, self.root, ocr_backend=NoOCR())

    def test_budget_control_changes_only_the_generation_allowance(self):
        a = Client({"raw": "Need more evidence"})
        b = Client({"raw": "Need more evidence"})
        text_direct(self.env(), self.case, a)
        text_direct(self.env(), self.case, OutputBudget(b, 384))
        left, right = a.payloads[0], b.payloads[0]
        self.assertEqual(left["max_new_tokens"], 96)
        self.assertEqual(right["max_new_tokens"], 384)
        self.assertEqual(
            {k: v for k, v in left.items() if k != "max_new_tokens"},
            {k: v for k, v in right.items() if k != "max_new_tokens"},
        )
        self.assertNotIn("minimal_evidence_sets", json.dumps(right))

    def test_terminal_control_uses_the_real_finish_schema(self):
        client = Client({"parse_error": False, "raw": "", "text": "",
                         "tool_call": {"name": "finish", "arguments": {
                             "verdict": "Need more evidence", "citations": [], "links": []}}})
        env = self.env()
        trace = finish_direct(env, OutputBudget(client, 384))
        self.assertEqual(trace["termination"], "finished")
        self.assertEqual(trace["decision"]["verdict"], "Need more evidence")
        self.assertEqual(env.requests, 0)
        self.assertEqual([x["function"]["name"] for x in client.payloads[0]["tools"]], ["finish"])

    def test_terminal_control_does_not_repair_an_acquisition_call(self):
        client = Client({"parse_error": False, "raw": "", "text": "",
                         "tool_call": {"name": "inspect", "arguments": {"image_id": "photo"}}})
        trace = finish_direct(self.env(), OutputBudget(client, 384))
        self.assertIsNone(trace["decision"])
        self.assertEqual(trace["termination"], "max_turns")
        self.assertEqual(trace["errors"], 1)


class QueueTests(unittest.TestCase):
    def test_task_failure_does_not_exit_worker_or_block_next_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = {"test": [
                {"id": "fails", "command": [sys.executable, "-c", "raise SystemExit(7)"],
                 "cwd": directory},
                {"id": "works", "command": [sys.executable, "-c", "pass"], "cwd": directory},
            ]}
            (root / "tasks.json").write_text(json.dumps(tasks))
            worker = Path(__file__).resolve().parents[1] / "scripts/gpu_queue_worker.py"
            with (root / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, str(worker), directory, "test"],
                    env=dict(os.environ, SLURM_JOB_ID="test-process-no-gpu-allocation"),
                    stdout=log, stderr=log,
                )
                try:
                    deadline = time.monotonic() + 10
                    rows = []
                    while time.monotonic() < deadline:
                        if (root / "queue.sqlite").exists():
                            with sqlite3.connect(root / "queue.sqlite") as db:
                                try:
                                    rows = db.execute("SELECT id,status,exit_code FROM tasks ORDER BY id").fetchall()
                                except sqlite3.OperationalError:
                                    pass
                        if len(rows) == 2 and rows[-1][1] == "completed":
                            break
                        time.sleep(.1)
                    self.assertEqual(rows, [("fails", "failed", 7), ("works", "completed", 0)])
                    self.assertIsNone(process.poll(), "Queue completion must not release the worker")
                finally:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
