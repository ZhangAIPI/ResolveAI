"""Allocation reuse must not execute a shared task twice or bypass failed dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.gpu_task_pool import claim, finish


class PoolTests(unittest.TestCase):
    def test_claims_are_exclusive_and_dependencies_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [{"id":"train","command":["true"]},
                     {"id":"eval","command":["true"],"requires":["train"]}]
            task, lock = claim(root,tasks,"one")
            self.assertEqual(task["id"],"train")
            self.assertIsNone(claim(root,tasks,"two"))
            finish(root,task,0)
            lock.close()
            task, lock = claim(root,tasks,"two")
            self.assertEqual(task["id"],"eval")
            finish(root,task,1)
            lock.close()
            self.assertIsNone(claim(root,tasks,"one"))

    def test_dead_worker_resumes_but_registered_command_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id":"train","command":["true"]}
            _, lock = claim(root,[task],"one")
            lock.close()  # Process died before its completion marker.
            resumed, lock = claim(root,[task],"two")
            self.assertEqual(resumed,task)
            finish(root,resumed,0)
            lock.close()
            with self.assertRaisesRegex(ValueError,"changed"):
                claim(root,[{**task,"command":["false"]}],"three")
