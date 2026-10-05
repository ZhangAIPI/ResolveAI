"""Verify the public process protocol using temporary image fixtures."""
import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class ServerTests(unittest.TestCase):
    def test_json_protocol_and_private_annotation_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), "gray").save(root / "image.png")
            case = {"claim": "fixture claim", "initial": ["a"], "evidence": [
                {"id": "a", "path": "image.png", "source_id": "source",
                 "party": "A", "object": "table", "time": "after",
                 "view": "front", "available": True}],
                "annotation": {"verdict": "Refuted", "secret": "PRIVATE_MARKER",
                               "minimal_evidence_sets": {"Refuted": [["a"]]}}}
            (root / "case.json").write_text(json.dumps(case))
            commands = [{"type": "fork", "new_branch": "alternative"},
                        {"type": "inspect", "branch": "alternative", "image_id": "a"},
                        {"type": "inspect", "image_id": "private-id"}]
            result = subprocess.run([sys.executable, "-m", "resolveai.server",
                str(root / "case.json"), str(root), "--allow-forks"],
                input="\n".join(json.dumps(c) for c in commands) + "\n",
                capture_output=True, text=True, check=True, timeout=30)
            self.assertNotIn("PRIVATE_MARKER", result.stdout)
            self.assertNotIn(str(root), result.stdout)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(base64.b64decode(rows[0]["images"][0]["image_png"]).startswith(b"\x89PNG"))
            self.assertEqual(rows[1]["branch"], "alternative")
            self.assertEqual(rows[2]["budget"], 11)
            self.assertEqual(rows[3], {"error": "unreleased_image"})
