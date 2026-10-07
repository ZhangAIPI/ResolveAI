"""Exercise the HTTP permission boundary, staged evidence and durable answers."""

import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from PIL import Image
from resolveai.curation import proposal, variants
from resolveai.human_study import Study, create_server


class HumanWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / "data"
        data.mkdir()
        for name, size in [("original", 32), ("preview", 8), ("reference", 32)]:
            Image.new("RGB", (size, size), "white").save(data / (name + ".png"))
        evidence = [
            {
                "id": name,
                "source_id": name,
                "path": name + ".png",
                "available": True,
                "object": "normal-reference" if name == "reference" else "item",
                "party": "submitted",
                "time": "capture",
                "view": "original" if name == "original" else name,
                "source_size": [32, 32],
                "source_bbox": [0, 0, 32, 32],
            }
            for name in ("original", "preview", "reference")
        ]
        family = {
            "family_id": "f",
            "task": "state",
            "category": "bottle",
            "claim": "The bottle rim is broken.",
            "asset_root": str(data),
            "provenance": {"criterion": "a chipped or broken rim"},
            "initial": ["reference", "preview"],
            "evidence": evidence,
            "request_options": {
                "objects": ["item"],
                "times": ["capture"],
                "views": ["original"],
            },
            "annotation": proposal(
                "Supported",
                "condition",
                [{"image_id": "original", "time": "capture", "bbox": [0, 0, 32, 32]}],
            ),
        }
        cases = list(variants(family, "original"))
        (data / "cases.json").write_text(json.dumps(cases))
        plan = {
            "data": str(data),
            "dataset_sha256": hashlib.sha256(
                (data / "cases.json").read_bytes()
            ).hexdigest(),
            "family_ids": ["f"],
            "budget": 12,
            "max_turns": 8,
            "search_seconds": 75,
            "review_seconds": 120,
        }
        (self.root / "plan.json").write_text(json.dumps(plan))
        access = {
            "person": {
                "id": "P01",
                "role": "participant",
                "tasks": [
                    {
                        "mode": "search",
                        "family_id": "f",
                        "case_id": "f-Obtainable",
                        "condition": "interactive",
                    }
                ],
            },
            "reviewer": {
                "id": "P02",
                "role": "participant",
                "tasks": [{"mode": "review", "family_id": "f"}],
            },
            "organizer": {"id": "admin", "role": "admin"},
        }
        (self.root / "access.json").write_text(json.dumps(access))
        self.study = Study(self.root)

    def tearDown(self):
        self.study.db.close()
        self.temp.cleanup()

    def test_no_images_before_consent_and_no_truth_in_public_state(self):
        state = self.study.state("person")
        self.assertTrue(state["intro"])
        self.assertNotIn("images", state)
        state = self.study.post("person", "consent", {"agree": True})
        self.assertEqual(
            {r["image_id"] for r in state["images"]}, {"preview", "reference"}
        )
        for key in ("annotation", "provenance_truth", "asset_root", "path"):
            self.assertNotIn('"' + key + '"', json.dumps(state))

    def test_failed_answer_validation_does_not_finish_the_environment(self):
        self.study.post("person", "consent", {"agree": True})
        with self.assertRaises(ValueError):
            self.study.post(
                "person",
                "answer",
                {
                    "index": 0,
                    "verdict": "Need more evidence",
                    "selected": [],
                    "confidence": 0,
                    "reason": "uncertain",
                },
            )
        self.assertFalse(self.study.environment(self.study.actor("person"), 0).finished)

    def test_actions_resume_without_time_points_or_step_limits(
        self,
    ):
        self.study.post("person", "consent", {"agree": True})
        state = self.study.post(
            "person",
            "action",
            {
                "index": 0,
                "action": {
                    "type": "request_photo",
                    "query": {"object": "item", "time": "capture", "view": "original"},
                },
            },
        )
        self.assertIn("original", [r["image_id"] for r in state["images"]])
        reopened = Study(self.root)
        self.assertIn(
            "original", [r["image_id"] for r in reopened.state("person")["images"]]
        )
        reopened.db.close()
        self.study.db.execute("UPDATE starts SET at=at-7200")
        self.study.db.commit()
        for _ in range(40):
            state = self.study.post(
                "person",
                "action",
                {"index": 0, "action": {"type": "inspect", "image_id": "original"}},
            )
        self.assertEqual(state["turns"], 41)
        self.assertGreaterEqual(state["elapsed_seconds"], 7200)
        self.assertNotIn("budget", state)
        self.assertNotIn("remaining_seconds", state)
        self.assertNotIn("max_turns", state)
        reopened = Study(self.root)
        self.assertEqual(reopened.state("person")["turns"], 41)
        reopened.db.close()
        with self.assertRaises(ValueError):
            self.study.post("person", "timeout", {"index": 0})
        self.assertEqual(self.study.export(), [])
        self.study.post(
            "person",
            "answer",
            {
                "index": 0,
                "verdict": "Need more evidence",
                "selected": [],
                "confidence": 3,
                "reason": "uncertain",
            },
        )
        exported = self.study.export()[0]
        self.assertEqual(exported["status"], "finished")
        self.assertEqual(exported["steps"], 41)
        self.assertGreaterEqual(exported["duration_s"], 7200)
        self.assertEqual(exported["ui_protocol"], "human-ui-v3-unrestricted")

    def test_review_cannot_skip_to_full_evidence_or_rewrite_an_initial_judgment(self):
        state = self.study.post("reviewer", "consent", {"agree": True})
        self.assertEqual(state["phase"], 0)
        self.assertNotIn("original", [r["image_id"] for r in state["images"]])
        body = {
            "index": 0,
            "phase": 4,
            "verdict": "Need more evidence",
            "selected": [],
            "confidence": 3,
            "reason": "unclear",
        }
        with self.assertRaises(ValueError):
            self.study.post("reviewer", "review", body)
        body["phase"] = 0
        state = self.study.post("reviewer", "review", body)
        self.assertEqual(state["phase"], 1)
        self.assertNotIn("original", [r["image_id"] for r in state["images"]])
        with self.assertRaises(ValueError):
            self.study.post("reviewer", "review", body)

    def test_http_blocks_unreleased_images_and_private_exports(self):
        server = create_server(self.root, 0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_port)
            conn.request("GET", "/api/state")
            r = conn.getresponse()
            r.read()
            self.assertEqual(r.status, 403)
            conn.request(
                "GET", "/api/export", headers={"Authorization": "Bearer person"}
            )
            r = conn.getresponse()
            r.read()
            self.assertEqual(r.status, 403)
            self.study.post("reviewer", "consent", {"agree": True})
            conn.request(
                "GET",
                "/api/image/original",
                headers={"Authorization": "Bearer reviewer"},
            )
            r = conn.getresponse()
            r.read()
            self.assertEqual(r.status, 403)
            conn.request("GET", "/", headers={})
            r = conn.getresponse()
            self.assertIn("看图与搜证", r.read().decode())
            self.assertEqual(r.status, 200)
            conn.request("GET", "/i18n.js")
            r = conn.getresponse()
            self.assertEqual(r.status, 200)
            self.assertIn("applyLanguage", r.read().decode())
            conn.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
