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
        self.assertEqual(exported["ui_protocol"], "human-ui-v6-optional-reason")

    def test_review_has_tools_and_finishes_with_one_truth_only_vote(self):
        from resolveai.review import validate_truth_vote, validate_vote

        state = self.study.post("reviewer", "consent", {"agree": True})
        self.assertNotIn("phase", state)
        self.assertEqual(state["review_scope"], "full_pool_only")
        self.assertIn("original", [r["image_id"] for r in state["images"]])
        state = self.study.post(
            "reviewer",
            "action",
            {
                "index": 0,
                "action": {"type": "zoom", "image_id": "original", "factor": 2},
            },
        )
        self.assertEqual(state["turns"], 1)
        original = next(r for r in state["images"] if r["view_id"] == "original")
        self.assertEqual(original["display_scale"], 2)
        self.assertEqual(original["display_size"], [32, 32])
        answer = {
            "index": 0,
            "verdict": "Need more evidence",
            "selected": [],
            "confidence": 3,
            "reason": "unclear",
            "clear": True,
        }
        self.assertTrue(self.study.post("reviewer", "review", answer)["submitted"])
        self.assertTrue(self.study.state("reviewer")["done"])
        vote = self.study.export()[0]["vote"]
        validate_truth_vote(vote)
        self.assertNotIn("initial_verdicts", vote)
        self.assertNotIn("pool_verdicts", vote)
        with self.assertRaises(ValueError):
            validate_vote(vote)
        with self.assertRaises(ValueError):
            self.study.post("reviewer", "review", answer)

    def test_reason_is_optional_in_search_and_review(self):
        from copy import deepcopy
        from resolveai.review import validate_truth_vote

        for template, route in [("person", "answer"), ("reviewer", "review")]:
            for index, reason in enumerate([None, "", "划"]):
                with self.subTest(route=route, reason=reason):
                    token = f"{template}-{index}"
                    self.study.access[token] = deepcopy(self.study.access[template])
                    self.study.access[token]["id"] = token
                    self.study.post(token, "consent", {"agree": True})
                    body = {
                        "index": 0,
                        "verdict": "Supported",
                        "confidence": 3,
                        "clear": True,
                        "selected": [],
                    }
                    if reason is not None:
                        body["reason"] = reason
                    self.assertTrue(self.study.post(token, route, body)["submitted"])
                    row = json.loads(
                        self.study.db.execute(
                            "SELECT payload FROM answers WHERE actor=?", (token,)
                        ).fetchone()[0]
                    )
                    self.assertEqual(
                        row.get("reason", row.get("vote", {}).get("reason")),
                        reason or "",
                    )
                    if route == "review":
                        validate_truth_vote(row["vote"])
                        self.assertEqual(row["vote"]["reason"], reason or "")
                        self.assertNotIn("annotation", row["vote"])
        # Earlier evidence-annotating reviews keep their original requirements.
        legacy = {
            k: v
            for k, v in row["vote"].items()
            if k not in {"verdict", "regions", "links"}
        }
        legacy.update(
            protocol="visual-truth-review-v2",
            reason="visible",
            annotation=deepcopy(next(iter(self.study.cases.values()))["annotation"]),
        )
        validate_truth_vote(legacy)
        legacy["reason"] = ""
        with self.assertRaises(ValueError):
            validate_truth_vote(legacy)

    def test_definite_answers_need_no_photo_selection(self):
        from resolveai.review import full_truth_consensus
        from copy import deepcopy

        for actor, route in [("person", "answer"), ("reviewer", "review")]:
            self.study.post(actor, "consent", {"agree": True})
            result = self.study.post(
                actor,
                route,
                {
                    "index": 0,
                    "verdict": "Supported",
                    "confidence": 3,
                    "reason": "visible rim",
                    "clear": True,
                    "selected": [],
                },
            )
            self.assertTrue(result["submitted"])
        rows = self.study.export()
        search = next(r for r in rows if r["task"]["mode"] == "search")
        self.assertEqual(search["decision"]["citations"], [])
        self.assertFalse(search["citation_scoring_available"])
        vote = next(r["vote"] for r in rows if "vote" in r)
        self.assertEqual(vote["regions"], [])
        self.assertNotIn("annotation", vote)
        other = deepcopy(vote)
        other["reviewer_id"] = "independent"
        consensus = full_truth_consensus(
            [vote, other],
            list(self.study.cases.values()),
            self.study.plan["dataset_sha256"],
        )
        self.assertEqual(consensus["consensus"][0]["verdict"], "Supported")
        self.assertFalse(consensus["formal_evaluation_ready"])

    def test_truth_consensus_does_not_admit_release_sufficiency(self):
        from copy import deepcopy
        from resolveai.review import full_truth_consensus, validate_truth_vote

        self.study.post("reviewer", "consent", {"agree": True})
        self.study.post(
            "reviewer",
            "review",
            {
                "index": 0,
                "verdict": "Supported",
                "confidence": 3,
                "reason": "visible",
                "clear": True,
                "selected": [{"view_id": "original", "bbox": [0, 0, 32, 32]}],
            },
        )
        vote = self.study.export()[0]["vote"]
        cases = list(self.study.cases.values())
        one = full_truth_consensus([vote], cases, self.study.plan["dataset_sha256"])
        self.assertEqual(one["consensus"], [])
        other = deepcopy(vote)
        other["reviewer_id"] = "independent"
        two = full_truth_consensus(
            [vote, other], cases, self.study.plan["dataset_sha256"]
        )
        self.assertEqual(two["consensus"][0]["verdict"], "Supported")
        self.assertFalse(two["formal_evaluation_ready"])
        self.assertFalse(two["availability_review_complete"])
        other["initial_verdicts"] = {}
        with self.assertRaises(ValueError):
            validate_truth_vote(other)

    def test_preview_writes_only_to_its_separate_database(self):
        preview = self.root / "preview"
        preview.mkdir()
        (preview / "plan.json").write_text((self.root / "plan.json").read_text())
        (preview / "access.json").write_text(
            json.dumps(
                {
                    "try": {
                        "id": "TRY",
                        "role": "participant",
                        "tasks": self.study.actor("person")["tasks"],
                    }
                }
            )
        )
        server = create_server(self.root, 0)
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_port)
            headers = {
                "Authorization": "Bearer try",
                "Content-Type": "application/json",
            }
            for route, body in [
                ("consent", {"agree": True}),
                (
                    "answer",
                    {
                        "index": 0,
                        "verdict": "Need more evidence",
                        "selected": [],
                        "confidence": 3,
                        "reason": "uncertain",
                    },
                ),
            ]:
                body["ui_protocol"] = "human-ui-v6-optional-reason"
                conn.request("POST", "/api/" + route, json.dumps(body), headers)
                response = conn.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
            self.assertEqual(self.study.export(), [])
            import sqlite3

            db = sqlite3.connect(preview / "responses.sqlite")
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM answers").fetchone()[0], 1
            )
            db.close()
            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_full_material_scores_skip_missing_pools_and_keep_parse_failures(self):
        import importlib.util
        import sqlite3

        path = Path(__file__).resolve().parents[1] / "scripts/analyze_human_study.py"
        spec = importlib.util.spec_from_file_location("human_analysis", path)
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        models = self.root / "models"
        worker = models / "toy"
        worker.mkdir(parents=True)
        (worker / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": "toy",
                    "dataset_sha256": self.study.plan["dataset_sha256"],
                }
            )
        )
        db = sqlite3.connect(worker / "episodes.sqlite")
        db.execute("CREATE TABLE episodes(arm TEXT,result TEXT,trace TEXT)")
        for case in self.study.cases.values():
            decision = (
                None if case["variant"] == "Obtainable" else {"verdict": "Supported"}
            )
            db.execute(
                "INSERT INTO episodes VALUES (?,?,?)",
                (
                    "full_available_text",
                    json.dumps({"case_id": case["case_id"]}),
                    json.dumps({"decision": decision}),
                ),
            )
        db.commit()
        db.close()
        consensus = {
            "consensus": [
                {
                    "family_id": "f",
                    "verdict": "Supported",
                    "full_pool_ids": ["original", "preview", "reference"],
                }
            ]
        }
        result = analysis.full_material_agreement(
            models,
            list(self.study.cases.values()),
            consensus,
            self.study.plan["dataset_sha256"],
        )[0]
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["submitted"], 1)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["label_agreement"], 0.5)

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
            self.study.post("person", "consent", {"agree": True})
            conn.request(
                "GET",
                "/api/image/original",
                headers={"Authorization": "Bearer person"},
            )
            r = conn.getresponse()
            r.read()
            self.assertEqual(r.status, 403)
            conn.request("GET", "/", headers={})
            r = conn.getresponse()
            self.assertIn("看图与搜证", r.read().decode())
            self.assertEqual(r.status, 200)
            conn.request(
                "POST",
                "/api/answer",
                json.dumps(
                    {
                        "index": 0,
                        "verdict": "Need more evidence",
                        "selected": [],
                        "confidence": 3,
                        "reason": "old page",
                    }
                ),
                {"Content-Type": "application/json", "Authorization": "Bearer person"},
            )
            r = conn.getresponse()
            self.assertEqual(r.status, 400)
            self.assertIn("页面已更新", r.read().decode())
            self.assertEqual(self.study.export(), [])
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
