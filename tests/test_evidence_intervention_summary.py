"""Small CPU fixtures exercise complete-key gates and decoded-pixel parity."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from PIL import Image

path = Path(__file__).with_name("summarize_evidence_interventions.py")
if not path.exists():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_evidence_interventions.py"
spec = importlib.util.spec_from_file_location("staged_intervention_summary", path)
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)


class InterventionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.families = ["f-state", "f-identity"]
        cases = [{"case_id": f + "-" + v, "family_id": f, "variant": v,
                  "task": "state" if f.endswith("state") else "identity",
                  "provenance_truth": "Supported"}
                 for f in self.families for v in summary.VARIANTS]
        (self.root / "cases.json").write_text(json.dumps(cases))
        self.plan = {"data": str(self.root), "dataset_sha256": summary.sha256(self.root / "cases.json"),
                     "family_ids": self.families, "budget": 3, "max_turns": 8}
        self.cohort = {"dataset_sha256": self.plan["dataset_sha256"],
                       "family_ids": self.families, "_file_sha256": "cohort"}
        self.new, self.old = self.root / "new", self.root / "old"
        self.build(self.new, summary.NEW_ARMS, ("Obtainable",), "v2")
        self.build(self.old, summary.OLD_ARMS, summary.VARIANTS, "v1")

    def tearDown(self):
        self.temp.cleanup()

    def directory(self, root, model, shard):
        return root / model / "budget3" / ("shard-" + str(shard))

    def build(self, root, arms, variants, version):
        for model in summary.MODELS:
            for shard, family in enumerate(self.families):
                directory = self.directory(root, model, shard)
                directory.mkdir(parents=True)
                config = {
                    "families": [family], "shards": 2, "shard": shard,
                    "arms": list(arms), "variants": list(variants),
                    "cases": len(variants), "expected_episodes": len(variants) * len(arms),
                    "dataset_sha256": self.plan["dataset_sha256"], "cohort_sha256": "cohort",
                    "source_sha256": "core", "storage_script_sha256": "storage",
                    "terminal_script_sha256": "terminal", "script_sha256": version,
                    "experiment": version, "seed": 20261105, "budget": 3, "max_turns": 8,
                    "max_new_tokens": 384, "max_context_tokens": 8192, "image_max_pixels": 262144,
                    "tool_adapter": "auto", "assistant_prefix": True, "grounding_manifest": {},
                    "model": {"model_id": "8b", "base_revision": "base",
                              "adapter_sha256": "adapter" if model.endswith("sftrl") else None},
                }
                (directory / "manifest.json").write_text(json.dumps(
                    {"configuration": config, "source_commit": version, "worker": {"gpu": "mock"}}))
                with sqlite3.connect(directory / "episodes.sqlite") as db:
                    db.execute("CREATE TABLE episodes(case_id TEXT,arm TEXT,result TEXT,trace TEXT,PRIMARY KEY(case_id,arm))")
                    for variant in variants:
                        for arm in arms:
                            cid = family + "-" + variant
                            row = {
                                "case_id": cid, "family_id": family, "variant": variant, "arm": arm,
                                "task": "state" if family.endswith("state") else "identity",
                                "source_label": "Supported", "source_label_match": False,
                                "verdict": None, "termination": "context_limit", "requests": 0,
                                "request_cost": 0, "tool_cost": 0, "assistant_turns": 1,
                                "latency_s": 1.5, "input_tokens": 3, "output_tokens": 0,
                            }
                            trace = {
                                "metadata": {"case_id": cid, "arm": arm}, "decision": None,
                                "messages": [{"role": "system", "content": "native system"},
                                             {"role": "user", "content": "same photographs"}],
                                "tools": [{"function": {"name": "finish"}}],
                                "generation": [{"halt": "context_limit", "input_tokens": 3}],
                                "wall_latency_s": 1.5,
                            }
                            db.execute("INSERT INTO episodes VALUES(?,?,?,?)",
                                       (cid, arm, json.dumps(row), json.dumps(trace)))

    def collect(self):
        return summary.collect(self.plan, self.cohort, self.new, self.old, 2)

    def test_complete_join_retains_failures_and_null_reviewed_metrics(self):
        report, _ = self.collect()
        self.assertTrue(report["complete_v2"])
        self.assertTrue(report["joined_reference_complete"])
        self.assertEqual(report["native_initial_input_parity"][summary.MODELS[0]]["families"], 2)
        value = report["models"][summary.MODELS[0]]["adaptive_no_new_photo"]["overall"]
        self.assertEqual(value["episodes"], 2)
        self.assertEqual(value["source_label_accuracy_pct"], 0)
        self.assertEqual(value["finished_pct"], 0)
        self.assertEqual(value["efficiency"]["mean_input_tokens"], 3)
        self.assertTrue(all(v is None for v in value["reviewed_metrics"].values()))
        self.assertIn("Full available", summary.render(report))

    def test_incomplete_v2_exposes_no_scores(self):
        path = self.directory(self.new, summary.MODELS[0], 0) / "episodes.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM episodes WHERE arm=?", (summary.NEW_ARMS[0],))
        report, progress = self.collect()
        self.assertIsNone(report)
        self.assertLess(progress["v2"][summary.MODELS[0] + "/0"]["completed"],
                        progress["v2"][summary.MODELS[0] + "/0"]["expected"])

    def test_incomplete_v1_does_not_publish_a_join_even_if_adaptive_obtainable_is_complete(self):
        path = self.directory(self.old, summary.MODELS[0], 0) / "episodes.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM episodes WHERE case_id=? AND arm=?",
                       (self.families[0] + "-Missing", "finish_only"))
        report, _ = self.collect()
        self.assertTrue(report["complete_v2"])
        self.assertFalse(report["joined_reference_complete"])
        with self.assertRaises(ValueError):
            summary.render(report)

    def test_extra_unplanned_key_is_rejected(self):
        path = self.directory(self.new, summary.MODELS[0], 0) / "episodes.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("INSERT INTO episodes VALUES('unexpected','rogue','{}','{}')")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.collect()

    def test_source_or_weights_mismatch_is_rejected(self):
        path = self.directory(self.new, summary.MODELS[0], 1) / "manifest.json"
        value = json.loads(path.read_text())
        value["configuration"]["model"]["adapter_sha256"] = "changed"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Shard configurations"):
            self.collect()

    def test_paired_native_prompt_change_is_rejected(self):
        path = self.directory(self.old, summary.MODELS[0], 0) / "episodes.sqlite"
        with sqlite3.connect(path) as db:
            cid = self.families[0] + "-Obtainable"
            raw = db.execute("SELECT trace FROM episodes WHERE case_id=? AND arm='adaptive'", (cid,)).fetchone()[0]
            trace = json.loads(raw)
            trace["messages"][0]["content"] = "different system"
            db.execute("UPDATE episodes SET trace=? WHERE case_id=? AND arm='adaptive'",
                       (json.dumps(trace), cid))
        with self.assertRaisesRegex(ValueError, "initial prompts"):
            self.collect()

    def test_initial_input_compares_decoded_rgb_not_png_encoding_or_filename(self):
        helper, _ = summary.helpers()
        a, b = self.root / "a", self.root / "b"
        a.mkdir()
        b.mkdir()
        Image.new("RGB", (8, 8), "red").save(a / "raw.png", compress_level=0)
        Image.new("RGB", (8, 8), "red").save(b / "compressed.png", compress_level=9)
        Image.new("RGB", (8, 8), "blue").save(b / "other.png")
        def item(directory, name):
            return {"directory": directory, "trace": {"messages": [
                {"role": "user", "content": [{"type": "image", "image_ref": name}]}], "tools": []}}
        first = summary.initial_input(item(a, "raw.png"), helper)
        self.assertEqual(first, summary.initial_input(item(b, "compressed.png"), helper))
        self.assertNotEqual(first, summary.initial_input(item(b, "other.png"), helper))


if __name__ == "__main__":
    unittest.main()
