"""CPU-only integrity, failure-preservation and rollback tests for cohort reuse."""
from copy import deepcopy
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from test_multistep import fixture
from resolveai.benchmark import source_digest
from resolveai.multistep import evaluate
from scripts.evaluate_multistep import (
    POLICIES, asset_fingerprint, canonical, model_fingerprint, sha_file, sha_value,
)
from scripts.reuse_multistep_cohort import reuse


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset = self.root / "assets"
        self.asset.mkdir()
        (self.asset / "asset.png").write_bytes(b"unchanged frozen evidence bytes")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"tiny checkpoint fixture")
        self.cases = []
        self.decisions, self.tokens = {}, {}
        for i in range(4):
            case, decision, tokens = fixture(positive=i % 2 == 0)
            case.update(case_id=f"case-{i}", family_id=f"family-{i}", variant="Obtainable",
                        asset_root=str(self.asset), evidence=[{"path": "asset.png"}],
                        multistep_regions={"frozen": True})
            self.cases.append(case)
            self.decisions[case["case_id"]] = decision
            self.tokens[case["case_id"]] = tokens
        self.old_data = self.data("pilot", self.cases[:2])
        self.new_data = self.data("full", self.cases)
        self.source = self.make_run("source", self.old_data, self.cases[:2], records=True)
        self.target = self.make_run("target", self.new_data, self.cases, shard=0, shards=2)
        self.source_before = self.tree_hashes(self.source)

    def data(self, name, cases):
        directory = self.root / name
        directory.mkdir()
        path = directory / "cases.json"
        path.write_text(json.dumps(cases))
        (directory / "dataset_manifest.json").write_text(json.dumps({
            "dataset_sha256": sha_file(path), "annotation_status": "procedural"}))
        return path

    def make_run(self, name, data, cases, *, shard=0, shards=1, records=False):
        directory = self.root / name
        directory.mkdir()
        (directory / "run.lock").touch()
        families = sorted({c["family_id"] for c in cases})[shard::shards]
        selected = [c for c in cases if c["family_id"] in families]
        runner = Path(__file__).resolve().parents[1] / "scripts/evaluate_multistep.py"
        config = {
            "experiment": "procedural-multistep-two-request-v2",
            "dataset": str(data), "dataset_sha256": sha_file(data),
            "assets": asset_fingerprint(cases, data.parent),
            "model": model_fingerprint(self.model, None),
            "model_id": "fixture", "revision": "frozen",
            "source_sha256": source_digest(), "script_sha256": sha_file(runner),
            "families": families, "case_ids": [c["case_id"] for c in selected],
            "policies": list(POLICIES), "variants": ["Obtainable"],
            "shard": shard, "shards": shards, "budget": 6,
            "max_turns": 8, "max_new_tokens": 768, "max_context_tokens": 8192,
            "seed": 20261107, "do_sample": False, "request_slots": 2,
        }
        (directory / "manifest.json").write_text(json.dumps({"configuration": config,
                                                          "versions": {"Pillow": "fixture"}}))
        (directory / "summary.json").write_text(json.dumps({"preflight": True, "model_loaded": False}))
        with sqlite3.connect(directory / "episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes(case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
            if records:
                (directory / "images").mkdir()
                for case in selected:
                    blob = b"immutable recorded pixels " + case["case_id"].encode()
                    blob_hash = sha256(blob).hexdigest()
                    (directory / "images" / (blob_hash + ".png")).write_bytes(blob)
                    for policy in POLICIES:
                        # Invalid native terminal submissions are retained with their exact costs.
                        decision = None if policy == "adaptive_acquisition" else self.decisions[case["case_id"]]
                        score = evaluate(case, decision, self.tokens[case["case_id"]])
                        trace = {
                            "decision": decision, "observation_tokens": self.tokens[case["case_id"]],
                            "multistep_evaluation": score,
                            "messages": [{"content": [{"type": "image", "image_ref": "images/" + blob_hash + ".png"}]}],
                            "acquisition_messages": [{"content": [{"type": "image", "image_ref": "images/" + blob_hash + ".png"}]}],
                        }
                        row = {"case_id": case["case_id"], "family_id": case["family_id"],
                               "policy": policy, "multistep_evaluation": score,
                               "finished": decision is not None, "wall_latency_s": 7.25,
                               "input_tokens": 123, "output_tokens": 41,
                               "initial_public_sha256": sha_value(case["case_id"]),
                               "trace_sha256": sha_value(trace)}
                        # Noncanonical spacing detects accidental deserialize/reserialize.
                        db.execute("INSERT INTO episodes VALUES(?,?,?,?)", (
                            case["case_id"], policy, json.dumps(row, indent=1), json.dumps(trace, indent=2)))
        return directory

    def tree_hashes(self, directory):
        return {str(p.relative_to(directory)): sha_file(p) for p in directory.rglob("*") if p.is_file()}

    def change_manifest(self, directory, change):
        path = directory / "manifest.json"
        value = json.loads(path.read_text())
        change(value["configuration"])
        path.write_text(json.dumps(value))

    def rows(self, directory):
        with sqlite3.connect(directory / "episodes.sqlite") as db:
            return list(db.execute("SELECT * FROM episodes ORDER BY case_id,policy"))

    def test_exact_sql_text_pixels_failures_and_shard_filter_preserved(self):
        audit = reuse(self.source, self.target)
        self.assertEqual(audit["source_complete_episodes"], 10)
        self.assertEqual(audit["copied_episodes"], 5)
        expected = [r for r in self.rows(self.source) if r[0] == "case-0"]
        self.assertEqual(self.rows(self.target), expected)
        failed = json.loads(next(r[2] for r in expected if r[1] == "adaptive_acquisition"))
        self.assertFalse(failed["finished"])
        self.assertEqual(failed["wall_latency_s"], 7.25)
        for ref in audit["blobs"]:
            self.assertEqual((self.source / ref).read_bytes(), (self.target / ref).read_bytes())
        self.assertEqual(self.tree_hashes(self.source), self.source_before)
        self.assertEqual(len(audit["rows"]), 5)
        with self.assertRaisesRegex(ValueError, "already has"):
            reuse(self.source, self.target)

    def test_incomplete_source_cannot_select_only_successful_rows(self):
        with sqlite3.connect(self.source / "episodes.sqlite") as db:
            db.execute("DELETE FROM episodes WHERE case_id='case-1' AND policy='oracle'")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            reuse(self.source, self.target)
        self.assertEqual(self.rows(self.target), [])

    def test_partial_choice_requires_flag_same_dataset_and_retains_failures(self):
        target = self.make_run("same-data-target", self.old_data, self.cases[:2], shard=0, shards=2)
        runner = Path(__file__).resolve().parents[1] / "scripts/evaluate_multistep_choice.py"
        native = runner.with_name("evaluate_multistep.py")
        for directory in (self.source, target):
            self.change_manifest(directory, lambda c: c.update(
                experiment="procedural-multistep-interface-assisted-choice-v1",
                script_sha256=sha_file(runner), shared_native_script_sha256=sha_file(native),
                interface_scope="interface-assisted-finite-choice-diagnostic"))
        with sqlite3.connect(self.source / "episodes.sqlite") as db:
            db.execute("DELETE FROM episodes WHERE case_id='case-1' AND policy='oracle'")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            reuse(self.source, target)
        with self.assertRaisesRegex(ValueError, "same complete dataset"):
            # This flag never licenses mixing a partial pilot into a new cohort.
            self.change_manifest(self.target, lambda c: c.update(
                experiment="procedural-multistep-interface-assisted-choice-v1",
                script_sha256=sha_file(runner), shared_native_script_sha256=sha_file(native),
                interface_scope="interface-assisted-finite-choice-diagnostic"))
            reuse(self.source, self.target, allow_partial_source=True)
        audit = reuse(self.source, target, allow_partial_source=True)
        self.assertFalse(audit["source_complete"])
        self.assertEqual((audit["source_rows_actual"], audit["source_rows_expected"]), (9, 10))
        self.assertEqual(audit["copied_episodes"], 5)
        self.assertEqual(self.rows(target), [r for r in self.rows(self.source) if r[0] == "case-0"])
        self.assertFalse(json.loads(next(r[2] for r in self.rows(target) if r[1] == "adaptive_acquisition"))["finished"])

    def test_changed_case_even_outside_target_shard_is_rejected(self):
        full = deepcopy(self.cases)
        full[1]["claim"] = "changed text"
        self.new_data.write_text(json.dumps(full))
        manifest = self.new_data.parent / "dataset_manifest.json"
        manifest.write_text(json.dumps({"dataset_sha256": sha_file(self.new_data), "annotation_status": "procedural"}))
        self.change_manifest(self.target, lambda c: c.update(dataset_sha256=sha_file(self.new_data)))
        with self.assertRaisesRegex(ValueError, "changed an existing case"):
            reuse(self.source, self.target)

    def test_setting_source_weight_or_asset_mutation_rejected(self):
        self.change_manifest(self.target, lambda c: c.update(max_new_tokens=384))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            reuse(self.source, self.target)
        self.change_manifest(self.target, lambda c: c.update(max_new_tokens=768))
        (self.model / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            reuse(self.source, self.target)
        (self.model / "config.json").write_text("{}")
        (self.asset / "asset.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "asset bytes changed"):
            reuse(self.source, self.target)

    def test_symlink_and_mutated_recorded_blob_rejected(self):
        path = next((self.source / "images").iterdir())
        original = path.read_bytes()
        moved = self.root / "outside.png"
        moved.write_bytes(original)
        path.unlink()
        path.symlink_to(moved)
        with self.assertRaises(ValueError):
            reuse(self.source, self.target)
        path.unlink()
        path.write_bytes(b"changed pixels")
        with self.assertRaises(ValueError):
            reuse(self.source, self.target)

    def test_mid_copy_oserror_removes_partial_blob_and_preserves_original_error(self):
        before = self.tree_hashes(self.target)
        def interrupted_copy(inp, out):
            out.write(inp.read(3))
            raise OSError("injected mid-copy failure")
        with patch("scripts.reuse_multistep_cohort.shutil.copyfileobj", side_effect=interrupted_copy):
            with self.assertRaisesRegex(OSError, "injected mid-copy failure"):
                reuse(self.source, self.target)
        self.assertEqual(self.rows(self.target), [])
        self.assertEqual(self.tree_hashes(self.target), before)
        self.assertEqual(self.tree_hashes(self.source), self.source_before)

    def test_transaction_failure_rolls_back_rows_audit_and_copied_blobs(self):
        from scripts.reuse_multistep_cohort import validate_saved
        calls = []
        def fail_on_target(*args):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError("injected target score disagreement")
            return validate_saved(*args)
        with patch("scripts.reuse_multistep_cohort.validate_saved", side_effect=fail_on_target):
            with self.assertRaisesRegex(ValueError, "injected target"):
                reuse(self.source, self.target)
        self.assertEqual(self.rows(self.target), [])
        self.assertFalse((self.target / "images").exists())
        self.assertFalse((self.target / "reuse_origin.json").exists())
        self.assertEqual(self.tree_hashes(self.source), self.source_before)

    def test_live_target_lock_nonempty_target_and_changed_trace_rejected(self):
        with (self.target / "run.lock").open("r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                reuse(self.source, self.target)
        with sqlite3.connect(self.source / "episodes.sqlite") as db:
            db.execute("UPDATE episodes SET trace='{}' WHERE case_id='case-0' AND policy='oracle'")
        with self.assertRaisesRegex(ValueError, "trace changed"):
            reuse(self.source, self.target)
        self.assertEqual(self.rows(self.target), [])


if __name__ == "__main__":
    unittest.main()
