"""Budget-only resume keeps inference and completed answers fixed, without network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from resolveai.api_budget import ApiBudget, BudgetStop
from resolveai.api_resume import continue_budget_manifest, inference_code_digest
from scripts import resume_obtainable_api as driver


class PaidCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = ApiBudget(self.root/"parent.sqlite", 1)
        self.old = self.parent.allocate("old", self.root/"old.sqlite", .5)
        ticket, _ = self.old.reserve("paid-key", .2, stage_limit_usd=.5)
        self.old.settle(ticket, .15, {"id": "saved", "usage": {"input_tokens": 2}})
        self.parent.close_allocation("old", self.root/"old.sqlite")
        self.old.close()
        self.new = self.parent.allocate("new", self.root/"new.sqlite", .4)

    def tearDown(self):
        self.new.close()
        self.parent.close()
        self.temp.cleanup()

    def test_paid_cache_has_zero_new_charge_and_old_ledger_is_immutable(self):
        old_hash = hashlib.sha256((self.root/"old.sqlite").read_bytes()).hexdigest()
        audit = self.new.import_paid_cache(self.root/"old.sqlite")
        self.assertEqual(audit["imported_charge_usd"], 0)
        self.assertEqual(audit["imported_responses"], 1)
        ticket, response = self.new.reserve("paid-key", .3, stage_limit_usd=.4)
        self.assertIsNone(ticket)
        self.assertEqual(response["id"], "saved")
        self.assertEqual(self.new.status()["accounted_upper_usd"], 0)
        self.assertEqual(old_hash, hashlib.sha256((self.root/"old.sqlite").read_bytes()).hexdigest())
        self.assertEqual(self.new.import_paid_cache(self.root/"old.sqlite")["imported_responses"], 0)
        with self.assertRaises(BudgetStop):
            self.new.reserve("new-too-large", .41, stage_limit_usd=.4)
        ticket, _ = self.new.reserve("new-fits", .4, stage_limit_usd=.4)
        self.new.settle(ticket, .3, {"id": "new"})
        self.parent.close_allocation("new", self.root/"new.sqlite")
        self.assertAlmostEqual(self.parent.status()["accounted_upper_usd"], .45)

    def test_unsealed_or_unresolved_source_is_rejected(self):
        source = self.parent.allocate("inflight", self.root/"inflight.sqlite", .2)
        try:
            with self.assertRaises(BudgetStop):
                self.new.import_paid_cache(self.root/"inflight.sqlite")
            source.reserve("unknown", .1, stage_limit_usd=.2)
            self.parent.close_allocation("inflight", self.root/"inflight.sqlite")
            with self.assertRaises(BudgetStop):
                self.new.import_paid_cache(self.root/"inflight.sqlite")
        finally:
            source.close()

    def test_conflicting_cached_response_is_rejected(self):
        ticket, _ = self.new.reserve("paid-key", .1, stage_limit_usd=.4)
        self.new.settle(ticket, .1, {"id": "different"})
        with self.assertRaises(ValueError):
            self.new.import_paid_cache(self.root/"old.sqlite")


class BudgetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_repo, self.new_repo = self.root/"old", self.root/"new"
        script = ("def select_cases(): pass\ndef configure_api_key(): pass\n"
                  "def main():\n    ledger = ApiBudget()\n    execute_model()\n")
        for repo in [self.old_repo, self.new_repo]:
            (repo/"src/resolveai").mkdir(parents=True)
            (repo/"scripts").mkdir()
            (repo/"src/resolveai/openai_client.py").write_text("FIXED_MODEL = 1\n")
            (repo/"src/resolveai/api_budget.py").write_text("FINANCE = 1\n")
            (repo/"src/resolveai/api_resume.py").write_text("AUDIT = 1\n")
            (repo/"scripts/evaluate_study.py").write_text("FIXED_TOOLS = 1\n")
            (repo/"scripts/evaluate_openai.py").write_text(script)
        source = hashlib.sha256()
        for path in sorted((self.old_repo/"src/resolveai").glob("*.py")):
            source.update(path.name.encode()); source.update(path.read_bytes())
        self.output = self.root/"output"
        self.output.mkdir()
        self.path = self.output/"manifest.json"
        self.old = {"model_id": "fixed", "family_ids": ["family"],
                    "budget": 12, "max_turns": 8, "ledger": "old.sqlite",
                    "ledger_cap_usd": .5, "stage_cap_usd": .5, "budget_stage": "old",
                    "source_sha256": source.hexdigest(),
                    "script_sha256": hashlib.sha256(script.encode()).hexdigest()}
        self.path.write_text(json.dumps({"configuration": self.old, "source_commit": "frozen"}))
        with sqlite3.connect(self.output/"episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes(result TEXT,trace TEXT)")
            db.execute("INSERT INTO episodes VALUES (?,?)", ('{"answer":"saved"}', '{"messages":[]}'))
        self.new = {**self.old, "ledger": str(self.output/"resume.sqlite"),
                    "ledger_cap_usd": .4, "stage_cap_usd": .4, "budget_stage": "new",
                    "source_sha256": "finance-only-update", "script_sha256": "cli-only-update"}

    def tearDown(self):
        self.temp.cleanup()

    def fake_lock(self, path, configuration):
        result = {"configuration": configuration, "source_commit": "resumed"}
        path.write_text(json.dumps(result))
        return result

    @patch("resolveai.api_resume.locked_manifest")
    def test_finance_changes_preserve_answers_and_archive_exact_manifest(self, lock):
        lock.side_effect = self.fake_lock
        original = self.path.read_bytes()
        (self.new_repo/"src/resolveai/api_budget.py").write_text("FINANCE = 2\n")
        result = continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)
        continuation = result["configuration"]["continuation"]
        self.assertEqual(continuation["preserved_answers"], 1)
        self.assertIn("Budget reallocation", continuation["reason"])
        self.assertNotIn("Client", continuation["reason"])
        self.assertEqual(self.path.with_name("manifest.before-budget-resume.json").read_bytes(), original)
        again = continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)
        self.assertEqual(again, result)

    def test_task_model_tools_and_actual_inference_changes_are_rejected(self):
        for key, value in [("model_id", "other"), ("budget", 13), ("family_ids", ["other"]), ("max_turns", 9)]:
            with self.assertRaises(ValueError):
                continue_budget_manifest(self.path, {**self.new, key: value}, self.old_repo, self.new_repo)
        (self.new_repo/"src/resolveai/openai_client.py").write_text("FIXED_MODEL = 2\n")
        with self.assertRaisesRegex(ValueError, "inference code"):
            continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)

    def test_generation_loop_change_is_rejected(self):
        path = self.new_repo/"scripts/evaluate_openai.py"
        path.write_text(path.read_text().replace("execute_model()", "different_call()"))
        with self.assertRaisesRegex(ValueError, "inference code"):
            continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)

    @patch("resolveai.api_resume.locked_manifest")
    def test_modified_completed_answer_is_rejected_on_resume(self, lock):
        lock.side_effect = self.fake_lock
        continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)
        with sqlite3.connect(self.output/"episodes.sqlite") as db:
            db.execute("UPDATE episodes SET result='changed'")
        with self.assertRaisesRegex(ValueError, "completed answer changed"):
            continue_budget_manifest(self.path, self.new, self.old_repo, self.new_repo)


class ContinuationDriverTests(unittest.TestCase):
    def test_detected_answer_change_never_marks_a_complete_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, runtime = root/"repo", root/"runtime"
            (repo/"docs/results").mkdir(parents=True)
            (repo/"docs/results/cvpr_20261102_candidate_scores.json").write_text("{}")
            (runtime/"evaluation/cvpr-20261101").mkdir(parents=True)
            (runtime/"evaluation/cvpr-20261101/plan.json").write_text("{}")
            output = runtime/"evaluation/cvpr-paper-priority/reviewer-api-obtainable"
            output.mkdir(parents=True)
            child = SimpleNamespace(import_paid_cache=lambda path: {}, close=lambda: None)
            parent = SimpleNamespace(allocate=lambda *args: child, close=lambda: None,
                                     close_allocation=lambda *args: {"accounted_upper_usd": 0})
            cases = [{"task": "state" if i < 153 else "identity"} for i in range(209)]
            args = SimpleNamespace(repository=repo, runtime=runtime, execute=True)
            with patch.object(driver.argparse.ArgumentParser, "parse_args", return_value=args), \
                 patch.dict(driver.os.environ, {"SLURMD_NODENAME": "macula"}), \
                 patch.object(driver, "PROFILES", (("mini", .25),)), \
                 patch.object(driver, "ledger_status", return_value={"accounted_upper_usd": 0}), \
                 patch.object(driver, "select_cases", return_value=cases), \
                 patch.object(driver, "saved_count", return_value=209), \
                 patch.object(driver, "answer_snapshot", return_value={"rows": [], "images": {}}), \
                 patch.object(driver, "verify_snapshot", side_effect=ValueError("answer changed")), \
                 patch.object(driver, "ApiBudget", return_value=parent), \
                 patch.object(driver.subprocess, "check_output", return_value="fixed\n"), \
                 patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                (output/"mini").mkdir()
                with self.assertRaises(ValueError):
                    driver.main()
            state = json.loads((output/"budget-resume-20261106-status.json").read_text())
            self.assertEqual(state["models"]["mini"]["phase"], "stopped")
            self.assertFalse(state["models"]["mini"]["original_answers_unchanged"])
            self.assertEqual(state["models"]["mini"]["stop_reason"], "ValueError")


if __name__ == "__main__":
    unittest.main()
