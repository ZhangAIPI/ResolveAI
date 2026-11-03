"""Continuation preserves completed answers and rejects evaluation changes."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from resolveai.api_resume import continue_manifest

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/"manifest.json"
        self.old={"model_id":"fixed-model","family_ids":["a"],"budget":12,
                  "ledger":"old.sqlite","source_sha256":"old"}
        self.path.write_text(json.dumps({"configuration":self.old,"source_commit":"original"}))
        with sqlite3.connect(self.path.parent/"episodes.sqlite") as db:
            db.execute("CREATE TABLE episodes(result TEXT,trace TEXT)")
            db.execute("INSERT INTO episodes VALUES (?,?)",('{"answer":"original"}','{"image":"same"}'))
    def tearDown(self):
        self.temp.cleanup()
    def lock(self,path,configuration):
        manifest={"configuration":configuration,"source_commit":"patched"}
        path.write_text(json.dumps(manifest))
        return manifest
    @patch("resolveai.api_resume.locked_manifest")
    def test_completed_answers_and_original_source_are_retained(self,lock):
        lock.side_effect=self.lock
        new={**self.old,"ledger":"new.sqlite","source_sha256":"new"}
        result=continue_manifest(self.path,new)
        self.assertEqual(result["configuration"]["continuation"]["preserved_answers"],1)
        self.assertEqual(result["configuration"]["continuation"]["previous_source_commit"],"original")
        archived=json.loads(self.path.with_name("manifest.before-client-fix.json").read_text())
        self.assertEqual(archived["configuration"],self.old)
    def test_model_or_budget_change_is_rejected(self):
        for field,value in [("model_id","different"),("budget",20),("family_ids",["b"])]:
            with self.assertRaises(ValueError):
                continue_manifest(self.path,{**self.old,field:value})
    @patch("resolveai.api_resume.locked_manifest")
    def test_modified_completed_answer_is_rejected(self,lock):
        lock.side_effect=self.lock
        new={**self.old,"ledger":"new.sqlite","source_sha256":"new"}
        continue_manifest(self.path,new)
        with sqlite3.connect(self.path.parent/"episodes.sqlite") as db:
            db.execute("UPDATE episodes SET result='changed'")
        with self.assertRaises(ValueError):
            continue_manifest(self.path,new)
