"""Sharding retains the same case/arm denominator and original timing traces."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from scripts.adaptation_shards import finalize
from scripts.score_candidate_labels import read_rows


class ShardTests(unittest.TestCase):
    def test_complete_disjoint_shards_are_indexed_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [{"case_id":f,"family_id":f} for f in ["a","b"]]
            payload = json.dumps(cases).encode()
            (root/"cases.json").write_bytes(payload)
            plan = {"data":str(root),"family_ids":["a","b"],
                    "dataset_sha256":hashlib.sha256(payload).hexdigest()}
            (root/"plan.json").write_text(json.dumps(plan))
            for index,family in enumerate(["a","b"]):
                shard = root/f"shard-{index}"
                shard.mkdir()
                config = {"shards":2,"shard":index,"families":[family],"cases":1,"arms":["initial_text"],
                          "dataset_sha256":plan["dataset_sha256"]}
                (shard/"manifest.json").write_text(json.dumps({"configuration":config,"worker":{"gpu":"same"}}))
                with sqlite3.connect(shard/"episodes.sqlite") as db:
                    db.execute("CREATE TABLE episodes(case_id,arm,result,trace)")
                    db.execute("INSERT INTO episodes VALUES(?,?,?,?)",
                               (family,"initial_text",json.dumps({**cases[index],"arm":"initial_text"}),'{"wall_latency_s":1.2}'))
            finalize(root,root/"plan.json",2)
            self.assertEqual({r["family_id"] for r in read_rows(root/"episodes.sqlite")},{"a","b"})
            self.assertFalse((root/"episodes.sqlite").exists())
            with sqlite3.connect(root/"shard-1/episodes.sqlite") as db:
                db.execute("DELETE FROM episodes")
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                finalize(root,root/"plan.json",2)

    def test_index_refuses_a_changed_shard_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root/"shard-0"
            shard.mkdir()
            (shard/"manifest.json").write_text("{}")
            entries = [{"directory":"shard-0","manifest_sha256":"incorrect"}]
            (root/"manifest.json").write_text(json.dumps({"aggregation":{"shards":entries}}))
            (root/"shard_index.json").write_text(json.dumps({"shards":entries}))
            with self.assertRaisesRegex(ValueError,"changed"):
                read_rows(root/"episodes.sqlite")
