"""Lossless migration of completed episodes, including partial families."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from resolveai.repartition import repartition


class RepartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "model"
        (self.source / "images").mkdir(parents=True)
        (self.source / "images/pixel.png").write_bytes(b"pixel fixture")
        self.data = self.root / "data"
        self.data.mkdir()
        cases = [{"case_id":family+"-"+variant, "family_id":family}
                 for family in ("a","b","c") for variant in ("Sufficient","Obtainable","Missing","Unavailable")]
        raw = json.dumps(cases).encode()
        (self.data / "cases.json").write_bytes(raw)
        manifest = {"source_commit":"fixed", "slurm_job":"old",
            "configuration":{"dataset_sha256":hashlib.sha256(raw).hexdigest(),
                "shards":1, "selected_families":3, "limit_families":None,
                "policies":["initial","agent"], "budget":12, "revision":"fixed-weights"}}
        (self.source / "manifest.json").write_text(json.dumps(manifest))
        self.rows = []
        traces = []
        for index, case in enumerate(cases[:5]):
            for policy in ("initial","agent"):
                row = {**case, "policy":policy, "decision":None,
                       "termination":"max_turns", "wall_latency_s":index+1.0}
                self.rows.append(row)
                traces.append({"metadata":{**case,"policy":policy,"termination":"max_turns"},
                    "decision":None,"wall_latency_s":index+1.0,
                    "messages":[{"role":"user","content":[{"type":"image","image_ref":"images/pixel.png"}]}]})
        (self.source / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        (self.source / "trajectories.jsonl").write_text("".join(json.dumps(t)+"\n" for t in traces))
        self.output = self.root / "split"

    def test_preserves_completed_episodes_partial_families_and_image_bytes(self):
        report = repartition(self.source,self.data,self.output)
        seeded = []
        assigned = {}
        for index, name in enumerate(report["paths"]):
            folder = Path(name)
            manifest = json.loads((folder/"manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["budget"],12)
            self.assertEqual(manifest["configuration"]["revision"],"fixed-weights")
            self.assertEqual(manifest["configuration"]["shard_index"],index)
            rows = [json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
            seeded += rows
            self.assertEqual(len(rows),len((folder/"trajectories.jsonl").read_text().splitlines()))
            for row in rows:
                self.assertEqual(assigned.setdefault(row["family_id"],index),index)
            image = folder/"images/pixel.png"
            self.assertEqual(image.stat().st_ino,(self.source/"images/pixel.png").stat().st_ino)
            self.assertEqual(image.read_bytes(),b"pixel fixture")
        order = lambda row:(row["case_id"],row["policy"])
        self.assertEqual(sorted(seeded,key=order),sorted(self.rows,key=order))
        self.assertEqual(sum(report["shard_episodes"]),len(self.rows))
        self.assertEqual(len(self.rows),10)

    def test_rejects_changed_dataset_and_missing_matching_trace(self):
        data = self.data/"cases.json"
        original = data.read_bytes();data.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError,"dataset differs"):
            repartition(self.source,self.data,self.output)
        data.write_bytes(original)
        trace = self.source/"trajectories.jsonl"
        trace.write_text("\n".join(trace.read_text().splitlines()[:-1])+"\n")
        with self.assertRaisesRegex(ValueError,"no matching trajectory"):
            repartition(self.source,self.data,self.output)
        self.assertFalse(self.output.exists())

    def test_live_writer_change_aborts_without_publishing_shards(self):
        import os
        link = os.link
        def concurrent_link(source,destination):
            result = link(source,destination)
            with (self.source/"results.jsonl").open("a") as stream:
                stream.write("\n")
            return result
        with patch("resolveai.repartition.os.link",side_effect=concurrent_link):
            with self.assertRaisesRegex(ValueError,"source changed"):
                repartition(self.source,self.data,self.output)
        self.assertEqual(list(self.output.iterdir()),[])
