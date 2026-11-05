"""Synthetic validator checks; no fixture is a real participant response."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest
SCORER=Path(__file__).with_name("score_blind40.py")
if not SCORER.exists():
 SCORER=Path(__file__).resolve().parents[1]/"scripts/score_blind_sufficiency_review.py"
API=runpy.run_path(str(SCORER))
class Checks(unittest.TestCase):
 def setUp(self):
  self.unit={"images":[{"id":"I1"}]}
  self.mapping={"package_id":"synthetic-package","units":{"synthetic-unit":self.unit},"families":{},
   "packets":{"A":{"unit_ids":["synthetic-unit"]},"B":{"unit_ids":["synthetic-unit"]}}}
  self.answer={"unit_id":"synthetic-unit","claim_clear":True,"sufficiency":"sufficient","answer":"Supported","citations":["I1"],"regions":[]}
 def payload(self,pid,answer):
  return {"format":"resolveai-blind-sufficiency-v1","package_id":"synthetic-package","packet_id":pid,
   "reviewer_id":"synthetic-"+pid,"reviewer_type":"human","independent_review":True,"answers":[answer]}
 def runscore(self,payloads,adj=()):
  with tempfile.TemporaryDirectory(prefix="resolveai-synthetic-review-")as d:
   paths=[]
   for i,p in enumerate(payloads):
    path=Path(d)/str(i);path.write_text(json.dumps(p));paths.append(path)
   return API["score"](self.mapping,paths,adj)
 def test_empty_is_explicitly_unreviewed(self):
  report,q=self.runscore([]);self.assertEqual(report["status"],"no_human_annotations");self.assertEqual(report["human_votes"],0)
 def test_consensus_requires_two_independent_votes(self):
  report,q=self.runscore([self.payload("A",self.answer)])
  self.assertEqual(report["consensus_materials"],0)
  report,q=self.runscore([self.payload("A",self.answer),self.payload("B",self.answer)])
  self.assertEqual(report["consensus_materials"],1);self.assertFalse(report["formal_grounded_accuracy_ready"])
 def test_unknown_image_rejected(self):
  with self.assertRaises(ValueError):API["validate_answer"]({**self.answer,"citations":["hidden"]},self.unit)
 def test_out_of_bounds_region_rejected(self):
  a={**self.answer,"regions":[{"image_id":"I1","bbox":[0,0,1.1,1]}]}
  with self.assertRaises(ValueError):API["validate_answer"](a,self.unit)
 def test_conflicting_sufficiency_rejected(self):
  with self.assertRaises(ValueError):API["validate_answer"]({**self.answer,"sufficiency":"insufficient"},self.unit)
 def test_disagreement_queued(self):
  report,q=self.runscore([self.payload("A",self.answer),self.payload("B",{**self.answer,"answer":"Refuted"})])
  self.assertEqual(report["consensus_materials"],0);self.assertEqual(len(q),1)
 def test_duplicate_packet_rejected(self):
  a=self.payload("A",self.answer)
  with self.assertRaises(ValueError):self.runscore([a,a])
 def test_repeated_person_rejected(self):
  a=self.payload("A",self.answer);b=self.payload("B",self.answer);b["reviewer_id"]=a["reviewer_id"]
  with self.assertRaises(ValueError):self.runscore([a,b])
 def test_adjudicator_must_be_third_person(self):
  a=self.payload("A",self.answer);b=self.payload("B",{**self.answer,"answer":"Refuted"})
  adj={**self.answer,"reviewer_type":"human","reviewer_id":"synthetic-A"}
  with self.assertRaises(ValueError):self.runscore([a,b],[adj])
if __name__=="__main__":unittest.main(verbosity=2)
