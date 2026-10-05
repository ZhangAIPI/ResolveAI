"""Semantic evidence-chain invariants, using software fixtures only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from resolveai.environment import Environment
from resolveai.conversation import Conversation, sft_example
from resolveai.tools import ActionError


def region(image, box=(2,2,6,6)):
    return {"image_id": image, "time": image, "bbox": list(box)}


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("a", "b"):
            Image.new("RGB", (16,16)).save(self.root/(name+".png"))
        self.link = {"relation": "same_object", "left": region("a"), "right": region("b")}
        self.case = {"claim": "same object, intact then scratched", "family_id": "fixture",
            "claim_parts": [{"id": "identity", "text": "same object"}], "initial": ["a","b"],
            "scene": {"identity": "SECRET"}, "evidence": [
                {"id": n, "source_id": n, "path": n+".png", "time": n,
                 "party": "A", "object": "item", "view": n, "available": True} for n in ("a","b")],
            "annotation": {"protocol": "evidence-chain-v1", "status": "reviewed", "verdict": "Supported",
                "subclaims": [
                    {"id":"identity", "kind":"identity", "truth":"Supported", "minimal_evidence_sets":{"Supported":[[self.link]]}},
                    {"id":"before", "kind":"state", "truth":"Supported", "minimal_evidence_sets":{"Supported":[[region("a")]]}},
                    {"id":"after", "kind":"state", "truth":"Supported", "minimal_evidence_sets":{"Supported":[[region("b")]]}}]}}
        self.decision = {"type":"finish", "verdict":"Supported", "citations":[region("a"),region("b")], "links":[self.link]}

    def test_all_regions_without_identity_link_are_insufficient(self):
        env = Environment(self.case,self.root)
        missing = deepcopy(self.decision); missing.pop("links")
        self.assertFalse(env.evaluate(missing)["grounded_correct"])
        self.assertTrue(env.evaluate(self.decision)["grounded_correct"])
        bad = deepcopy(self.decision); bad["links"][0]["relation"] = "different_object"
        self.assertTrue(env.evaluate(bad)["unsupported_decision"])
        missing = deepcopy(self.decision); missing["citations"].pop()
        self.assertFalse(env.evaluate(missing)["grounded_correct"])

    def test_identity_localization_and_symmetric_but_directed_time(self):
        env = Environment(self.case,self.root)
        bad = deepcopy(self.decision)
        bad["links"][0]["left"]["bbox"] = [0,0,16,16]
        self.assertFalse(env.evaluate(bad)["grounded_correct"])
        reverse = deepcopy(self.decision)
        reverse["links"][0]["left"],reverse["links"][0]["right"] = reverse["links"][0]["right"],reverse["links"][0]["left"]
        self.assertTrue(env.evaluate(reverse)["grounded_correct"])
        case = deepcopy(self.case); case["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0][0]["relation"] = "earlier_than"
        directed = deepcopy(self.decision); directed["links"][0]["relation"] = "earlier_than"
        self.assertTrue(Environment(case,self.root).evaluate(directed)["grounded_correct"])
        reverse["links"][0]["relation"] = "earlier_than"
        self.assertFalse(Environment(case,self.root).evaluate(reverse)["grounded_correct"])

    def test_decisive_refutation_uncertain_and_annotation_consistency(self):
        case = deepcopy(self.case); case["annotation"]["verdict"] = "Refuted"
        case["annotation"]["subclaims"][2].update(truth="Refuted",minimal_evidence_sets={"Refuted":[[region("b")]]})
        decision = {"verdict":"Refuted", "citations":[region("b")]}
        self.assertTrue(Environment(case,self.root).evaluate(decision)["grounded_correct"])
        case["annotation"]["subclaims"][2]["truth"] = "uncertain"
        with self.assertRaises(ValueError): Environment(case,self.root)
        case["annotation"]["verdict"] = "Need more evidence"
        env = Environment(case,self.root)
        self.assertFalse(env.evaluate(self.decision)["grounded_correct"])
        self.assertTrue(env.evaluate({"verdict":"Need more evidence","citations":[]})["grounded_correct"])

    def test_links_are_validated_and_truth_never_reaches_policy(self):
        session = Conversation(Environment(self.case,self.root))
        import json
        public = json.dumps(session.public())
        self.assertNotIn("SECRET",public); self.assertNotIn("minimal_evidence_sets",public)
        bad = deepcopy(self.decision); bad["links"][0]["right"]["time"] = "a"
        with self.assertRaises(ActionError): session._env.step(bad)
        self.assertFalse(session._env.finished)
        session.call({"name":"finish", "arguments":{k:v for k,v in self.decision.items() if k!="type"}})
        self.assertIsNotNone(sft_example(session.record()))
        case = deepcopy(self.case); case["annotation"]["status"] = "source-derived-unreviewed"
        session = Conversation(Environment(case,self.root))
        session.call({"name":"finish", "arguments":{k:v for k,v in self.decision.items() if k!="type"}})
        self.assertIsNone(sft_example(session.record()))

    def test_controlled_actions_preserve_world_prefix_and_parent(self):
        from resolveai.branching import compare_actions
        case = deepcopy(self.case); case["initial"] = ["a"]
        session = Conversation(Environment(case,self.root))
        finish = {"name":"finish", "arguments":{k:v for k,v in self.decision.items() if k!="type"}}
        class Policy:
            def ask_conversation(inner,payload,limit):
                import json
                assert "minimal_evidence_sets" not in json.dumps(payload)
                return {"tool_call":finish,"parse_error":False,"text":""}
        stop = {"name":"finish","arguments":{"verdict":"Need more evidence","citations":[]}}
        request = {"name":"request_photo","arguments":{"query":{"object":"item","time":"b","view":"b"}}}
        result = compare_actions(session,[request,stop],Policy(),max_turns=3)
        self.assertEqual(len(result["preferences"]),1)
        self.assertEqual(result["records"][0]["branch_audit"]["first_action_released"],["b"])
        self.assertEqual(result["records"][1]["branch_audit"]["first_action_released"],[])
        self.assertEqual(len(session.messages),2)
        self.assertEqual(session._env.budget,12)
        self.assertEqual(result["records"][0]["metadata"]["world_fingerprint"],result["records"][1]["metadata"]["world_fingerprint"])
