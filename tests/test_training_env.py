"""Training invariants on software fixtures, not research cases."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from resolveai.conversation import Conversation, TrajectoryStore, preference_pair, sft_example
from resolveai.environment import Environment
from resolveai.rollout import run
from resolveai.tools import ActionError
from resolveai.vlm_worker import parse_tool_call


class OCRFixture:
    available = True
    def read(self, image):
        assert isinstance(image, Image.Image)
        return [{"text":"fixture", "confidence":.9, "polygon":[[0,0],[2,0],[2,2],[0,2]]}]


class GroundingFixture:
    def available(self, kind): return True
    def text_image(self, image, text):
        assert isinstance(image,Image.Image)
        return [{"bbox":[0,0,2,2],"text":text,"score":.8}]
    def image_image(self, query, target, top_k):
        assert isinstance(query,Image.Image) and isinstance(target,Image.Image)
        return [{"bbox":[0,0,2,2],"similarity":.5}]
    def image_text(self, image, candidates):
        return [{"text":candidates[0],"similarity":.4}]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        image=Image.new("RGB",(16,16))
        image.putdata([(255,255,255) if (x+y)%2 else (0,0,0) for y in range(16) for x in range(16)])
        image.save(self.root/"before.png");image.save(self.root/"after.png")
        self.case={"case_id":"fixture","family_id":"fixture-family", "claim":"fixture conjunction",
            "scene":{"secret_condition":"PRIVATE_STATE_MARKER"}, "initial":["before"],
            "request_options":{"objects":["table"],"times":["before","after"],"views":["front","side"]},
            "evidence":[{"id":i,"source_id":i,"path":i+".png","party":"A","object":"table",
                         "time":i,"view":"front" if i=="before" else "side","available":True} for i in ["before","after"]],
            "annotation":{"status":"reviewed","verdict":"Supported","minimal_evidence_sets":{
                "Supported":[[{"image_id":"before","time":"before","bbox":[0,0,8,8]},
                              {"image_id":"after","time":"after","bbox":[4,4,12,12]}]],
                "Refuted":[[{"image_id":"before","time":"before","bbox":[0,0,4,4]}]]}}}
        self.env=Environment(self.case,self.root,24,ocr_backend=OCRFixture(),grounding_backend=GroundingFixture())

    def request(self):
        return {"name":"request_photo","arguments":{"query":{"object":"table","time":"after","view":"side"}}}

    def finish(self, include_after=True):
        citations=[{"image_id":"before","time":"before","bbox":[0,0,8,8]}]
        if include_after: citations.append({"image_id":"after","time":"after","bbox":[4,4,12,12]})
        return {"name":"finish","arguments":{"verdict":"Supported","citations":citations}}

    def test_complete_native_tool_history_and_all_optional_tools(self):
        session=Conversation(self.env)
        session.call({"name":"inspect","arguments":{"image_id":"before"}})
        zoom=session.call({"name":"zoom","arguments":{"image_id":"before","factor":2}})["images"][0]
        crop=session.call({"name":"crop","arguments":{"image_id":zoom["view_id"],"bbox":[4,6,16,18]}})["images"][0]
        self.assertEqual(crop["source_bbox"],[2,3,8,9])
        ocr=session.call({"name":"ocr","arguments":{"image_id":crop["view_id"]}})
        self.assertEqual(ocr["text_regions"][0]["source_polygon"][0],[2,3])
        for name in ["assess_quality","read_metadata"]:
            self.assertNotIn("error",session.call({"name":name,"arguments":{"image_id":crop["view_id"]}}))
        for name,args in [("ground_text_to_image",{"image_id":crop["view_id"],"text":"table"}),
                          ("ground_image_to_image",{"query_image_id":crop["view_id"],"target_image_id":"before"}),
                          ("ground_image_to_text",{"image_id":crop["view_id"],"candidates":["table"]})]:
            result=session.call({"name":name,"arguments":args})
            self.assertEqual(result["status"],"predicted")
            self.assertIn("source_bbox",result["proposals"][0])
        session.call(self.request())
        session.call({"name":"compare","arguments":{"image_ids":["before","after"]}})
        session.call(self.finish())
        self.assertTrue(session.record()["evaluation"]["grounded_correct"])
        self.assertEqual(self.env.budget,4)
        self.assertEqual(session.messages[0]["role"],"system")
        self.assertEqual([m["role"] for m in session.messages[2:]], ["assistant","tool"]*12)
        self.assertNotIn("PRIVATE_STATE_MARKER",json.dumps(session.public()))
        self.assertNotIn("world_fingerprint",json.dumps(session.public()))

    def test_downsample_then_zoom_does_not_recover_source_detail(self):
        down=self.env.step({"type":"zoom","image_id":"before","factor":.25})["images"][0]
        up=self.env.step({"type":"zoom","image_id":down["view_id"],"factor":4})["images"][0]
        original=self.env.observation()["images"][0]
        self.assertEqual(up["source_id"],original["source_id"])
        self.assertEqual(up["source_bbox"],original["source_bbox"])
        self.assertNotEqual(up["image_png"],original["image_png"])
        self.assertEqual(list(self.root.glob("*.png")),[self.root/"before.png",self.root/"after.png"])

    def test_region_union_time_and_decisive_counterexample(self):
        self.env.step({"type":"request_photo",**self.request()["arguments"]})
        decision={"verdict":"Supported","citations":[
            {"image_id":"before","time":"before","bbox":[0,0,4,8]},
            {"image_id":"before","time":"before","bbox":[4,0,8,8]},
            {"image_id":"after","time":"after","bbox":[4,4,12,12]}]}
        self.assertTrue(self.env.evaluate(decision)["grounded_correct"])
        decision["citations"][1]=deepcopy(decision["citations"][0])
        self.assertFalse(self.env.evaluate(decision)["grounded_correct"])
        bad=self.finish()["arguments"];bad["citations"][1]["time"]="before"
        with self.assertRaises(ActionError): self.env.step({"type":"finish",**bad})
        self.assertFalse(self.env.finished)
        case=deepcopy(self.case);case["annotation"]["verdict"]="Refuted"
        env=Environment(case,self.root)
        self.assertTrue(env.evaluate({"verdict":"Refuted","citations":[
            {"image_id":"before","time":"before","bbox":[0,0,4,4]}]})["grounded_correct"])

    def test_sibling_branches_and_training_exports(self):
        session=Conversation(self.env);a,b=session.fork(),session.fork()
        original=self.env._world.fingerprint
        a.call(self.request());a.call(self.finish());b.call(self.finish(False))
        self.assertEqual(self.env.budget,24)
        self.assertEqual(a._env._world.fingerprint,original)
        self.assertEqual(b._env._world.fingerprint,original)
        self.assertEqual(len(session.messages),2)
        pair=preference_pair(a.record(),b.record())
        self.assertEqual(pair["chosen"]["tool_calls"][0]["function"]["name"],"request_photo")
        store=TrajectoryStore(self.root/"traces.jsonl")
        record=store.write(a.record());store.write(a.record())
        self.assertEqual(len(list(store.assets.glob("*.png"))),1)
        example=sft_example(record)
        self.assertIsNotNone(example)
        self.assertNotIn("evaluation",example)
        self.assertNotIn("PRIVATE_STATE_MARKER",json.dumps(example))
        self.assertNotIn("image_png",json.dumps(example))

    def test_unknown_extra_arguments_unreleased_and_budget_errors_are_atomic(self):
        session=Conversation(self.env)
        for call in [{"name":"inspect","arguments":{"image_id":"after"}},
                     {"name":"crop","arguments":{"image_id":"before","bbox":[-1,0,4,4]}},
                     {"name":"inspect","arguments":{"image_id":"before","type":"finish"}},
                     {"name":"inspect","arguments":{"image_id":"before","hidden_truth":True}}]:
            result=session.call(call)
            self.assertIn("error",result)
            self.assertEqual(self.env.budget,24)
        self.env.budget=0
        result=session.call(self.request())
        self.assertEqual(result["error"],"insufficient_budget")
        session.call({"name":"finish","arguments":{"verdict":"Need more evidence","citations":[]}})
        length=len(session.messages)
        with self.assertRaises(ActionError): session.call(self.request())
        self.assertEqual(len(session.messages),length)

    def test_invalid_or_exhausted_rollouts_do_not_fabricate_finish(self):
        class Policy:
            def __init__(self): self.lengths=[]
            def ask_conversation(inner,payload,limit):
                inner.lengths.append(len(payload["messages"]))
                return {"tool_call":{"name":"inspect","arguments":{"image_id":"before"}},"text":"", "parse_error":False}
        policy=Policy()
        record=run(Conversation(self.env),policy,max_turns=3)
        self.assertEqual(policy.lengths,[2,4,6])
        self.assertEqual(record["metadata"]["termination"],"max_turns")
        self.assertIsNone(record["decision"])
        self.assertIsNone(record["evaluation"])
        self.assertIsNone(sft_example(record,allow_proxy=True))
        with self.assertRaises(ValueError): parse_tool_call('<tool_call>{"name":"inspect","arguments":{}}</tool_call>'*2)

    def test_missing_unavailable_same_response_and_scene_unchanged(self):
        missing=deepcopy(self.case);missing["evidence"]=missing["evidence"][:1]
        unavailable=deepcopy(self.case);unavailable["evidence"][1]["available"]=False
        a,b=Environment(missing,self.root),Environment(unavailable,self.root)
        self.assertEqual(a.observation(),b.observation())
        query={"type":"request_photo",**self.request()["arguments"]}
        self.assertEqual(a.step(query),b.step(query))
        self.assertEqual(a.requests,b.requests)
