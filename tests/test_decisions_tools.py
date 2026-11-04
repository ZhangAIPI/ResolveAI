"""Software-only multi-turn fixtures; gray pixels are not benchmark evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from resolveai.action_menu import action_candidates,action_question,selected_action
from resolveai.api_budget import ApiBudget
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.openai_client import OpenAIClient
from resolveai.rollout import run
from resolveai.tools import validate_action

class NoOCR:
    available=False

class DecisionsToolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ("front","side","blocked"):
            Image.new("RGB",(16,16),"gray").save(self.root/(name+".png"))
        self.case={"claim":"The table has a scratch.","initial":["front"],
            "request_options":{"objects":["table"],"times":["after"],
                               "views":["front","side","blocked","absent"]},
            "evidence":[{"id":name,"path":name+".png","source_id":name,
                         "party":"A","object":"table","time":"after","view":name,
                         "available":name!="blocked"} for name in ("front","side","blocked")],
            "annotation":{"verdict":"Supported","minimal_evidence_sets":{"Supported":[["side"]]}}}
        self.env=Environment(self.case,self.root,ocr_backend=NoOCR())
        self.session=Conversation(self.env)
        self.config={"model_id":"gpt-6-luna","endpoint":"decisions","reasoning_effort":"none",
                     "input_ceiling_usd_per_million":.25,"output_usd_per_million":0,
                     "decision_tools":"public-menu-v1"}

    def test_menu_contains_valid_actions_and_all_verdicts(self):
        public=self.session.public()
        menu=action_candidates(public)
        self.assertLessEqual(len(menu),64)
        for call in menu:
            validate_action({"type":call["name"],**call["arguments"]})
        self.assertEqual({c["arguments"]["verdict"] for c in menu if c["name"]=="finish"},
                         {"Supported","Refuted","Need more evidence"})
        self.assertNotIn("ocr",{c["name"] for c in menu})
        self.assertEqual(menu,action_candidates(public))
        self.assertEqual(len({q["value"] for q in action_question(menu)["choices"]}),len(menu))

    def test_hidden_truth_does_not_change_choices(self):
        expected=action_candidates(self.session.public())
        changed=deepcopy(self.case)
        changed["annotation"]["verdict"]="Refuted"
        changed["evidence"][1]["available"]=False
        other=Conversation(Environment(changed,self.root,ocr_backend=NoOCR()))
        self.assertEqual(expected,action_candidates(other.public()))
        queries=[c["arguments"]["query"]["view"] for c in expected if c["name"]=="request_photo"]
        self.assertIn("blocked",queries)
        self.assertIn("absent",queries)
        self.assertNotIn("side",{c["arguments"].get("image_id") for c in expected})

    def test_new_photos_and_derived_views_remain_actionable(self):
        self.session.call({"name":"request_photo","arguments":{
            "query":{"object":"table","time":"after","view":"side"}}})
        self.assertIn("side",{c["arguments"].get("image_id") for c in action_candidates(self.session.public())})
        result=self.session.call({"name":"crop","arguments":{"image_id":"side","bbox":[1,1,8,8]}})
        view=result["images"][0]["view_id"]
        self.assertIn(view,{c["arguments"].get("image_id") for c in action_candidates(self.session.public())})
        self.assertIn("compare",{c["name"] for c in action_candidates(self.session.public())})

    def test_zero_budget_retains_finish_without_paid_tools(self):
        self.session.call({"name":"request_photo","arguments":{
            "query":{"object":"table","time":"after","view":"side"}}})
        public=self.session.public()
        for message in public["messages"]:
            if message["role"]=="tool":
                value=json.loads(message["content"][0]["text"])
                value["budget"]=0
                message["content"][0]["text"]=json.dumps(value)
        self.assertEqual({c["name"] for c in action_candidates(public)},{"finish"})

    def test_choice_request_crop_finish_executes_actual_environment(self):
        sent=[]
        def transport(route,payload):
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            sent.append(payload)
            actions=[json.loads(c["description"]) for c in payload["questions"][0]["choices"]]
            if len(sent)==1:
                pick=next(i for i,c in enumerate(actions) if c["name"]=="request_photo"
                          and c["arguments"]["query"]["view"]=="side")
            elif len(sent)==2:
                pick=next(i for i,c in enumerate(actions) if c["name"]=="crop"
                          and c["arguments"]["image_id"]=="side")
            else:
                pick=next(i for i,c in enumerate(actions) if c["name"]=="finish"
                          and c["arguments"]["verdict"]=="Supported")
            return {"model":"gpt-6-luna","usage":{"input_tokens":100,"output_tokens":0},
                    "answers":[{"type":"choice","name":"action",
                                "choice":payload["questions"][0]["choices"][pick]["value"]}]}
        ledger=ApiBudget(self.root/"budget.sqlite",1)
        self.addCleanup(ledger.close)
        client=OpenAIClient(self.config,ledger,transport=transport,stage_limit_usd=1)
        client.begin_episode("software-fixture","interactive_agent")
        fingerprint=self.env._world.fingerprint
        trace=run(self.session,client,max_turns=8)
        self.assertEqual(trace["metadata"]["termination"],"finished")
        self.assertEqual(self.env.requests,1)
        self.assertEqual(self.env.calls,2)
        self.assertEqual(self.env._world.fingerprint,fingerprint)
        self.assertEqual(len(sent),3)
        outputs=[m for m in sent[1]["input"] if m.get("type")=="function_call_output"]
        self.assertEqual(outputs[0]["call_id"],"call-0000")
        self.assertTrue(any(b["type"]=="input_image" for b in outputs[0]["output"]))
        self.assertTrue(all(g["action_adapter"]=="public-menu-v1" for g in trace["generation"]))
        self.assertEqual(sum(g["output_tokens"] for g in trace["generation"]),0)

    def test_invalid_choice_cannot_invent_a_call(self):
        self.assertIsNone(selected_action({"type":"choice","name":"action","choice":"unlisted"},
                                         action_candidates(self.session.public())))

if __name__=="__main__":
    unittest.main()
