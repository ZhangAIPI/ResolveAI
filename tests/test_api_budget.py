"""No-network tests of dollar caps, uncertain charges, and native image/tool wiring."""
import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from resolveai.api_budget import ApiBudget, BudgetStop
from resolveai.openai_client import OpenAIClient, responses_input, wire_content


CONFIG = {"model_id":"gpt-6.1-terra","reasoning_effort":"none",
          "input_ceiling_usd_per_million":2.5,"output_usd_per_million":5}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"budget.sqlite"
        self.budget = ApiBudget(self.path,1)

    def tearDown(self):
        self.budget.close()
        self.temp.cleanup()

    def test_shared_cap_and_resume(self):
        ticket,_ = self.budget.reserve("first",.8)
        self.budget.settle(ticket,.7,{"id":"one"})
        second = ApiBudget(self.path,1)
        try:
            with self.assertRaises(BudgetStop):
                second.reserve("different-model",.31)
            ticket,response = second.reserve("first",.8)
            self.assertIsNone(ticket)
            self.assertEqual(response,{"id":"one"})
        finally:
            second.close()

    def test_uncertain_charge_blocks_another_process(self):
        self.budget.reserve("timeout",.4)
        second = ApiBudget(self.path,1)
        try:
            with self.assertRaises(BudgetStop):
                second.reserve("next",.1)
            self.assertEqual(second.status()["unresolved_upper_usd"],.4)
        finally:
            second.close()

    def test_stage_budget_survives_restart(self):
        ticket,_ = self.budget.reserve("pilot-one",.15,stage="pilot",stage_limit_usd=.2)
        self.budget.settle(ticket,.15,{})
        second = ApiBudget(self.path,1)
        try:
            with self.assertRaises(BudgetStop):
                second.reserve("pilot-two",.06,stage="pilot",stage_limit_usd=.2)
            with self.assertRaises(ValueError):
                second.reserve("changed",.01,stage="pilot",stage_limit_usd=.4)
        finally:
            second.close()

    def test_usage_above_reservation_does_not_free_unknown_balance(self):
        ticket,_ = self.budget.reserve("overshoot",.1)
        with self.assertRaises(BudgetStop):
            self.budget.settle(ticket,.11,{})
        with self.assertRaises(BudgetStop):
            self.budget.reserve("next",.1)

    def test_adapter_failure_is_not_retried(self):
        calls = []
        def post(route,payload):
            calls.append(route)
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            raise RuntimeError("simulated timeout")
        client = OpenAIClient(CONFIG,self.budget,transport=post)
        client.begin_episode("a","interactive_agent")
        public = {"mode":"tools","messages":[{"role":"user","content":"claim"}],"tools":[]}
        with self.assertRaises(RuntimeError):
            client.ask_conversation(public)
        with self.assertRaises(BudgetStop):
            client.ask_conversation(public)
        self.assertEqual(calls.count("responses"),1)

    def test_usage_and_completed_response_are_reused(self):
        calls=[]
        def post(route,payload):
            calls.append(route)
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            return {"id":"response-one","status":"completed","model":"gpt-6.1-terra",
                    "usage":{"input_tokens":100,"output_tokens":15,
                             "output_tokens_details":{"reasoning_tokens":0}},
                    "output":[{"type":"function_call","call_id":"api-id","name":"finish",
                               "arguments":json.dumps({"verdict":"Refuted","citations":[]})}]}
        client = OpenAIClient(CONFIG,self.budget,transport=post)
        client.begin_episode("a","interactive_agent")
        public = {"mode":"tools","messages":[{"role":"user","content":"claim"}],"tools":[]}
        first = client.ask_conversation(public)
        second = client.ask_conversation(public)
        self.assertEqual(first["tool_call"]["name"],"finish")
        self.assertEqual(first["latency_s"],second["latency_s"])
        self.assertTrue(second["response_reused"])
        self.assertEqual(calls.count("responses"),1)
        self.assertEqual(self.budget.status()["requests"],1)

    def test_context_overflow_does_not_generate(self):
        routes=[]
        def post(route,payload):
            routes.append(route)
            return {"input_tokens":9000}
        client = OpenAIClient(CONFIG,self.budget,transport=post)
        client.begin_episode("a","initial_text")
        result=client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})
        self.assertEqual(result["halt"],"context_limit")
        self.assertEqual(routes,["responses/input_tokens"])
        self.assertEqual(self.budget.status()["requests"],0)

    def test_images_and_function_results_keep_their_roles(self):
        pixels=BytesIO()
        Image.new("RGB",(1000,800)).save(pixels,format="PNG")
        blocks=[{"type":"text","text":"image metadata"},
                {"type":"image","image_png":base64.b64encode(pixels.getvalue()).decode()}]
        result=responses_input([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"call-0000","function":{"name":"crop","arguments":{"view_id":"v1"}}}]},
            {"role":"tool","tool_call_id":"call-0000","content":blocks}])
        self.assertEqual(result[0]["type"],"function_call")
        self.assertEqual(result[1]["call_id"],"call-0000")
        self.assertEqual(result[1]["output"][0]["text"],"image metadata")
        image=Image.open(BytesIO(base64.b64decode(result[1]["output"][1]["image_url"].split(",")[1])))
        self.assertLessEqual(image.width*image.height,512*512)


if __name__=="__main__":
    unittest.main()
