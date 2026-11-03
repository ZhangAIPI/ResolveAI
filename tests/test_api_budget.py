"""No-network tests of dollar caps, uncertain charges, and native image/tool wiring."""
import base64
from io import BytesIO
import json
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.evaluate_openai import configure_api_key
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

    def test_key_file_must_be_private_and_is_loaded_without_output(self):
        path = Path(self.temp.name)/"private.env"
        path.write_text("OPENAI_API_KEY=sk-"+"x"*45+"\n")
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            configure_api_key(path)
        path.chmod(0o600)
        with patch.dict(os.environ, clear=True):
            configure_api_key(path)
            self.assertTrue(os.environ["OPENAI_API_KEY"].startswith("sk-"))
        link = path.with_suffix(".link")
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            configure_api_key(link)

    def test_parallel_allocations_cannot_exceed_the_parent_cap(self):
        child_path=Path(self.temp.name)/"child.sqlite"
        child=self.budget.allocate("model-a",child_path,.6)
        try:
            with self.assertRaises(BudgetStop):
                self.budget.reserve("parent-next",.41)
            ticket,_=child.reserve("child-call",.3)
            child.settle(ticket,.2,{})
            self.budget.close_allocation("model-a",child_path)
            self.assertAlmostEqual(self.budget.status()["accounted_upper_usd"],.2)
            with self.assertRaises(BudgetStop):
                child.reserve("after-refund",.01)
        finally:
            child.close()

    def test_allocation_respects_inflight_parent_reservations(self):
        self.budget.reserve("inflight",.4)
        with self.assertRaises(BudgetStop):
            self.budget.allocate("too-large",Path(self.temp.name)/"large.sqlite",.7)
        child=self.budget.allocate("fits",Path(self.temp.name)/"fits.sqlite",.5)
        child.close()
        self.assertAlmostEqual(self.budget.status()["unresolved_upper_usd"],.4)

    def test_refund_keeps_the_full_bound_for_an_unknown_child_request(self):
        child_path=Path(self.temp.name)/"unknown.sqlite"
        child=self.budget.allocate("unknown",child_path,.6)
        try:
            child.reserve("timeout",.3)
            self.budget.close_allocation("unknown",child_path)
            self.assertAlmostEqual(self.budget.status()["accounted_upper_usd"],.3)
        finally:
            child.close()

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

    def test_received_response_survives_a_usage_mismatch(self):
        def post(route,payload):
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            return {"id":"known-response","usage":{"input_tokens":9000,"output_tokens":1},"output":[]}
        client=OpenAIClient(CONFIG,self.budget,transport=post)
        client.begin_episode("a","initial_text")
        with self.assertRaises(RuntimeError):
            client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})
        response,charge=self.budget.db.execute("SELECT response,charge FROM calls").fetchone()
        self.assertEqual(json.loads(response)["id"],"known-response")
        self.assertIsNone(charge)
        with self.assertRaises(BudgetStop):
            client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})

    def test_usage_within_declared_context_does_not_require_a_new_generation(self):
        count=[]
        def post(route,payload):
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            count.append(1)
            return {"status":"completed","model":"gpt-6.1-terra",
                    "usage":{"input_tokens":150,"output_tokens":2},
                    "output":[{"type":"message","content":[{"type":"output_text","text":"Refuted"}]}]}
        client=OpenAIClient(CONFIG,self.budget,transport=post)
        client.begin_episode("a","initial_text")
        result=client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})
        self.assertEqual(result["raw"],"Refuted")
        self.assertEqual(len(count),1)

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

    def test_decisions_is_direct_only_and_charges_no_output(self):
        routes=[]
        config={**CONFIG,"model_id":"gpt-6-luna","endpoint":"decisions",
                "input_ceiling_usd_per_million":.25,"output_usd_per_million":0}
        def post(route,payload):
            routes.append(route)
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            return {"model":"gpt-6-luna",
                    "usage":{"input_tokens":100,"output_tokens":0},
                    "answers":[{"type":"choice","name":"verdict","choice":"Supported","confidence":.8}]}
        client=OpenAIClient(config,self.budget,transport=post)
        client.begin_episode("a","initial_text")
        response=client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})
        self.assertEqual(response["raw"],"Supported")
        self.assertEqual(response["output_tokens"],0)
        self.assertIn("decisions",routes)
        with self.assertRaises(ValueError):
            client.ask_conversation({"mode":"tools","messages":[],"tools":[]})

    def test_reasoning_history_and_tokens_survive_native_tool_turns(self):
        requests=[]
        config={**CONFIG,"model_id":"gpt-6.1-sol","reasoning_effort":"low",
                "input_ceiling_usd_per_million":5,"output_usd_per_million":10,
                "max_total_output_tokens":2048}
        def post(route,payload):
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            requests.append(payload)
            return {"model":"gpt-6.1-sol","status":"completed",
                    "usage":{"input_tokens":100,"output_tokens":60,
                             "output_tokens_details":{"reasoning_tokens":40}},
                    "output":[{"type":"reasoning","id":"r1","summary":[],"encrypted_content":"opaque"},
                              {"type":"function_call","call_id":"native-call","name":"inspect",
                               "arguments":json.dumps({"view_id":"v1"})}]}
        client=OpenAIClient(config,self.budget,transport=post)
        client.begin_episode("a","interactive_agent")
        messages=[{"role":"user","content":"claim"}]
        first=client.ask_conversation({"mode":"tools","messages":messages,"tools":[]})
        self.assertEqual(first["reasoning_tokens"],40)
        self.assertEqual(first["visible_output_tokens"],20)
        messages += [{"role":"assistant","content":"","tool_calls":[
                        {"id":"call-0000","function":first["tool_call"]}]},
                     {"role":"tool","tool_call_id":"call-0000","content":[{"type":"text","text":"observation"}]}]
        client.ask_conversation({"mode":"tools","messages":messages,"tools":[]})
        self.assertTrue(any(item.get("type")=="reasoning" for item in requests[1]["input"]))
        self.assertEqual(requests[1]["max_output_tokens"],2048)
        tool_result=next(item for item in requests[1]["input"] if item.get("type")=="function_call_output")
        self.assertEqual(tool_result["call_id"],"native-call")

    def test_non_reasoning_legacy_models_omit_reasoning_parameter(self):
        payloads=[]
        config={**CONFIG,"model_id":"gpt-4.1-mini-2025-04-14","reasoning_effort":None,
                "input_ceiling_usd_per_million":1,"output_usd_per_million":1.6}
        def post(route,payload):
            payloads.append(payload)
            if route.endswith("input_tokens"):
                return {"input_tokens":100}
            return {"status":"completed","model":config["model_id"],
                    "usage":{"input_tokens":100,"output_tokens":1},
                    "output":[{"type":"message","content":[{"type":"output_text","text":"Refuted"}]}]}
        client=OpenAIClient(config,self.budget,transport=post)
        client.begin_episode("a","initial_text")
        self.assertEqual(client.ask_conversation({"mode":"text","messages":[{"role":"user","content":"claim"}]})["raw"],"Refuted")
        self.assertTrue(all("reasoning" not in payload for payload in payloads))

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
