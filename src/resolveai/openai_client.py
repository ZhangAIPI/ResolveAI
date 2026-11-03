"""Budgeted Responses/Decisions adapters; only released pixels leave the environment."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

# Conservative ceilings include the published 12-hour cache-write prices.
# Each tuple is (input ceiling per million, output rate, supported effort).
MODEL_RATES = {
    "gpt-6.1-terra": (2.5, 5, "none"),
    "gpt-6.1-sol": (5, 10, "low"),
    "gpt-5.6-sol": (10, 20, "none"),
    "gpt-4.1-2025-04-14": (5, 8, None),
    "gpt-4.1-mini-2025-04-14": (1, 1.6, None),
    "gpt-5.4-mini-2026-03-17": (1.875, 4.5, "none"),
    "gpt-6-luna": (.25, .5, "none"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def wire_content(content):
    if isinstance(content,str):
        return [{"type":"input_text","text":content}]
    result = []
    for block in content:
        if block["type"] == "text":
            result.append({"type":"input_text","text":block["text"]})
        elif block["type"] == "image":
            image = Image.open(BytesIO(base64.b64decode(block["image_png"],validate=True))).convert("RGB")
            if image.width*image.height > 512*512:
                scale = math.sqrt(512*512/(image.width*image.height))
                image = image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),
                                     Image.Resampling.LANCZOS)
            data = BytesIO()
            image.save(data,format="PNG")
            result.append({"type":"input_image","detail":"high",
                           "image_url":"data:image/png;base64,"+base64.b64encode(data.getvalue()).decode()})
        else:
            raise ValueError("Unsupported public content")
    return result


def responses_input(messages, native_outputs=None):
    result = []
    call_ids = {}
    for index,message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            result.append({"type":"function_call_output","call_id":call_ids.get(message["tool_call_id"],message["tool_call_id"]),
                           "output":wire_content(message["content"])})
        elif role == "assistant" and message.get("tool_calls"):
            # Retain provider-encrypted reasoning in a multi-turn Responses history.
            # It is never a substitute for environment observations or private labels.
            previous = (native_outputs or {}).get(digest(messages[:index]),[])
            if previous:
                native_calls = [item for item in previous if item.get("type")=="function_call"]
                if len(native_calls)!=len(message["tool_calls"]):
                    raise ValueError("Native and environment call counts differ")
                for local,native in zip(message["tool_calls"],native_calls):
                    call_ids[local["id"]] = native["call_id"]
                result.extend(deepcopy(previous))
                continue
            if message.get("content"):
                result.append({"role":"assistant","content":message["content"]})
            for call in message["tool_calls"]:
                function = call["function"]
                result.append({"type":"function_call","call_id":call["id"],"name":function["name"],
                               "arguments":json.dumps(function["arguments"])})
        else:
            content = message["content"]
            result.append({"role":role,"content":content if role=="assistant" else wire_content(content)})
    return result


class OpenAIClient:
    """Native function generation, or direct choice-only Decisions classification."""

    def __init__(self, config, ledger, *, stage="pilot", stage_limit_usd=1, transport=None):
        model = config["model_id"]
        if model not in MODEL_RATES:
            raise ValueError("Model pricing/capabilities have not been audited")
        rate_in, rate_out, effort = MODEL_RATES[model]
        self.endpoint = config.get("endpoint","responses")
        if self.endpoint not in {"responses","decisions"}:
            raise ValueError("Unsupported endpoint")
        if self.endpoint=="decisions":
            if model!="gpt-6-luna":
                raise ValueError("Decisions supports Luna only")
            rate_out = 0
        if (config["reasoning_effort"] != effort
                or config["input_ceiling_usd_per_million"] < rate_in
                or config["output_usd_per_million"] != rate_out):
            raise ValueError("Configuration differs from audited rates or reasoning settings")
        if effort=="low" and config.get("max_total_output_tokens") != 2048:
            raise ValueError("Reasoning runs require the declared 2048-token total cap")
        self.config, self.ledger = deepcopy(config), ledger
        self.stage, self.stage_limit_usd = stage, stage_limit_usd
        self.transport = transport or self._post
        self.key = os.environ.get("OPENAI_API_KEY")
        if transport is None and not self.key:
            raise RuntimeError("Configure OPENAI_API_KEY privately on the server")
        self.ready = {"provider":"OpenAI","model":model,"reasoning":effort,
                      "tool_adapter":"choice-only" if self.endpoint=="decisions" else "responses-functions-best-effort",
                      "endpoint":self.endpoint,"image_max_pixels":512*512}
        self.namespace = ""
        self.native_outputs = {}

    def begin_episode(self, case_id, arm):
        self.namespace = case_id+"/"+arm
        self.native_outputs = {}

    def _post(self, route, payload):
        request = Request("https://api.openai.com/v1/"+route,
                          data=json.dumps(payload).encode(),
                          headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"})
        try:
            with urlopen(request,timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"OpenAI HTTP {error.code}; stopped without retry") from None
        except OSError:
            raise RuntimeError("OpenAI transport failed; stopped without retry") from None

    def _request(self, payload, bound_input, max_output):
        rate_in = self.config["input_ceiling_usd_per_million"]
        rate_out = self.config["output_usd_per_million"]
        bound = (bound_input*rate_in + max_output*rate_out)/1e6
        key = digest({"episode":self.namespace,"config":self.config,"payload":payload})
        ticket,response = self.ledger.reserve(key,bound,stage=self.stage,
                                              stage_limit_usd=self.stage_limit_usd)
        started = time.perf_counter()
        reused = ticket is None
        if not reused:
            response = self.transport(self.endpoint,payload)
            response["_resolveai_latency_s"] = time.perf_counter()-started
            self.ledger.record_response(ticket,response)
            usage = response.get("usage")
            if (not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                                                 for k in ("input_tokens","output_tokens"))):
                raise RuntimeError("Missing usage; reservation retained, no more spending")
            if usage["input_tokens"]>bound_input or (self.endpoint=="responses" and usage["output_tokens"]>max_output):
                raise RuntimeError("Usage exceeded reservation; retained for reconciliation")
            charge = (usage["input_tokens"]*rate_in+usage["output_tokens"]*rate_out)/1e6
            response["_resolveai_latency_s"] = time.perf_counter()-started
            self.ledger.settle(ticket,charge,response)
        return response,reused,time.perf_counter()-started

    def ask_conversation(self, public, max_context_tokens=8192):
        if not self.namespace:
            raise ValueError("Set the episode namespace before making requests")
        if public["mode"] not in {"text","tools"} or max_context_tokens!=8192:
            raise ValueError("Unsupported benchmark setting")
        if self.endpoint=="decisions":
            return self._decision(public)
        started = time.perf_counter()
        tool_mode = public["mode"]=="tools"
        visible_limit = 384 if tool_mode else 96
        payload = {"model":self.config["model_id"],
                   "input":responses_input(public["messages"],self.native_outputs),
                   "store":False,"service_tier":"default","truncation":"disabled",
                   "max_output_tokens":self.config.get("max_total_output_tokens",visible_limit)}
        if self.config["reasoning_effort"] is not None:
            payload["reasoning"] = {"effort":self.config["reasoning_effort"]}
        if tool_mode:
            payload.update(
                tools=[{"type":"function",**deepcopy(t["function"]),"strict":False}
                       for t in public["tools"]],
                parallel_tool_calls=False,tool_choice="auto")
        count_payload = {k:v for k,v in payload.items()
                         if k in {"model","input","tools","tool_choice","parallel_tool_calls","reasoning","truncation"}}
        count = self.transport("responses/input_tokens",count_payload)["input_tokens"]
        if type(count) is not int or count<0:
            raise ValueError("Invalid server token count")
        if count>max_context_tokens:
            return {"halt":"context_limit","input_tokens":count}
        response,reused,replay_wall = self._request(payload,8192,payload["max_output_tokens"]+64)
        usage = response["usage"]
        output = response.get("output",[])
        text = "".join(b.get("text","") for item in output if item.get("type")=="message"
                       for b in item.get("content",[]) if b.get("type")=="output_text")
        reasoning = usage.get("output_tokens_details",{}).get("reasoning_tokens",0)
        visible = usage["output_tokens"]-reasoning
        if reasoning and self.config["reasoning_effort"] in (None,"none"):
            raise RuntimeError("Unexpected reasoning usage; stop and inspect configuration")
        metadata = {"input_tokens":usage["input_tokens"],"output_tokens":usage["output_tokens"],
                    "reasoning_tokens":reasoning,"visible_output_tokens":visible,
                    "api_usage":usage,"api_response_id":response.get("id"),"api_model":response.get("model"),
                    "api_status":response.get("status"),"response_reused":reused,
                    "latency_s":response["_resolveai_latency_s"],
                    "current_wall_s":time.perf_counter()-started,"replay_wall_s":replay_wall}
        if visible>visible_limit:
            return {"halt":"output_limit","raw":"","tool_call":None,"text":"","parse_error":False,**metadata}
        raw = text
        call,parse_error = None,False
        if tool_mode:
            calls = [item for item in output if item.get("type")=="function_call"]
            try:
                if response.get("status")!="completed" or len(calls)!=1:
                    raise ValueError("Expected one completed native call")
                item = calls[0]
                arguments = json.loads(item["arguments"])
                if not isinstance(arguments,dict):
                    raise ValueError("Arguments must be an object")
                call = {"name":item["name"],"arguments":arguments}
                self.native_outputs[digest(public["messages"])] = deepcopy(output)
            except (ValueError,KeyError,TypeError):
                parse_error = True
            raw = text+"\n"+json.dumps(calls)
        return {"raw":raw,"text":text,"tool_call":call,"parse_error":parse_error,**metadata}

    def _decision(self, public):
        if public["mode"]!="text":
            raise ValueError("Decisions does not generate native tool arguments")
        started = time.perf_counter()
        question = {"name":"verdict","type":"choice",
                    "instructions":"Choose the verdict for the stated visual claim.",
                    "choices":[{"value":v,"description":v} for v in
                               ("Supported","Refuted","Need more evidence")]}
        inputs = responses_input(public["messages"])
        # Decisions has no token-count endpoint. Count an explicit question proxy,
        # allow 512 tokens of interface overhead, and reserve the full input cap.
        proxy = inputs+[{"role":"user","content":[{"type":"input_text","text":json.dumps(question)}]}]
        count = self.transport("responses/input_tokens",{"model":self.config["model_id"],"input":proxy})["input_tokens"]
        if type(count) is not int or count<0:
            raise ValueError("Invalid server token count")
        if count+512>8192:
            return {"halt":"context_limit","input_tokens":count}
        payload = {"model":self.config["model_id"],"input":inputs,"questions":[question]}
        response,reused,replay_wall = self._request(payload,8192,0)
        answers = response.get("answers",[])
        answer = answers[0] if len(answers)==1 else {}
        verdict = answer.get("choice") if answer.get("type")=="choice" else None
        if verdict not in {"Supported","Refuted","Need more evidence"}:
            verdict = ""
        usage = response["usage"]
        return {"raw":verdict,"input_tokens":usage["input_tokens"],"output_tokens":usage["output_tokens"],
                "api_usage":usage,"api_model":response.get("model"),"api_answer":answer,
                "api_endpoint":"decisions","response_reused":reused,
                "latency_s":response["_resolveai_latency_s"],
                "current_wall_s":time.perf_counter()-started,"replay_wall_s":replay_wall}

    def close(self):
        pass
