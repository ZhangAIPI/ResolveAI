"""Responses adapter for released pixels only; no implicit retries or paid tools."""
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


def responses_input(messages):
    result = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            result.append({"type":"function_call_output","call_id":message["tool_call_id"],
                           "output":wire_content(message["content"])})
        elif role == "assistant" and message.get("tool_calls"):
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
    """Initially limited to Terra without reasoning, matching the local output caps."""

    def __init__(self, config, ledger, *, stage="pilot", stage_limit_usd=1, transport=None):
        if config["model_id"] != "gpt-6.1-terra" or config["reasoning_effort"] != "none":
            raise ValueError("Only the audited non-reasoning Terra configuration is enabled")
        self.config, self.ledger = deepcopy(config), ledger
        self.stage, self.stage_limit_usd = stage, stage_limit_usd
        self.transport = transport or self._post
        self.key = os.environ.get("OPENAI_API_KEY")
        if transport is None and not self.key:
            raise RuntimeError("Configure OPENAI_API_KEY on the server; do not put it in the repository")
        self.ready = {"provider":"OpenAI","model":config["model_id"],"reasoning":"none",
                      "tool_adapter":"responses-functions-best-effort","image_max_pixels":512*512}
        self.namespace = ""

    def begin_episode(self, case_id, arm):
        self.namespace = case_id+"/"+arm

    def _post(self, route, payload):
        request = Request("https://api.openai.com/v1/"+route,
                          data=json.dumps(payload).encode(),
                          headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"})
        # urllib does not retry POSTs. Never include server error bodies or headers in logs.
        try:
            with urlopen(request,timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"OpenAI HTTP {error.code}; stopped without retry") from None
        except OSError:
            raise RuntimeError("OpenAI transport failed; stopped without retry") from None

    def ask_conversation(self, public, max_context_tokens=8192):
        if not self.namespace:
            raise ValueError("Set the episode namespace before making requests")
        if public["mode"] not in {"text","tools"} or max_context_tokens != 8192:
            raise ValueError("Unsupported benchmark setting")
        started = time.perf_counter()
        tool_mode = public["mode"] == "tools"
        payload = {"model":self.config["model_id"],"input":responses_input(public["messages"]),
                   "reasoning":{"effort":"none"},"store":False,"service_tier":"default",
                   "truncation":"disabled",
                   "max_output_tokens":384 if tool_mode else 96}
        if tool_mode:
            payload.update(
                tools=[{"type":"function",**deepcopy(t["function"]),"strict":False}
                       for t in public["tools"]],
                parallel_tool_calls=False,tool_choice="auto")
        # Server-side counts include native schemas, images, and repeated history.
        count_payload = {k:v for k,v in payload.items()
                         if k in {"model","input","tools","tool_choice","parallel_tool_calls","reasoning","truncation"}}
        count = self.transport("responses/input_tokens",count_payload)["input_tokens"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid server token count")
        if count > max_context_tokens:
            return {"halt":"context_limit","input_tokens":count}
        # All input is charged conservatively at the configured ceiling, ignoring discounts.
        rate_in = self.config["input_ceiling_usd_per_million"]
        rate_out = self.config["output_usd_per_million"]
        if rate_in < 2.5 or rate_out < 5:
            raise ValueError("Prices below the audited conservative ceilings")
        bound = ((count+32)*rate_in + payload["max_output_tokens"]*rate_out)/1e6
        key = digest({"episode":self.namespace,"config":self.config,"payload":payload})
        ticket, response = self.ledger.reserve(key,bound,stage=self.stage,
                                              stage_limit_usd=self.stage_limit_usd)
        reused = ticket is None
        if not reused:
            response = self.transport("responses",payload)
            usage = response.get("usage")
            if (not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                                                 for k in ("input_tokens","output_tokens"))):
                raise RuntimeError("Missing usage; reservation retained, no more spending")
            charge = (usage["input_tokens"]*rate_in+usage["output_tokens"]*rate_out)/1e6
            if (usage["input_tokens"]>count+32 or usage["output_tokens"]>payload["max_output_tokens"]
                    or usage.get("output_tokens_details",{}).get("reasoning_tokens",0)):
                raise RuntimeError("Usage differs from audited configuration; reservation retained")
            response["_resolveai_latency_s"] = time.perf_counter()-started
            self.ledger.settle(ticket,charge,response)
        usage = response["usage"]
        output = response.get("output",[])
        text = "".join(b.get("text","") for item in output if item.get("type")=="message"
                       for b in item.get("content",[]) if b.get("type")=="output_text")
        raw = text
        call, parse_error = None, False
        if tool_mode:
            calls = [item for item in output if item.get("type")=="function_call"]
            try:
                if response.get("status") != "completed" or len(calls)!=1:
                    raise ValueError("Expected one completed native call")
                item = calls[0]
                arguments = json.loads(item["arguments"])
                if not isinstance(arguments,dict):
                    raise ValueError("Arguments must be an object")
                call = {"name":item["name"],"arguments":arguments}
            except (ValueError,KeyError,TypeError):
                parse_error = True
            raw = text+"\n"+json.dumps(calls)
        return {"raw":raw,"text":text,"tool_call":call,"parse_error":parse_error,
                "input_tokens":usage["input_tokens"],"output_tokens":usage["output_tokens"],
                "api_usage":usage,"api_response_id":response.get("id"),
                "api_model":response.get("model"),"api_status":response.get("status"),
                "response_reused":reused,"latency_s":response["_resolveai_latency_s"]}

    def close(self):
        pass
