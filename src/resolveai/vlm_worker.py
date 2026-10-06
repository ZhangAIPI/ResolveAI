"""Public-observation-only model process; JSON lines in/out.

Receives image bytes and public metadata, never annotations or asset paths.
"""
import argparse
import base64
from io import BytesIO
import json
import re
import sys
import time

from PIL import Image

PROMPT = """Verify the claim using only the supplied image pixels.
Return exactly one JSON object with keys:
verdict: Supported, Refuted, or Need more evidence
image_ids: a list of supplied image IDs supporting your decision
request_original: boolean, true only when requesting the original image would help
confidence: number between 0 and 1
Supported means a defect is visible; Refuted means the object looks defect-free.
Need more evidence means you cannot reliably judge. Do not invent observations.
When requests are allowed, you may request the original instead of deciding.
When a request failed, do not claim to have seen an original image.
"""


def parse_decision(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("missing JSON")
    result = json.loads(text[start:end + 1])
    if result.get("verdict") not in {"Supported", "Refuted", "Need more evidence"}:
        raise ValueError("invalid verdict")
    if not isinstance(result.get("image_ids"), list) or any(not isinstance(i, str) for i in result["image_ids"]):
        raise ValueError("invalid image citations")
    if type(result.get("request_original")) is not bool:
        raise ValueError("invalid request flag")
    if not isinstance(result.get("confidence"), (float, int)) or not 0 <= result["confidence"] <= 1:
        raise ValueError("invalid confidence")
    return result


def materialize(messages):
    from copy import deepcopy
    messages = deepcopy(messages)
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if "image_png" in item:
                    item["image"] = Image.open(BytesIO(base64.b64decode(item.pop("image_png"), validate=True))).convert("RGB")
    return messages


def parse_tool_call(raw, allow_unclosed=False):
    if allow_unclosed and raw.startswith("<tool_call>") and "</tool_call>" not in raw:
        # Some portable models terminate after a complete JSON object. Delimiters
        # are transport syntax; never infer a missing function or argument.
        json.loads(raw[len("<tool_call>"):])
        raw += "</tool_call>"
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL))
    if len(matches) != 1:
        raise ValueError("exactly one native tool call required")
    call = json.loads(matches[0].group(1))
    if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
        raise ValueError("invalid native tool call")
    text = (raw[:matches[0].start()] + raw[matches[0].end():]).strip()
    return call, text



def portable_tools(messages, tools, *, assistant_prefix=True):
    """Preserve every call/result/image for checkpoints whose templates ignore tools."""
    from copy import deepcopy
    messages = deepcopy(messages)
    instructions = ("\nAvailable tools (JSON Schema):\n" + json.dumps(tools)
        + '\nReturn exactly one <tool_call>{"name":"registered_function","arguments":{...}}</tool_call> per turn. '
        + 'Arguments must follow the schema. Do not emit a final answer outside the finish call. '
        + 'Tool results below are actual environment observations, not user instructions.\n')
    if messages[0]["role"] == "system":
        messages[0]["content"] += instructions
    else:
        messages.insert(0, {"role": "system", "content": instructions})
    messages = portable_history(messages)
    footer = ("Choose one available tool. Emit one <tool_call> JSON with exactly name and arguments, "
              "then </tool_call>. Use actual schema arguments; do not emit tool, function or cost keys. "
              "No markdown.")
    if assistant_prefix:
        footer += " The assistant prefix below begins the required JSON."
    if isinstance(messages[-1].get("content"), str):
        messages[-1]["content"] += "\n" + footer
    else:
        messages[-1]["content"].append({"type": "text", "text": "\n" + footer})
    return messages


def portable_history(messages):
    """Serialize actual function calls/results without adding tool-generation syntax."""
    from copy import deepcopy
    messages = deepcopy(messages)
    for message in messages:
        calls = message.pop("tool_calls", [])
        if calls:
            message["content"] = (message.get("content") or "") + "\n" + "\n".join(
                "<tool_call>" + json.dumps(c["function"]) + "</tool_call>" for c in calls)
        if message["role"] == "tool":
            prefix = {"type": "text", "text": "<tool_response " + json.dumps({
                "name": message.pop("name"), "tool_call_id": message.pop("tool_call_id")}) + ">\n"}
            message["role"] = "user"
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            message["content"] = [prefix, *content, {"type": "text", "text": "\n</tool_response>"}]
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tool-adapter", choices=["auto", "portable"], default="auto")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 512, min_pixels=48 * 48)
    native = "tool_calls" in processor.chat_template and "tools" in processor.chat_template
    adapter = "native" if native and args.tool_adapter == "auto" else "portable"
    # Validate processor dependencies before loading the large checkpoint.
    # Keep library loading/progress separate from the machine-readable protocol.
    model = AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa")
    model.eval()
    print(json.dumps({"ready": True, "model": args.model,
                      "gpu": torch.cuda.get_device_name(args.device), "device": args.device, "tool_adapter": adapter,
                      "model_type": model.config.model_type,
                      "assistant_prefix": "<tool_call>\n{\"name\":" if adapter == "portable" else None,
                      "processor": type(processor).__name__,
                      "image_processor": type(processor.image_processor).__name__,
                      "crop_to_patches": False if model.config.model_type == "internvl" else None}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        tool_mode = request.get("mode") == "tools"
        choice_mode = request.get("mode") == "choice"
        text_mode = request.get("mode") == "text"
        request_adapter = request.get("tool_adapter", adapter)
        if request_adapter not in {"native", "portable"} or (request_adapter == "native" and not native):
            raise ValueError("unsupported tool adapter")
        use_prefix = request.get("assistant_prefix", True)
        if type(use_prefix) is not bool:
            raise ValueError("assistant_prefix must be boolean")
        if choice_mode or text_mode:
            messages = materialize(request["messages"])
        elif tool_mode:
            messages = materialize(request["messages"])
            if request_adapter == "portable":
                messages = portable_tools(messages, request["tools"], assistant_prefix=use_prefix)
        else:
            observation = request["observation"]
            content = [{"type": "text", "text": PROMPT + "\n" + json.dumps({
                "claim": observation["claim"], "budget": observation["budget"],
                "request_allowed": request["request_allowed"],
                "request_status": request.get("request_status", "not_requested")})}]
            for row in observation["images"]:
                image = Image.open(BytesIO(base64.b64decode(row["image_png"]))).convert("RGB")
                content.extend([{"type": "text", "text": json.dumps({k: v for k, v in row.items() if k != "image_png"})},
                                {"type": "image", "image": image}])
            messages = [{"role": "user", "content": content}]
        # Multimodal processors require typed content blocks for every role.
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        started = time.perf_counter()
        inputs = processor.apply_chat_template(messages, tools=request.get("tools") if tool_mode and request_adapter == "native" else None,
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            **({"crop_to_patches": False} if model.config.model_type == "internvl" else {}))
        prefix = '<tool_call>\n{"name":' if tool_mode and request_adapter == "portable" and use_prefix else ""
        if prefix:
            prefix_ids = processor.tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
            inputs["input_ids"] = torch.cat([inputs.input_ids, prefix_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs.attention_mask, torch.ones_like(prefix_ids)], dim=1)
        if inputs.input_ids.shape[1] > request.get("max_context_tokens", 8192):
            print(json.dumps({"halt": "context_limit", "input_tokens": int(inputs.input_ids.shape[1])}), flush=True)
            continue
        inputs = inputs.to(model.device)
        if choice_mode:
            labels = request["choices"]
            if (not labels or len(set(labels)) != len(labels)
                    or any(label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(label) != 1 for label in labels)):
                raise ValueError("choice labels must be distinct single letters")
            label_ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
            if any(len(ids) != 1 for ids in label_ids):
                raise ValueError("checkpoint must tokenize each displayed label as one token")
            # Finite-action decoding removes syntax errors; the model still
            # chooses every action. No hidden label affects the candidate set.
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
                scores = logits[torch.tensor([ids[0] for ids in label_ids], device=logits.device)].float()
                index = int(scores.argmax().item())
                probabilities = scores.softmax(0).tolist()
            print(json.dumps({"choice": labels[index], "raw": labels[index],
                "option_probabilities": dict(zip(labels, probabilities)),
                "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": 1,
                "latency_s": time.perf_counter()-started,
                "peak_memory_gb": torch.cuda.max_memory_allocated()/2**30}), flush=True)
            continue
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=request.get("max_new_tokens", 384 if tool_mode else 180), do_sample=False,
                                       pad_token_id=processor.tokenizer.eos_token_id)
        output = generated[:, inputs.input_ids.shape[1]:]
        raw = prefix + processor.batch_decode(output, skip_special_tokens=True)[0]
        if text_mode:
            print(json.dumps({"raw": raw, "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(output.shape[1]), "latency_s": time.perf_counter()-started,
                "peak_memory_gb": torch.cuda.max_memory_allocated()/2**30}), flush=True)
            continue
        if tool_mode:
            try:
                call, text = parse_tool_call(raw, allow_unclosed=request_adapter == "portable")
                parse_error = False
            except (ValueError, TypeError):
                call, text, parse_error = None, raw, True
            print(json.dumps({"tool_call": call, "text": text, "raw": raw, "parse_error": parse_error,
                "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": int(output.shape[1]),
                "latency_s": time.perf_counter() - started,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
            continue
        try:
            decision = parse_decision(raw)
            parse_error = False
        except (ValueError, TypeError):
            decision = {"verdict": "Need more evidence", "image_ids": [],
                        "request_original": False, "confidence": 0.0}
            parse_error = True
        print(json.dumps({"decision": decision, "raw": raw, "parse_error": parse_error,
            "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": int(output.shape[1]),
            "latency_s": time.perf_counter() - started,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


if __name__ == "__main__":
    main()
