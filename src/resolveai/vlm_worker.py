"""Public-observation-only model process; JSON lines in/out.

Receives image bytes and public metadata, never annotations or asset paths.
"""
import argparse
import base64
from io import BytesIO
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 512, min_pixels=48 * 48)
    # Validate processor dependencies before loading the large checkpoint.
    # Keep library loading/progress separate from the machine-readable protocol.
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    print(json.dumps({"ready": True, "model": args.model,
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        observation = request["observation"]
        content = [{"type": "text", "text": PROMPT + "\n" + json.dumps({
            "claim": observation["claim"], "budget": observation["budget"],
            "request_allowed": request["request_allowed"],
            "request_status": request.get("request_status", "not_requested")})}]
        for row in observation["images"]:
            image = Image.open(BytesIO(base64.b64decode(row["image_png"]))).convert("RGB")
            content.extend([{"type": "text", "text": json.dumps({k: v for k, v in row.items() if k != "image_png"})},
                            {"type": "image", "image": image}])
        started = time.perf_counter()
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
        output = generated[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(output, skip_special_tokens=True)[0]
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
