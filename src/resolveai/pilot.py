"""Run paired initial/fixed-request/agent/oracle pilot policies.

Only public observations enter the model subprocess. Grounding is explicitly
an original-image quality proxy; semantic truth is from the independent dataset.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

from .environment import Environment

MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def transport(observation):
    return {**observation, "images": [{**image,
        "image_png": base64.b64encode(image["image_png"]).decode("ascii")}
        for image in observation["images"]]}


class ModelClient:
    def __init__(self, model):
        self.process = subprocess.Popen([sys.executable, "-m", "resolveai.vlm_worker", str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("model process failed to load; see stderr")
        self.ready = json.loads(line)

    def ask(self, observation, allowed=False, status="not_requested"):
        payload = {"observation": transport(observation), "request_allowed": allowed,
                   "request_status": status}
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("model process exited during inference")
        return json.loads(line)

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def episode(case, root, client, policy, budget):
    env = Environment(case, root, budget)
    status = "not_requested"
    responses = []
    if policy == "oracle":
        # Separate access condition, never compared as a same-pool policy.
        oracle_case = dict(case)
        oracle_case["evidence"] = [dict(e, available=True) for e in case["evidence"]]
        if not any(e["id"] == "original" for e in oracle_case["evidence"]):
            # The held-out original is evaluation-only in the Missing variant.
            preview = oracle_case["evidence"][0]
            oracle_case["evidence"].append(dict(preview, id="original", view="original",
                path=preview["path"].replace("preview.png", "original.png")))
        oracle_case["initial"] = ["original"]
        env = Environment(oracle_case, root, budget)
    elif policy == "fixed_request" and case["variant"] != "Sufficient":
        status = env.step({"type": "request_photo", "query": case["request_spec"]})["status"]
    response = client.ask(env.observation(), allowed=policy == "agent" and case["variant"] != "Sufficient",
                          status=status)
    responses.append(response)
    if policy == "agent" and case["variant"] != "Sufficient" and response["decision"]["request_original"]:
        status = env.step({"type": "request_photo", "query": case["request_spec"]})["status"]
        response = client.ask(env.observation(), status=status)
        responses.append(response)
    predicted = response["decision"]
    public = {i["image_id"]: i for i in env.observation()["images"]}
    citations = [{"image_id": i, "bbox": public[i]["source_bbox"]}
                 for i in predicted["image_ids"] if i in public]
    invalid_citations = any(i not in public for i in predicted["image_ids"])
    decision = {"type": "finish", "verdict": predicted["verdict"], "citations": citations}
    env.step(decision)
    metrics = env.evaluate(decision)
    if invalid_citations:
        metrics["grounded_correct"] = False
        metrics["unsupported_decision"] = predicted["verdict"] != "Need more evidence"
    return {"case_id": case["case_id"], "family_id": case["family_id"],
        "variant": case["variant"], "category": case["provenance"]["category"],
        "policy": policy, "visual_truth": case["annotation"]["visual_verdict"],
        "decision": decision, "confidence": predicted["confidence"],
        "request_status": status, "invalid_citations": invalid_citations,
        "parse_errors": sum(r["parse_error"] for r in responses),
        "model_calls": len(responses), "input_tokens": sum(r["input_tokens"] for r in responses),
        "output_tokens": sum(r["output_tokens"] for r in responses),
        "latency_s": sum(r["latency_s"] for r in responses),
        "peak_memory_gb": max(r["peak_memory_gb"] for r in responses),
        "raw_outputs": [r["raw"] for r in responses], **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policies", nargs="+", choices=["initial", "fixed_request", "agent", "oracle"],
                        default=["initial", "fixed_request", "agent", "oracle"])
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--limit-families", type=int)
    args = parser.parse_args()
    if args.budget < 3:
        parser.error("budget must allow one original-image request (>=3)")
    cases = json.loads((args.data / "cases.json").read_text())
    if args.limit_families:
        families = sorted({c["family_id"] for c in cases})[:args.limit_families]
        cases = [c for c in cases if c["family_id"] in families]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["case_id"], row["policy"]))
    client = ModelClient(args.model)
    manifest = {"model": "Qwen/Qwen3-VL-8B-Instruct", "revision": MODEL_REVISION,
        "budget": args.budget, "policies": args.policies, "families": len({c["family_id"] for c in cases}),
        "gpu": client.ready["gpu"], "slurm_job": os.getenv("SLURM_JOB_ID"),
        "decoding": {"do_sample": False, "max_new_tokens": 180},
        "image_max_pixels": 512 * 512, "dtype": "bfloat16", "attention": "sdpa",
        "prompt_version": "defect-poc-v1", "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "grounding": "full-resolution-proxy-v1; not human-annotated sufficiency"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with args.output.open("a") as output:
            for case in cases:
                for policy in args.policies:
                    if (case["case_id"], policy) in completed:
                        continue
                    row = episode(case, args.data, client, policy, args.budget)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(json.dumps({k: row[k] for k in ("case_id", "policy", "correct", "requests", "latency_s")}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
