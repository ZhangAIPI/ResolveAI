"""Paired interface checks on unchanged public development pictures.

No choice-mode failure or malformed answer is converted to an abstention.
Run on an allocated GPU; use a fresh output directory for every code revision.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import content_for
from resolveai.diagnostic import DIRECT_SYSTEM, VERDICTS, original_id, public_observation, request_query
from resolveai.environment import Environment
from resolveai.pilot import ModelClient
from resolveai.tools import tool_schemas


def verdict_from_text(raw):
    """An explicit answer only; ambiguity and silence remain parse failures."""
    match = re.fullmatch(
        r"\s*(?:VERDICT:\s*)?(Supported|Refuted|Need more evidence)\s*[.!]?\s*", raw
    )
    return match.group(1) if match else None


def select_probe(cases):
    buckets = defaultdict(list)
    for case in cases:
        category = case.get("category", case.get("provenance", {}).get("category"))
        buckets[(case["diagnostic_source"], category, case["diagnostic_truth"])].append(case)
    chosen = []
    for (source, _, _), bucket in sorted(buckets.items()):
        bucket.sort(key=lambda c: hashlib.sha256(("interface-v1|" + c["family_id"]).encode()).hexdigest())
        chosen.extend(bucket[:1 if source == "mvtec" else 2])
    return chosen


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run on the allocated GPU node")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = select_probe(json.loads(args.plan.read_text())["cases"])
    config = {
        "protocol": "interface-check-v1", "model_id": args.model_id,
        "revision": args.revision, "source_sha256": source_digest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "families": [c["family_id"] for c in cases], "max_new_tokens": 96,
        "scope": "adapter controls, not reviewed semantic accuracy",
    }
    manifest = locked_manifest(args.output / "manifest.json", config)
    path = args.output / "results.jsonl"
    if path.exists():
        parser.error("Use a fresh output directory; do not append duplicate checks")
    client = ModelClient(args.model, args.device, "auto")
    manifest["worker"] = client.ready
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    def save(row):
        rows.append(row)
        append(path, row)
        print(json.dumps({"completed": len(rows), "kind": row["kind"]}), flush=True)
    try:
        for verdict in VERDICTS:
            response = client.ask_conversation({
                "mode": "text", "messages": [{"role": "user", "content":
                    "This is an instruction-following check, not an image question. "
                    f"Reply with exactly these words and nothing else: {verdict}"}],
                "max_new_tokens": 32,
            })
            save({"kind": "text_copy", "expected": verdict,
                  "parsed": verdict_from_text(response.get("raw", "")),
                  "passed": verdict_from_text(response.get("raw", "")) == verdict,
                  "response": response})
        case = cases[0]
        env = Environment(case, case["asset_root"])
        observation = public_observation(env, case)
        calls = [
            {"name": "inspect", "arguments": {"image_id": observation["images"][0]["view_id"]}},
            {"name": "finish", "arguments": {"verdict": "Need more evidence", "citations": []}},
        ]
        adapters = [("portable-prefix", "portable", True), ("portable-plain", "portable", False)]
        if client.ready["tool_adapter"] == "native":
            adapters.append(("native", "native", False))
        for transport, adapter, prefix in adapters:
            for call in calls:
                messages = [{"role": "system", "content":
                    "This is a tool-interface check. Copy the requested tool call exactly. "
                    "Do not solve the image claim or choose another action."},
                    {"role": "user", "content": content_for(observation)}]
                messages.append({"role": "user", "content":
                    "Emit exactly this registered call: " + json.dumps(call)})
                response = client.ask_conversation({
                    "mode": "tools", "messages": messages,
                    "tools": [t for t in tool_schemas(False) if t["function"]["name"] in {"finish", "inspect"}],
                    "tool_adapter": adapter, "assistant_prefix": prefix,
                    "max_new_tokens": 128,
                })
                exact = response.get("tool_call") == call
                valid = False
                if response.get("tool_call"):
                    from resolveai.conversation import Conversation
                    session = Conversation(env.fork())
                    valid = "error" not in session.call(response["tool_call"])
                save({"kind": "tool_copy", "transport": transport, "expected": call,
                      "exact": exact, "valid": valid, "response": response})
        for case in cases:
            env = Environment(case, case["asset_root"])
            result = env.step({"type": "request_photo", "query": request_query(case)})
            if result["status"] != "provided" or original_id(case) not in env._released:
                raise RuntimeError("retrieval preflight failed")
            observation = public_observation(env, case)
            metadata = {k: v for k, v in observation.items()
                        if k in {"claim", "claim_parts", "images", "target_regions"}}
            base = [{"role": "system", "content": DIRECT_SYSTEM},
                    {"role": "user", "content": content_for(metadata)}]
            for order in ["ABC", "BCA", "CAB"]:
                options = dict(zip(order, VERDICTS))
                messages = deepcopy(base)
                messages.append({"role": "user", "content":
                    json.dumps({"options": options}) + "\nReply with exactly one option letter."})
                response = client.ask_conversation({
                    "mode": "choice", "messages": messages, "choices": list(options)})
                verdict = options.get(response.get("choice"))
                save({"kind": "full_choice", "family_id": case["family_id"],
                      "source": case["diagnostic_source"], "order": order,
                      "label": case["diagnostic_truth"], "verdict": verdict,
                      "label_match": verdict == case["diagnostic_truth"], "response": response})
            for kind, instruction in [
                ("full_text", "Reply with exactly one verdict: Supported, Refuted, or Need more evidence."),
                ("full_letter", "Options: A=Supported, B=Refuted, C=Need more evidence. Reply with exactly one option letter."),
            ]:
                messages = deepcopy(base)
                messages.append({"role": "user", "content": instruction})
                response = client.ask_conversation({"mode": "text", "messages": messages, "max_new_tokens": 96})
                raw = response.get("raw", "")
                if kind == "full_text":
                    verdict = verdict_from_text(raw)
                else:
                    match = re.fullmatch(r"\s*([ABC])\s*[.!]?\s*", raw)
                    verdict = dict(zip("ABC", VERDICTS)).get(match.group(1)) if match else None
                save({"kind": kind, "family_id": case["family_id"], "source": case["diagnostic_source"],
                      "label": case["diagnostic_truth"], "verdict": verdict,
                      "parse_error": verdict is None, "label_match": verdict == case["diagnostic_truth"],
                      "response": response})
    finally:
        client.close()
    metrics = {}
    for kind in ["text_copy", "tool_copy", "full_choice", "full_text", "full_letter"]:
        selected = [r for r in rows if r["kind"] == kind]
        metrics[kind] = {"records": len(selected)}
        if kind == "text_copy":
            metrics[kind]["passed"] = sum(r["passed"] for r in selected)
        elif kind == "tool_copy":
            metrics[kind]["by_transport"] = {
                transport: {"exact": sum(r["exact"] for r in selected if r["transport"] == transport),
                            "valid": sum(r["valid"] for r in selected if r["transport"] == transport)}
                for transport in sorted({r["transport"] for r in selected})}
        else:
            metrics[kind].update(label_matches=sum(r["label_match"] for r in selected),
                                 parse_errors=sum(r.get("parse_error", False) for r in selected),
                                 answers=dict(Counter(str(r["verdict"]) for r in selected)))
    summary = {"complete": True, "model_id": args.model_id, "families": len(cases),
               "records": len(rows), "metrics": metrics,
               "scope": "source-label agreement and interface controls; no reviewed sufficiency claim"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
