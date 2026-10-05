"""Resumable paired static/interactive evaluation on a frozen public case pool."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from .conversation import Conversation, TrajectoryStore
from .environment import Costs, Environment
from .grounding_tools import FrozenGrounding
from .pilot import ModelClient
from .rollout import run


def source_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def locked_manifest(path, configuration):
    """Refuse to combine different data, weights, source code or rollout settings."""
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["configuration"] != configuration:
            raise ValueError("resume configuration differs; use a fresh output directory")
        return previous
    manifest = {"configuration": configuration,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "slurm_job": os.getenv("SLURM_JOB_ID"),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "Pillow", "rapidocr-onnxruntime")}}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--grounding-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--limit-families", type=int)
    args = parser.parse_args()
    if args.max_turns < 1 or args.budget < 0 or args.max_context_tokens < 1:
        parser.error("invalid rollout bounds")
    cases = json.loads((args.data / "cases.json").read_text())
    if args.limit_families:
        families = sorted({c["family_id"] for c in cases})[:args.limit_families]
        cases = [c for c in cases if c["family_id"] in families]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"dataset_sha256": hashlib.sha256((args.data / "cases.json").read_bytes()).hexdigest(),
        "dataset_manifest": json.loads((args.data / "dataset_manifest.json").read_text()),
        "source_sha256": source_digest(), "model_id": args.model_id, "revision": args.revision,
        "grounding_manifest": json.loads((args.grounding_root / "manifest.json").read_text()),
        "budget": args.budget, "max_turns": args.max_turns, "max_context_tokens": args.max_context_tokens,
        "policies": ["initial", "agent"], "tool_adapter": "portable",
        "costs": asdict(Costs()), "limit_families": args.limit_families,
        "decoding": {"do_sample": False, "max_new_tokens": 384},
        "image_max_pixels": 512 * 512, "dtype": "bfloat16", "attention": "sdpa"}
    manifest_path = args.output / "manifest.json"
    manifest = locked_manifest(manifest_path, config)
    results_path = args.output / "results.jsonl"
    completed = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            key = row["case_id"], row["policy"]
            if key in completed:
                raise ValueError("duplicate completed episode")
            completed.add(key)
    expected = {(c["case_id"], p) for c in cases for p in config["policies"]}
    if not completed <= expected:
        raise ValueError("results contain unexpected cases")
    if completed == expected:
        print(json.dumps({"complete": True, "episodes": len(completed)}), flush=True)
        return
    backend = FrozenGrounding(args.grounding_root, device=f"cuda:{args.device}")
    store = TrajectoryStore(args.output / "trajectories.jsonl")
    client = ModelClient(args.model, args.device, "portable")
    manifest["worker"] = client.ready
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with results_path.open("a") as output:
            for case in cases:
                for policy in config["policies"]:
                    if (case["case_id"], policy) in completed:
                        continue
                    env = Environment(case, args.data, args.budget, grounding_backend=backend)
                    session = Conversation(env)
                    if policy == "initial":
                        session.tools = [t for t in session.tools if t["function"]["name"] == "finish"]
                        session.messages[0]["content"] += "\nThis is the static baseline: decide from the initially supplied images; only finish is available."
                    record = run(session, client, args.max_turns, args.max_context_tokens)
                    record["metadata"]["policy"] = policy
                    store.write(record)
                    metrics = record["evaluation"] or {"correct": False, "grounded_correct": False,
                        "unsupported_decision": False, "coverage": False, "requests": env.requests,
                        "tool_calls": env.calls, "request_cost": env.request_cost,
                        "tool_cost": env.tool_cost, "remaining_budget": env.budget,
                        "grounding_protocol": "image-set-proxy-v1"}
                    row = {"case_id": case["case_id"], "family_id": case["family_id"],
                        "variant": case["variant"], "category": case["provenance"]["category"],
                        "policy": policy, "visual_truth": case["annotation"]["visual_verdict"],
                        "decision": record["decision"], "termination": session.termination,
                        "turns": session.turns, "errors": session.errors,
                        "calls": [m["tool_calls"][0]["function"]["name"] for m in session.messages if m.get("tool_calls")],
                        "model_calls": len(record["generation"]),
                        "input_tokens": sum(g.get("input_tokens", 0) for g in record["generation"]),
                        "output_tokens": sum(g.get("output_tokens", 0) for g in record["generation"]),
                        "latency_s": sum(g.get("latency_s", 0) for g in record["generation"]),
                        "wall_latency_s": record["wall_latency_s"],
                        "peak_memory_gb": max((g.get("peak_memory_gb", 0) for g in record["generation"]), default=0),
                        **metrics}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    completed.add((case["case_id"], policy))
                    print(json.dumps({"completed": len(completed), "expected": len(expected),
                        "case_id": case["case_id"], "policy": policy, "termination": session.termination,
                        "grounded_correct": row["grounded_correct"], "latency_s": row["latency_s"]}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
