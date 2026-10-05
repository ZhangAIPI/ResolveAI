"""Bounded native multi-turn rollouts and training-compatible trajectory capture."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .conversation import Conversation, TrajectoryStore
from .environment import Environment
from .grounding_tools import FrozenGrounding
from .pilot import ModelClient


def run(session, client, max_turns=12, max_context_tokens=8192):
    started = time.perf_counter()
    generation = []
    for _ in range(max_turns):
        output = client.ask_conversation(session.public(), max_context_tokens)
        generation.append({k: v for k, v in output.items() if k not in {"raw", "text", "tool_call"}})
        if output.get("halt"):
            session.termination = output["halt"]
            break
        if output["parse_error"]:
            session.invalid_output(output["raw"])
        else:
            session.call(output["tool_call"], output["text"])
        if session.termination == "finished":
            break
    if session.termination is None:
        session.termination = "max_turns"
    record = session.record()
    record["generation"] = generation
    record["wall_latency_s"] = time.perf_counter() - started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grounding-root", type=Path)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--families", type=int, default=2)
    args = parser.parse_args()
    if args.max_turns < 1 or args.families < 1:
        parser.error("turn/family limits must be positive")
    cases = json.loads((args.data / "cases.json").read_text())
    families = sorted({c["family_id"] for c in cases})[:args.families]
    cases = [c for c in cases if c["family_id"] in families]
    store = TrajectoryStore(args.output)
    if args.output.exists():
        parser.error("use a fresh output path to avoid mixing rollout configurations")
    grounding = FrozenGrounding(args.grounding_root) if args.grounding_root else None
    client = ModelClient(args.model)
    results = []
    try:
        for case in cases:
            env = Environment(case, args.data, args.budget, grounding_backend=grounding)
            record = run(Conversation(env), client, args.max_turns)
            record["metadata"]["case_id"] = case["case_id"]
            saved = store.write(record)
            results.append({**saved["metadata"], "evaluation": saved["evaluation"],
                "calls": [m["tool_calls"][0]["function"]["name"] for m in saved["messages"] if m.get("tool_calls")],
                "input_tokens": sum(g.get("input_tokens",0) for g in saved["generation"]),
                "output_tokens": sum(g.get("output_tokens",0) for g in saved["generation"])})
            print(json.dumps(results[-1]), flush=True)
    finally:
        client.close()
    manifest = {"source_commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "budget": args.budget, "max_turns": args.max_turns,
                "model": str(args.model), "gpu": client.ready["gpu"],
                "grounding_root": str(args.grounding_root) if args.grounding_root else None,
                "cases": results, "scope": "native conversation integration, not a new accuracy benchmark"}
    args.output.with_suffix(".summary.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
