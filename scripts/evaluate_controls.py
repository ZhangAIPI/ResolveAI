"""Paired output-budget and terminal-format controls on a frozen case list."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.evaluate_study import Records, normalize_agent_record
from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import Conversation
from resolveai.diagnostic import text_direct
from resolveai.environment import Environment
from resolveai.pilot import ModelClient
from resolveai.rollout import run


class OutputBudget:
    def __init__(self, client, tokens):
        self.client = client
        self.tokens = tokens

    def ask_conversation(self, payload, max_context_tokens=8192):
        return self.client.ask_conversation(
            {**payload, "max_new_tokens": self.tokens}, max_context_tokens
        )


TERMINAL_SYSTEM = (
    "Make a terminal decision about the stated visual claim using only the supplied images. "
    "Supported requires visible evidence for the claim; Refuted requires a reliable "
    "visible contradiction. Choose Need more evidence when the supplied pixels do not "
    "permit a reliable judgment. A missing original alone does not require abstention. "
    "Use the finish function exactly once, with verdict, citations and links. "
    "Citations use released image_id, source-pixel bbox and source time from the "
    "accompanying metadata. A source_id or derived view_id is not a citation image_id. "
    "For cross-image identity or time claims, include the relevant same_object, "
    "different_object, earlier_than or same_time links between cited source regions. "
    "Need more evidence may have empty citations and links. "
    "For a conjunction, support all parts; one decisive contradiction can refute it. "
    "Judge visual facts, not responsibility."
)


def finish_direct(env, client):
    session = Conversation(env)
    session.tools = [
        tool for tool in session.tools if tool["function"]["name"] == "finish"
    ]
    instruction = (
        " Compare the specified foreground targets. Similar appearance alone does not "
        "establish physical identity; a reliable geometric or distinctive-feature "
        "mismatch may refute it. Submission slots do not encode identity."
        if env._case.get("task") == "identity" else
        " Judge only the named condition. A normal reference depicts a different "
        "specimen, not the target's earlier state."
    )
    session.messages[0]["content"] = TERMINAL_SYSTEM + instruction
    return normalize_agent_record(run(session, client, max_turns=1), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=["text", "finish"], required=True)
    parser.add_argument("--tokens", type=int, default=384)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run on an allocated GPU node")
    if args.tokens < 1:
        parser.error("Output allowance must be positive")
    plan = json.loads(args.plan.read_text())
    data = Path(plan["data"])
    if hashlib.sha256((data / "cases.json").read_bytes()).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    families = set(plan["family_ids"])
    cases = sorted(
        (case for case in json.loads((data / "cases.json").read_text())
         if case["family_id"] in families),
        key=lambda case: case["case_id"],
    )
    suffix = "text" if args.mode == "text" else "finish"
    arms = [f"initial_{suffix}", f"full_available_{suffix}"]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "output-budget-and-terminal-format-v1",
        "mode": args.mode,
        "terminal_instructions": "standalone-v2" if args.mode == "finish" else None,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "dataset_sha256": plan["dataset_sha256"],
        "source_sha256": source_digest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "storage_script_sha256": hashlib.sha256(Path(__file__).with_name("evaluate_study.py").read_bytes()).hexdigest(),
        "core_source_commit": os.environ.get("RESOLVEAI_CORE_COMMIT"),
        "model_id": args.model_id,
        "revision": args.revision,
        "arms": arms,
        "families": plan["family_ids"],
        "cases": len(cases),
        "expected_episodes": len(cases) * len(arms),
        "budget": plan["budget"],
        "max_new_tokens": args.tokens,
        "max_assistant_turns": 1,
        "max_context_tokens": 8192,
        "image_max_pixels": 512 * 512,
        "tool_adapter": "auto",
        "assistant_prefix": True,
        "scope": "Unreviewed behavior controls; not sufficiency accuracy",
        "comparison": (
            "Same direct prompts as baseline, only output allowance changes"
            if args.mode == "text" else
            "Same finish schema and output allowance as agent; initial/full evidence, no search"
        ),
    }
    manifest = locked_manifest(args.output / "manifest.json", config)
    records = Records(args.output)
    client = ModelClient(args.model, 0, "auto", assistant_prefix=True)
    manifest["worker"] = client.ready
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    controlled = OutputBudget(client, args.tokens)
    try:
        for case in cases:
            for arm in arms:
                if records.has(case["case_id"], arm):
                    continue
                env = Environment(case, case["asset_root"], plan["budget"])
                fingerprint = env._world.fingerprint
                if arm.startswith("full_available"):
                    env._released.update(
                        image["id"] for image in case["evidence"] if image["available"]
                    )
                trace = (
                    text_direct(env, case, controlled) if args.mode == "text"
                    else finish_direct(env, controlled)
                )
                if args.mode == "text":
                    trace["tools"] = []
                if fingerprint != env._world.fingerprint:
                    raise RuntimeError("World changed")
                trace.update(evaluation=None, schema_version="resolveai-diagnostic-v1")
                trace["metadata"] = {
                    "family_id": case["family_id"], "case_id": case["case_id"],
                    "arm": arm, "official_split": case["official_split"],
                    "annotation_status": "unreviewed-development",
                }
                verdict = trace["decision"]["verdict"] if trace.get("decision") else None
                row = {
                    "case_id": case["case_id"], "family_id": case["family_id"],
                    "task": case["task"], "variant": case["variant"], "arm": arm,
                    "verdict": verdict, "termination": trace["termination"],
                    "errors": trace["errors"], "requests": env.requests,
                    "tool_calls": env.calls, "released": sorted(env._released),
                    "source_label": case["provenance_truth"],
                    "tool_cost": env.tool_cost, "request_cost": env.request_cost,
                    "latency_s": trace["wall_latency_s"],
                }
                records.save(row, trace)
                count = records.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                print(json.dumps({"completed": count, "expected": config["expected_episodes"],
                                  "arm": arm, "termination": row["termination"]}), flush=True)
    finally:
        client.close()
        rows = [json.loads(v[0]) for v in records.db.execute("SELECT result FROM episodes")]
        summary = {
            "episodes": len(rows), "expected": config["expected_episodes"],
            "complete": len(rows) == config["expected_episodes"],
            "arms": {
                arm: dict(Counter(row["termination"] for row in rows if row["arm"] == arm))
                for arm in arms
            },
            "scope": config["scope"],
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        records.db.close()


if __name__ == "__main__":
    main()
