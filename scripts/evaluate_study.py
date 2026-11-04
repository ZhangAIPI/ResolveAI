"""Resumable model evaluation on all release versions of the human-study families."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import Conversation, TrajectoryStore
from resolveai.diagnostic import text_direct
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding
from resolveai.pilot import ModelClient
from resolveai.model_processing import processor_options
from resolveai.rollout import run

ARMS = ("initial_text", "full_available_text", "interactive_agent")


def normalize_agent_record(trace, env):
    """Native rollouts keep terminal fields in metadata; direct text uses top-level fields."""
    trace["termination"] = trace["metadata"]["termination"]
    trace["errors"] = trace["metadata"]["errors"]
    trace["requests"] = env.requests
    trace["tool_calls"] = env.calls
    return trace


class Records:
    """Commit each result and its actual trace in one transaction."""

    def __init__(self, output):
        self.store = TrajectoryStore(output / "pending-trace.jsonl")
        self.db = sqlite3.connect(output / "episodes.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS episodes (case_id TEXT, arm TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,arm))"
        )
        self.db.commit()
        self._summary = {"episodes": 0, "arms": {arm: {
            "episodes": 0, "finished": 0, "errors": 0, "requests": 0,
            "known_source_labels": 0, "source_label_matches": 0,
        } for arm in ARMS}}
        for (raw,) in self.db.execute("SELECT result FROM episodes"):
            self._count(json.loads(raw))

    def _count(self, row):
        group = self._summary["arms"][row["arm"]]
        self._summary["episodes"] += 1
        group["episodes"] += 1
        group["finished"] += row["termination"] == "finished"
        group["errors"] += row["errors"]
        group["requests"] += row["requests"]
        group["known_source_labels"] += row["source_label"] is not None
        group["source_label_matches"] += row["source_label_match"] is True

    def has(self, case_id, arm):
        return (
            self.db.execute(
                "SELECT 1 FROM episodes WHERE case_id=? AND arm=?", (case_id, arm)
            ).fetchone()
            is not None
        )

    def save(self, row, trace):
        saved = self.store.write(trace)
        with self.db:
            self.db.execute(
                "INSERT INTO episodes VALUES (?,?,?,?)",
                (row["case_id"], row["arm"], json.dumps(row), json.dumps(saved)),
            )
        self._count(row)
        self.store.output.unlink(missing_ok=True)

    def summary(self):
        # A single writer owns this database; resume rebuilds counters once.
        return deepcopy(self._summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--grounding-root", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--priority-cohort", type=Path, help="Order these fixed families first; retain every planned case")
    args = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Run on an allocated GPU node")
    plan = json.loads(args.plan.read_text())
    data = Path(plan["data"])
    if (
        hashlib.sha256((data / "cases.json").read_bytes()).hexdigest()
        != plan["dataset_sha256"]
    ):
        raise ValueError("Frozen dataset changed")
    if not 0 <= args.shard < args.shards:
        p.error("Invalid shard")
    selected = [
        f for i, f in enumerate(plan["family_ids"]) if i % args.shards == args.shard
    ]
    cases = [
        c
        for c in json.loads((data / "cases.json").read_text())
        if c["family_id"] in selected
    ]
    priority = None
    if args.priority_cohort:
        priority = json.loads(args.priority_cohort.read_text())
        if priority["dataset_sha256"] != plan["dataset_sha256"] or not set(priority["family_ids"]) <= set(plan["family_ids"]):
            raise ValueError("Priority cohort differs from the frozen study")
    cases.sort(key=lambda c: (priority is not None and c["family_id"] not in priority["family_ids"], c["case_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "study-model-eval-v1",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "dataset_sha256": plan["dataset_sha256"],
        "source_sha256": source_digest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_id": args.model_id,
        "revision": args.revision,
        "shards": args.shards,
        "shard": args.shard,
        "arms": list(ARMS),
        "families": selected,
        "cases": len(cases),
        "budget": plan["budget"],
        "max_turns": plan["max_turns"],
        "tool_adapter": "auto",
        "assistant_prefix": True,
        "decoding": {"do_sample": False, "text_max_new_tokens": 96, "tool_max_new_tokens": 384},
        "max_context_tokens": 8192,
        "image_max_pixels": 512 * 512,
        "processor_options": processor_options(args.model),
        "grounding_manifest": json.loads((args.grounding_root / "manifest.json").read_text()),
        "scope": "Unreviewed diagnosis; actual trajectories can be regraded after human review",
    }
    if priority is not None:
        config["priority_cohort_sha256"] = hashlib.sha256(args.priority_cohort.read_bytes()).hexdigest()
    manifest = locked_manifest(args.output / "manifest.json", config)
    records = Records(args.output)
    backend = FrozenGrounding(args.grounding_root, device="cuda:0")
    client = ModelClient(args.model, 0, "auto", assistant_prefix=True)
    manifest["worker"] = client.ready
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for case in cases:
            for arm in ARMS:
                if records.has(case["case_id"], arm):
                    continue
                env = Environment(
                    case, case["asset_root"], plan["budget"], grounding_backend=backend
                )
                fingerprint = env._world.fingerprint
                if arm == "full_available_text":
                    env._released.update(
                        e["id"] for e in case["evidence"] if e["available"]
                    )
                if arm.endswith("text"):
                    trace = text_direct(env, case, client)
                    trace["tools"] = []
                else:
                    trace = normalize_agent_record(
                        run(Conversation(env), client, plan["max_turns"]), env
                    )
                if fingerprint != env._world.fingerprint:
                    raise RuntimeError("World changed")
                trace.update(evaluation=None, schema_version="resolveai-diagnostic-v1")
                trace["metadata"] = {
                    "family_id": case["family_id"],
                    "case_id": case["case_id"],
                    "arm": arm,
                    "official_split": case["official_split"],
                    "annotation_status": "unreviewed-development",
                }
                verdict = (
                    trace["decision"]["verdict"] if trace.get("decision") else None
                )
                row = {
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "task": case["task"],
                    "variant": case["variant"],
                    "arm": arm,
                    "verdict": verdict,
                    "termination": trace["termination"],
                    "errors": trace["errors"],
                    "requests": env.requests,
                    "tool_calls": env.calls,
                    "released": sorted(env._released),
                    "source_label": case["provenance_truth"],
                    "source_label_match": verdict == case["provenance_truth"]
                    if case["provenance_truth"]
                    else None,
                    "tool_cost": env.tool_cost,
                    "request_cost": env.request_cost,
                    "latency_s": trace["wall_latency_s"],
                }
                records.save(row, trace)
                print(
                    json.dumps(
                        {
                            "completed": records.summary()["episodes"],
                            "expected": len(cases) * 3,
                            "case_id": case["case_id"],
                            "arm": arm,
                            "termination": trace["termination"],
                        }
                    ),
                    flush=True,
                )
    finally:
        client.close()
        summary = records.summary()
        summary.update(
            complete=summary["episodes"] == len(cases) * 3,
            model_id=args.model_id,
            expected=len(cases) * 3,
            scope=config["scope"],
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        records.db.close()


if __name__ == "__main__":
    main()
