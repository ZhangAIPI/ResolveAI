"""Paired smoke evaluation of the repaired public curation pool.

This verifies transport and acquisition behavior, not reviewed grounded accuracy.
Unknown same-category identity candidates are retained without scoring their truth.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import TrajectoryStore, content_for
from resolveai.diagnostic import rich_rollout, text_direct
from resolveai.environment import Environment
from resolveai.pilot import ModelClient


def select(cases):
    buckets = defaultdict(list)
    for case in cases:
        if case["variant"] != "Obtainable":
            continue
        role = case["provenance"].get("construction", case["provenance_truth"])
        buckets[(case["task"], case["category"], role)].append(case)
    chosen = []
    for key, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda c: hashlib.sha256(("repaired-check-v1|" + c["family_id"]).encode()).hexdigest())
        chosen.append(bucket[0])
    return chosen


class Transport:
    def __init__(self, client, adapter, prefix):
        self.client, self.adapter, self.prefix = client, adapter, prefix

    def ask_conversation(self, payload, max_context_tokens=8192):
        return self.client.ask_conversation({**payload, "tool_adapter": self.adapter,
                                            "assistant_prefix": self.prefix}, max_context_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run on an allocated GPU node")
    cases = select(json.loads((args.data / "cases.json").read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        parser.error("Use a fresh output directory")
    config = {
        "protocol": "repaired-check-v1", "model_id": args.model_id, "revision": args.revision,
        "source_sha256": source_digest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256((args.data / "cases.json").read_bytes()).hexdigest(),
        "families": [c["family_id"] for c in cases], "max_turns": 6, "budget": 12,
        "arms": ["initial_text", "supplied_text", "portable_prefix_agent", "auto_plain_agent"],
        "scope": "unreviewed development source-label agreement; unknown identity pairs unscored",
    }
    manifest = locked_manifest(args.output / "manifest.json", config)
    client = ModelClient(args.model, args.device, "auto", assistant_prefix=False)
    manifest["worker"] = client.ready
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    store = TrajectoryStore(args.output / "trajectories.jsonl")
    rows = []
    path = args.output / "results.jsonl"
    try:
        for case in cases:
            original = "original" if case["task"] == "state" else next(
                e["id"] for e in case["evidence"] if e["object"] == "subject-B" and e["view"] == "view-02")
            for arm in config["arms"]:
                env = Environment(case, case["asset_root"], 12)
                fingerprint = env._world.fingerprint
                if arm == "supplied_text":
                    row = next(e for e in case["evidence"] if e["id"] == original)
                    result = env.step({"type": "request_photo", "query": {k: row[k] for k in ["object", "time", "view"]}})
                    if result["status"] != "provided" or original not in env._released:
                        raise RuntimeError("Correct fixed retrieval must work")
                if arm.endswith("text"):
                    record = text_direct(env, case, client)
                    record["tools"] = []
                else:
                    adapter = "portable" if arm.startswith("portable") else client.ready["tool_adapter"]
                    wrapper = Transport(client, adapter, arm.startswith("portable"))
                    record = rich_rollout(env, case, wrapper, 6)
                if env._world.fingerprint != fingerprint:
                    raise RuntimeError("World state changed")
                record.update(evaluation=None, schema_version="resolveai-diagnostic-v1",
                              metadata={"family_id": case["family_id"], "arm": arm,
                                        "annotation_status": "source-derived-unreviewed-development-only"})
                store.write(record)
                verdict = record["decision"]["verdict"] if record.get("decision") else None
                truth = case["provenance_truth"]
                row = {
                    "family_id": case["family_id"], "task": case["task"], "category": case["category"],
                    "construction": case["provenance"].get("construction"), "arm": arm,
                    "source_label": truth, "verdict": verdict,
                    "source_label_match": verdict == truth if truth else None,
                    "termination": record["termination"], "errors": record["errors"],
                    "requests": env.requests, "tool_calls": env.calls,
                    "obtained_original": original in env._released,
                    "new_views": sorted(i for i in env._released if i not in case["initial"] and i != original),
                }
                rows.append(row)
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps({"completed": len(rows), "expected": len(cases)*4, **row}), flush=True)
    finally:
        client.close()
    groups = {}
    for arm in config["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        groups[arm] = {
            "episodes": len(selected), "known_label_cases": sum(r["source_label"] is not None for r in selected),
            "source_label_matches": sum(r["source_label_match"] is True for r in selected),
            "finished": sum(r["termination"] == "finished" for r in selected),
            "errors": sum(r["errors"] for r in selected), "originals_obtained": sum(r["obtained_original"] for r in selected),
            "new_view_cases": sum(bool(r["new_views"]) for r in selected),
            "verdicts": dict(Counter(str(r["verdict"]) for r in selected)),
        }
    summary = {"complete": True, "model_id": args.model_id, "families": len(cases), "episodes": len(rows),
               "metrics": groups, "scope": config["scope"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
