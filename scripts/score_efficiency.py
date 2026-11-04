"""Summarize measured inference cost on the frozen paper cohort; never rerun models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean
from contextlib import ExitStack, closing
from scripts.score_candidate_labels import episode_databases


def episode_usage(result, trace):
    generated = []
    rejected = 0
    for row in trace["generation"]:
        if row.get("halt"):
            # Context rejection ran the processor, but no model forward/generation.
            if "output_tokens" in row:
                raise ValueError("A rejected request unexpectedly has generated tokens")
            rejected += 1
            continue
        for key in ("input_tokens", "output_tokens", "latency_s"):
            value = row[key]  # Missing measurements are not silently treated as zero.
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid generation measurement: " + key)
        generated.append(row)
    wall = trace["wall_latency_s"]
    if not math.isfinite(wall) or wall < 0 or not math.isclose(wall, result["latency_s"], abs_tol=1e-6):
        raise ValueError("Episode latency differs from the stored result")
    return {
        "task": result["task"],
        "correct": result["verdict"] == result["source_label"],
        "finished": result["termination"] == "finished" and result["verdict"] is not None,
        "wall_s": wall,
        "model_s": sum(row["latency_s"] for row in generated),
        "input_tokens": sum(row["input_tokens"] for row in generated),
        "output_tokens": sum(row["output_tokens"] for row in generated),
        "model_calls": len(generated),
        "rejected_calls": rejected,
        "tool_calls": result["tool_calls"],
        "requests": result["requests"],
    }


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty cohort")
    n = len(rows)
    ordered = sorted(row["wall_s"] for row in rows)
    return {
        "episodes": n,
        "correct": sum(row["correct"] for row in rows),
        "accuracy_pct": 100 * mean(row["correct"] for row in rows),
        "finished_pct": 100 * mean(row["finished"] for row in rows),
        **{"mean_" + key: mean(row[key] for row in rows) for key in (
            "wall_s", "model_s", "input_tokens", "output_tokens", "model_calls",
            "rejected_calls", "tool_calls", "requests")},
        "p50_wall_s": ordered[math.ceil(.5*n)-1],
        "p90_wall_s": ordered[math.ceil(.9*n)-1],
    }


def summarize_run(directory, families, dataset_sha256):
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["configuration"]["dataset_sha256"] != dataset_sha256:
        raise ValueError("Efficiency run uses a different dataset")
    wanted = set(families)
    groups = {}
    with ExitStack() as stack:
        databases = [stack.enter_context(closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)))
                     for path in episode_databases(directory)]
        for arm in ("initial_text", "full_available_text", "interactive_agent"):
            rows, seen = [], set()
            query = "SELECT result, trace FROM episodes WHERE arm=? AND json_extract(result,'$.variant')='Sufficient'"
            for raw_result, raw_trace in (row for db in databases for row in db.execute(query, (arm,))):
                result = json.loads(raw_result)
                family = result["family_id"]
                if family not in wanted:
                    continue
                if family in seen or result["source_label"] not in {"Supported", "Refuted"}:
                    raise ValueError("Duplicate or unlabeled efficiency episode")
                seen.add(family)
                rows.append(episode_usage(result, json.loads(raw_trace)))
            if seen != wanted:
                raise ValueError("Efficiency cohort incomplete")
            groups[arm] = {
                **summarize(rows),
                "by_task": {task: summarize([row for row in rows if row["task"] == task])
                            for task in sorted({row["task"] for row in rows})},
            }
    return {
        "model_id": manifest["configuration"]["model_id"],
        "revision": manifest["configuration"]["revision"],
        "gpu": manifest["worker"]["gpu"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_commit": manifest["source_commit"],
        "versions": manifest["versions"],
        "arms": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("priority", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    workers = []
    for worker in cohort["workers"]:
        candidates = [p/worker["name"] for p in (args.original, args.priority)
                      if (p/worker["name"]/"episodes.sqlite").exists()]
        if len(candidates) != 1:
            raise ValueError("Expected one completed baseline directory: "+worker["name"])
        row = summarize_run(candidates[0], cohort["family_ids"], cohort["dataset_sha256"])
        if row["model_id"] != worker["model_id"] or row["revision"] != worker["revision"]:
            raise ValueError("Efficiency checkpoint differs from accuracy checkpoint")
        for arm, scores in worker["arms"].items():
            if row["arms"][arm]["correct"] != scores["label_matches"]:
                raise ValueError("Accuracy and efficiency cohorts differ")
        workers.append(row)
    report = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Observed per-attempt cost on the same unreviewed source-label cohort as Table 1",
        "dataset_sha256": cohort["dataset_sha256"],
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "family_ids": cohort["family_ids"],
        "common_families": cohort["common_families"],
        "definitions": {
            "denominator": "All attempted episodes, including wrong, uncertain and unfinished runs",
            "wall_s": "Episode wall time including processing, model calls, tools and failed requests; excludes policy-checkpoint loading before the episode",
            "input_tokens": "Sum of native input token IDs over executed model calls, including repeated history; not unique text or FLOPs",
            "output_tokens": "All generated tokens, including malformed or unused outputs",
            "context_rejection": "No forward call or tokens counted for a rejected overlength prompt; its wall time remains",
            "tool_calls": "Environment-recorded nonterminal calls; finish excluded; model parsing errors do not count as executed tools",
            "timing_limits": "Same GPU type, ordinary evaluation runs with other jobs on the host; not isolated throughput measurements",
        },
        "workers": workers,
    }
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
