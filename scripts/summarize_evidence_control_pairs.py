"""Paired source-label outcomes for the completed budget-3 evidence controls.

Usage: python summarize_evidence_control_pairs.py REPO RUNTIME OUTPUT_JSON
No model inference or publication. Requires complete V1 (8,360) and V2 (1,254).
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

MODELS = ("qwen3_8b_base", "qwen3_8b_sftrl")
COMPARISONS = (
    ("adaptive", "adaptive_no_new_photo"),
    ("restore_preview", "adaptive"),
    ("adaptive", "random_order"),
    ("restore_preview", "finish_only"),
)
WITNESSES = ("case-00bb48fa5a6b3e55662e", "case-008fddbd675adc4570ad",
             "case-03415461c3ae41ab58f8", "case-0d15c862cc16bbe4c099")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(a, b, families, seed=20261106, replicates=20000):
    """A minus B; resample paired family outcomes, retaining every attempt."""
    if not families or len(a) != len(b) or len(a) != len(families):
        raise ValueError("Invalid paired cohort")
    a, b = np.asarray(a, dtype=np.int8), np.asarray(b, dtype=np.int8)
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError("Correctness vectors must be binary")
    difference = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), (replicates, len(families)))
    interval = np.percentile(difference[draws].mean(axis=1) * 100, [2.5, 97.5])
    gain = [f for f, x, y in zip(families, a, b) if x and not y]
    loss = [f for f, x, y in zip(families, a, b) if y and not x]
    discordant = len(gain) + len(loss)
    p = min(1.0, 2 * sum(math.comb(discordant, k)
                        for k in range(min(len(gain), len(loss)) + 1))
            / 2 ** discordant) if discordant else 1.0
    return {
        "families": len(families), "a_correct": int(a.sum()), "b_correct": int(b.sum()),
        "both_correct": int((a & b).sum()), "both_incorrect": int(((1-a) & (1-b)).sum()),
        "a_correct_b_incorrect": len(gain), "a_incorrect_b_correct": len(loss),
        "difference_pp": float(difference.mean() * 100),
        "paired_bootstrap_ci95_pp": interval.tolist(), "mcnemar_exact_p": p,
        "gained_family_ids": gain, "lost_family_ids": loss,
    }


def trace_excerpt(row, trace):
    """Record actual calls/responses, without embedding image bytes or interpretations."""
    calls, responses = [], []
    for message in trace["messages"]:
        if message["role"] == "assistant":
            calls.extend(call["function"] for call in message.get("tool_calls", []))
        content = message.get("content")
        if message["role"] != "tool" or not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "text":
                continue
            try:
                value = json.loads(part["text"])
            except (ValueError, KeyError):
                continue
            if isinstance(value, dict):
                responses.append({k: value[k] for k in ("status", "error", "budget") if k in value})
    return {"verdict": row["verdict"], "source_label": row["source_label"],
            "source_label_match": row["source_label_match"], "termination": row["termination"],
            "latency_s": row["latency_s"], "requests": row["requests"],
            "calls": calls, "tool_responses": responses,
            "offered_tools": [t["function"]["name"] for t in trace["tools"]],
            "system_prompt_sha256": hashlib.sha256(trace["messages"][0]["content"].encode()).hexdigest()}


def witness(case):
    from PIL import Image
    evidence = []
    for item in case["evidence"]:
        path = Path(case["asset_root"]) / item["path"]
        with Image.open(path) as pixels:
            dimensions = list(pixels.size)
        evidence.append({"image_id": item["id"], "initial": item["id"] in case["initial"],
                         "camera_view": item["view"], "dimensions": dimensions,
                         "path": str(path), "file_sha256": sha(path)})
    return {"claim": case["claim"], "task": case["task"],
            "source_label": case["provenance_truth"],
            "source_provenance": case["provenance"], "actual_evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("Use at least 1,000 bootstrap replicates")
    sys.path.insert(0, str(args.repo.resolve()))
    controls = importlib.import_module("scripts.summarize_evidence_interventions")
    pool = args.runtime / "evaluation/cvpr-paper-priority"
    plan_path = pool / "gpu-pool-evidence-controls-v1/budget3-plan.json"
    cohort_path = args.repo / "docs/results/cvpr_20261102_candidate_scores.json"
    plan, cohort = json.loads(plan_path.read_text()), json.loads(cohort_path.read_text())
    dataset_path = Path(plan["data"]) / "cases.json"
    cohort["_file_sha256"] = sha(cohort_path)
    families = cohort["family_ids"]
    if (len(families) != 209 or len(set(families)) != 209
            or sha(dataset_path) != plan["dataset_sha256"]
            or cohort["dataset_sha256"] != plan["dataset_sha256"]
            or not set(families) <= set(plan["family_ids"]) or plan["budget"] != 3):
        raise ValueError("Frozen 209-family dataset/cohort mismatch")
    # Reuse strict source/weights/shard/allowance and decoded-RGB initial-input checks.
    complete, _ = controls.collect(plan, cohort, pool / "reviewer-controls-v2",
                                   pool / "evidence-controls-v1", 2)
    if complete is None or not complete["complete_v2"] or not complete["joined_reference_complete"]:
        raise ValueError("Full V1/V2 matrices must complete before paired analysis")
    cases = {c["family_id"]: c for c in json.loads(dataset_path.read_text())
             if c["family_id"] in families and c["variant"] == "Obtainable"}
    if set(cases) != set(families):
        raise ValueError("Missing paired Obtainable cases")
    indexed, excerpts = {}, {}
    for generation in ("evidence-controls-v1", "reviewer-controls-v2"):
        for model in MODELS:
            for db in sorted((pool / generation / model / "budget3").glob("shard-*/episodes.sqlite")):
                with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as connection:
                    for cid, arm, raw, trace_raw in connection.execute(
                            "SELECT case_id,arm,result,trace FROM episodes ORDER BY case_id,arm"):
                        if not cid.endswith("-Obtainable"):
                            continue
                        row = json.loads(raw)
                        key = (model, arm, row["family_id"])
                        if key in indexed or row["family_id"] not in cases:
                            raise ValueError("Duplicate or foreign paired result")
                        indexed[key] = row
                        if row["family_id"] in WITNESSES:
                            excerpts["/".join(key)] = trace_excerpt(row, json.loads(trace_raw))
    report = {
        "scope": "Agreement with unreviewed source-derived labels; not grounded accuracy or human-reviewed sufficiency.",
        "variant": "Obtainable", "family_ids": families,
        "denominator": "All 209 paired attempts, including abstentions and failed submissions.",
        "bootstrap": {"seed": args.seed, "replicates": args.bootstrap, "unit": "case family",
                      "method": "paired percentile bootstrap; central 95% interval; NumPy default_rng",
                      "numpy": np.__version__, "multiple_comparison_adjustment": None},
        "interpretation": {
            "no_new_photo": "Matched initial prompt/tools; access-permission intervention also changes request feedback.",
            "restore_vs_adaptive": "Scripted restoration uses a terminal prompt and finish-only tools; not a pure action-selection intervention.",
            "request_strata": "Selected by the adaptive policy; descriptive, not randomized subgroups.",
            "uncertainty": "Family sampling only; one model-training seed and one random-order seed; unreviewed labels are not validated by bootstrap."},
        "inputs": {"plan_sha256": sha(plan_path), "cohort_sha256": sha(cohort_path),
                   "dataset_sha256": sha(dataset_path), "script_sha256": sha(__file__),
                   "collector_sha256": sha(controls.__file__),
                   "provenance_v1": complete["provenance_v1"], "provenance_v2": complete["provenance_v2"],
                   "native_initial_input_parity": complete["native_initial_input_parity"]},
        "models": {}, "witnesses": {f: witness(cases[f]) for f in WITNESSES},
        "witness_traces": excerpts,
    }
    for model in MODELS:
        results = {}
        for aa, bb in COMPARISONS:
            groups = {}
            for task in ("overall", "state", "identity"):
                selected = [f for f in families if task == "overall" or cases[f]["task"] == task]
                groups[task] = paired(
                    [indexed[model, aa, f]["source_label_match"] for f in selected],
                    [indexed[model, bb, f]["source_label_match"] for f in selected],
                    selected, args.seed, args.bootstrap)
            results[aa + "_minus_" + bb] = groups
        results["restore_minus_adaptive_by_observed_request"] = {}
        for requested in (False, True):
            selected = [f for f in families if bool(indexed[model, "adaptive", f]["requests"]) == requested]
            if selected:
                results["restore_minus_adaptive_by_observed_request"][str(requested).lower()] = paired(
                    [indexed[model, "restore_preview", f]["source_label_match"] for f in selected],
                    [indexed[model, "adaptive", f]["source_label_match"] for f in selected],
                    selected, args.seed, args.bootstrap)
        transitions = Counter()
        for family in families:
            before = indexed[model, "adaptive", family]
            after = indexed[model, "adaptive_no_new_photo", family]
            transitions[(before["verdict"], after["verdict"],
                         bool(before["source_label_match"]),
                         bool(after["source_label_match"]))] += 1
        results["blocked_photo_answer_transitions"] = [
            {"from": before, "to": after, "source_match_before": matched_before,
             "source_match_after": matched_after, "families": count}
            for (before, after, matched_before, matched_after), count
            in sorted(transitions.items(), key=lambda item: repr(item[0]))]
        report["models"][model] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
