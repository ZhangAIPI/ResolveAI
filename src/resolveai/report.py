"""Aggregate the pilot with paired family bootstrap intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

METRICS = ("correct", "grounded_correct", "unsupported_decision", "coverage",
           "requests", "tool_cost", "request_cost", "model_calls", "input_tokens",
           "output_tokens", "latency_s", "parse_errors")


def aggregate(rows):
    return {"episodes": len(rows), **{k: statistics.mean(float(r[k]) for r in rows) for k in METRICS}}


def paired_interval(rows, policy, variant, iterations=5000):
    pairs = defaultdict(dict)
    for row in rows:
        if row["variant"] == variant and row["policy"] in ("initial", policy):
            pairs[row["family_id"]][row["policy"]] = row
    differences = [float(p[policy]["correct"]) - float(p["initial"]["correct"])
                   for p in pairs.values() if set(p) == {"initial", policy}]
    if not differences:
        return None
    rng = random.Random(20261004)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(iterations))
    return {"families": len(differences), "accuracy_delta": statistics.mean(differences),
            "ci95": [samples[int(.025 * iterations)], samples[int(.975 * iterations)]],
            "bootstrap_unit": "source-image family", "iterations": iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    keys = [(r["case_id"], r["policy"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate policy/case results")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["variant"])].append(row)
    table = {policy: {variant: aggregate(group) for (p, variant), group in sorted(groups.items()) if p == policy}
             for policy in sorted({r["policy"] for r in rows})}
    comparison = {p: {v: paired_interval(rows, p, v) for v in sorted({r["variant"] for r in rows})}
                  for p in ("fixed_request", "agent")}
    risk = {}
    for policy in sorted({r["policy"] for r in rows}):
        group = [r for r in rows if r["policy"] == policy]
        curve = []
        for threshold in (0, .5, .7, .8, .9, .95):
            covered = [r for r in group if r["coverage"] and r["confidence"] >= threshold]
            curve.append({"threshold": threshold, "coverage": len(covered) / len(group),
                          "risk": None if not covered else 1 - statistics.mean(float(r["correct"]) for r in covered)})
        risk[policy] = curve
    summary = {"families": len({r["family_id"] for r in rows}), "episodes": len(rows),
        "metrics": table, "paired_accuracy_delta_vs_initial": comparison,
        "coverage_risk": risk, "max_peak_memory_gb": max(r["peak_memory_gb"] for r in rows),
        "limitations": ["Small balanced official-test subset; descriptive pilot only.",
            "Resolution release from the same image, not independent views or new photography.",
            "Grounded correctness and unsupported decisions use an original-image access/citation proxy.",
            "Confidence is model self-report, not calibrated probability.",
            "Oracle has privileged evidence access and must be reported separately."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"families": summary["families"], "episodes": summary["episodes"],
                      "obtainable": {p: v.get("Obtainable") for p, v in table.items()}}), flush=True)


if __name__ == "__main__":
    main()
