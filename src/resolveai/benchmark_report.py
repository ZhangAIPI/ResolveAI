"""Source-family intervals and integrity checks for paired multi-model evaluation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

VARIANTS = ["Sufficient", "Obtainable", "Missing", "Unavailable"]


def _ci95(values, iterations=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(42)
    samples = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def interval(values, iterations=10000):
    """Public statistics API: one value per source family."""
    return {"families": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "ci95": _ci95(values, iterations)}


def summarize(directory, allow_partial=False):
    """Final run API rejects incomplete evaluation by default."""
    return model_summary(Path(directory), allow_partial)


def summarize_rows(rows):
    if not rows:
        return None
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    answer = {"episodes": len(rows), "source_families": len(families)}
    for key in ["correct", "grounded_correct", "unsupported_decision", "coverage"]:
        values = [statistics.mean(r[key] for r in group) for group in families.values()]
        answer[key] = {"mean": statistics.mean(r[key] for r in rows), "family_ci95": _ci95(values)}
    determinate = [r for r in rows if r["coverage"]]
    answer["selective_error"] = statistics.mean(not r["correct"] for r in determinate) if determinate else None
    answer["completed"] = sum(r["termination"] == "finished" for r in rows)
    answer["terminations"] = dict(Counter(r["termination"] for r in rows))
    answer["errors"] = sum(r["errors"] for r in rows)
    for key in ["requests", "tool_calls", "request_cost", "tool_cost", "input_tokens", "output_tokens", "latency_s", "wall_latency_s", "model_calls"]:
        answer["mean_" + key] = statistics.mean(r[key] for r in rows)
    answer["max_peak_memory_gb"] = max(r["peak_memory_gb"] for r in rows)
    answer["tools"] = dict(Counter(name for r in rows for name in r["calls"]))
    return answer


def paired(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["case_id"]][row["policy"]] = row
    differences = defaultdict(list)
    for pair in grouped.values():
        if set(pair) != {"initial", "agent"}:
            continue
        differences[pair["agent"]["family_id"]].append(
            int(pair["agent"]["grounded_correct"]) - int(pair["initial"]["grounded_correct"]))
    values = [statistics.mean(group) for group in differences.values()]
    return {"grounded_proxy_delta": statistics.mean(values), "family_ci95": _ci95(values),
            "paired_source_families": len(values)} if values else None


def model_summary(directory, allow_partial=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    config = manifest["configuration"]
    expected_families = config.get("selected_families") or config["limit_families"] or config["dataset_manifest"]["families"]
    expected_episodes = expected_families * 4 * 2
    keys = [(r["case_id"], r["policy"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode results")
    complete = len(rows) == expected_episodes
    if not complete and not allow_partial:
        raise ValueError(f"incomplete run: {directory} ({len(rows)}/{expected_episodes})")
    if complete:
        per_family = defaultdict(set)
        for row in rows:
            per_family[row["family_id"]].add((row["variant"], row["policy"]))
        expected = {(v, p) for v in VARIANTS for p in ("initial", "agent")}
        if len(per_family) != expected_families or any(group != expected for group in per_family.values()):
            raise ValueError("missing condition or policy in a completed family")
    result = {"model": config["model_id"], "complete": complete, "expected_episodes": expected_episodes,
        "actual_episodes": len(rows), "source_families": len({r["family_id"] for r in rows}),
        "source_commit": manifest["source_commit"], "dataset_sha256": config["dataset_sha256"],
        "revision": config["revision"], "worker": manifest.get("worker"),
        "overall": {p: summarize_rows([r for r in rows if r["policy"] == p]) for p in ("initial", "agent")},
        "by_variant": {}, "by_category": {}}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        result["by_variant"][variant] = {p: summarize_rows([r for r in selected if r["policy"] == p]) for p in ("initial", "agent")}
        result["by_variant"][variant]["paired"] = paired(selected)
    for category in sorted({r["category"] for r in rows}):
        selected = [r for r in rows if r["category"] == category]
        result["by_category"][category] = {p: summarize_rows([r for r in selected if r["policy"] == p]) for p in ("initial", "agent")}
    return result



def merge_shards(directories, destination):
    """Join disjoint complete runs of one model; preserve the original manifests."""
    manifests = [json.loads((p / "manifest.json").read_text()) for p in directories]
    first = manifests[0]
    shards = first["configuration"]["shards"]
    if {m["configuration"]["shard_index"] for m in manifests} != set(range(shards)):
        raise ValueError("all execution shards are required")
    def common(m):
        return {k:v for k,v in m["configuration"].items() if k not in {"shard_index","selected_families"}}
    if any(common(m) != common(first) for m in manifests):
        raise ValueError("shard configurations differ")
    for path in directories:
        model_summary(path)
    rows = [json.loads(line) for path in directories for line in (path / "results.jsonl").read_text().splitlines()]
    if len({(r["case_id"],r["policy"]) for r in rows}) != len(rows):
        raise ValueError("overlapping shard episodes")
    manifest = dict(first)
    manifest["configuration"] = {**first["configuration"], "shard_index": None,
                                 "selected_families": len({r["family_id"] for r in rows})}
    manifest["execution_shards"] = manifests
    manifest["worker"] = {"workers": [m.get("worker") for m in manifests]}
    destination.mkdir(parents=True,exist_ok=True)
    if (destination / "results.jsonl").exists():
        raise ValueError("merged results already exist")
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (destination / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in sorted(rows,key=lambda r:(r["case_id"],r["policy"]))))
    model_summary(destination)


def markdown(report):
    lines = ["# Fixed public multi-model evaluation", "",
        "300 original MVTec test images, 15 categories and four availability versions. "
        "The evidence-sufficiency metric below is an original-image quality proxy, not human-annotated grounded accuracy.", "",
        "| Model | Condition | Static proxy accuracy | Agent proxy accuracy | Paired change (95% source-family CI) | Agent requests | Completed agent episodes |", 
        "|---|---|---:|---:|---|---:|---:|"]
    for model in report["models"]:
        for variant, data in model["by_variant"].items():
            if not data["initial"] or not data["agent"] or not data["paired"]:
                continue
            delta = data["paired"]
            lo, hi = delta["family_ci95"]
            lines.append(f"| {model['model']} | {variant} | {data['initial']['grounded_correct']['mean']:.1%} | {data['agent']['grounded_correct']['mean']:.1%} | {delta['grounded_proxy_delta']:+.1%} [{lo:+.1%}, {hi:+.1%}] | {data['agent']['mean_requests']:.2f} | {data['agent']['completed']}/{data['agent']['episodes']} |")
    lines += ["", "Safe abstention is not a correct binary visual label. Missing/Unavailable therefore have zero proxy accuracy when originals cannot be cited; examine unsupported decisions, coverage and completion in the JSON summary.", "",
        "| Model | Agent coverage | Unsupported proxy decisions | Selective error | Mean tool cost | Mean request cost | Mean episode seconds | Errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        a = model["overall"]["agent"]
        if a:
            risk = f"{a['selective_error']:.1%}" if a['selective_error'] is not None else 'n/a'
            lines.append(f"| {model['model']} | {a['coverage']['mean']:.1%} | {a['unsupported_decision']['mean']:.1%} | {risk} | {a['mean_tool_cost']:.2f} | {a['mean_request_cost']:.2f} | {a['mean_wall_latency_s']:.2f} | {a['errors']} |")
    lines += ["", "Status: " + ("all runs complete" if report["complete"] else "partial results; evaluation remains incomplete") + ".", "",
        "Intervals resample original-image families, keeping correlated conditions together. "
        "Static policies receive initial evidence only; agents have the same frozen tool pool and budget 12. "
        "All models use a lossless portable function-call adapter and greedy decoding. "
        "Unfinished episodes contribute zero accuracy and retain their actual termination reason.", "",
        "These results measure frozen visual policies and controlled resolution release. "
        "They do not demonstrate cross-view identity, before/after real damage, LoRA/DPO gains, or simulation-training transfer. "
        "See [protocol](../benchmark.md) and the companion JSON for per-category results, usage and manifests."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        merge_shards(args.runs, args.output)
        return
    models = [model_summary(path, args.allow_partial) for path in args.runs]
    if len({m["dataset_sha256"] for m in models}) != 1:
        raise ValueError("models evaluated different frozen datasets")
    report = {"complete": all(m["complete"] for m in models), "models": models,
              "bootstrap": "10000 source-family resamples; seed 42", "metric": "original-image quality proxy"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"complete": report["complete"], "episodes": sum(m["actual_episodes"] for m in models)}), flush=True)


if __name__ == "__main__":
    main()
