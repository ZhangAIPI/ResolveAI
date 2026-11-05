"""Aggregate the exact Table 2 policies on frozen Obtainable cases, without inference.

Only this staging directory is written. All SQLite connections are read-only.
No Sufficient score is recalculated or changed.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


TASKS = {"overall": 209, "state": 153, "identity": 56}
TRAINED = {
    "Qwen2.5-VL-7B": ("qwen25", [
        ("SFT", "qwen25_mixed_sft"),
        ("RL", "qwen25_rl_only"),
        ("SFT+RL", "qwen25_grpo")]),
    "Qwen3-VL-2B": ("qwen3_2b", [
        ("SFT", "qwen3_2b_mixed_sft"),
        ("RL", "qwen3_2b_rl_only"),
        ("SFT+RL", "qwen3_2b_grpo_final")]),
    "Qwen3-VL-8B": ("qwen3", [
        ("SFT", "qwen3_mixed_sft"),
        ("RL", "qwen3_8b_rl_only"),
        ("SFT+RL", "qwen3_8b_grpo_final")])
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(repo, runtime, output):
    helper_path = Path(__file__).with_name("aggregate_obtainable.py")
    spec = importlib.util.spec_from_file_location("baseline_obtainable_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reference_path = repo / "docs/results/cvpr_20261102_candidate_scores.json"
    reference = json.loads(reference_path.read_text())
    families = set(reference["family_ids"])
    assert len(families) == 209
    case_path = runtime / "benchmark-v041/cases.json"
    assert sha(case_path) == reference["dataset_sha256"]
    cases = {x["family_id"]: x for x in json.loads(case_path.read_text())
             if x["family_id"] in families and x["variant"] == "Obtainable"}
    assert Counter(x["task"] for x in cases.values()) == {"state": 153, "identity": 56}
    config_path = runtime / "evaluation/cvpr-paper-priority/adaptation-publication/runs.json"
    runs = {x["name"]: x for x in json.loads(config_path.read_text())["runs"]}
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "resolveai-adaptation-obtainable-v1",
        "scope": "Source-label agreement and recorded behavior on the frozen Obtainable cohort; not grounded accuracy. No inference or database mutation.",
        "variant": "Obtainable", "arm": "interactive_agent",
        "dataset_sha256": reference["dataset_sha256"],
        "family_ids": sorted(families), "task_counts": TASKS,
        "definitions": {
            "accuracy": "Percentage matching the source-derived Supported/Refuted label, counting all 209 attempts. Abstentions and invalid or missing final answers score zero.",
            "finished": "Valid final answer, including Need more evidence.",
            "new_image": "A request returns an RGB shape-and-pixel SHA256 absent from earlier observations. Not independent or sufficient visual evidence.",
            "efficiency": "All frozen-cohort attempts, including failures and abstentions, enter means.",
            "checkpoints": "Exactly the Table 2 policies: validation-selected SFT; fixed completed 64-group RL and SFT+RL checkpoints.",
            "sufficient_results": "Not recomputed or modified by this analysis. An externally supplied scalar for a different access version does not supply Obtainable case predictions."
        },
        "provenance": {
            "reference_path": str(reference_path), "reference_sha256": sha(reference_path),
            "runs_config_path": str(config_path), "runs_config_sha256": sha(config_path),
            "helper_path": str(helper_path), "helper_sha256": sha(helper_path),
            "script_sha256": sha(Path(__file__))
        },
        "policies": []
    }
    ids = [f + "-Obtainable" for f in sorted(families)]
    marks = ",".join("?" for _ in ids)
    for model_label, (base_name, plans) in TRAINED.items():
        base_directory = Path(runs[plans[0][1]]["base"])
        for policy, run_name in [("Base", None), *plans]:
            if policy == "Base":
                directory = base_directory
                report_path = reference_path
                checkpoint = None
                expected_model = {"qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
                                  "qwen3_2b": "Qwen/Qwen3-VL-2B-Instruct",
                                  "qwen3": "Qwen/Qwen3-VL-8B-Instruct"}[base_name]
                databases = [directory / "episodes.sqlite"]
            else:
                run = runs[run_name]
                directory = Path(run["run"])
                report_path = repo / run["public_report"]
                report = json.loads(report_path.read_text())
                assert set(report["family_ids"]) == families
                assert report["dataset_sha256"] == reference["dataset_sha256"]
                checkpoint = report["checkpoint"]
                expected_model = report["base_model_id"]
                databases = sorted((directory / "benchmark").glob("**/episodes.sqlite"))
            if not databases:
                raise ValueError(f"No raw databases for {directory}")
            indexed, manifests = {}, []
            for db_path in databases:
                raw_manifest = (db_path.parent / "manifest.json").read_bytes()
                manifest = json.loads(raw_manifest)
                cfg = manifest["configuration"]
                assert cfg["dataset_sha256"] == reference["dataset_sha256"]
                assert cfg["model_id"].startswith(expected_model)
                if checkpoint is not None:
                    assert manifest["worker"]["lora_adapter"] == checkpoint["adapter"]
                provenance = {
                    "database": str(db_path.relative_to(runtime)),
                    "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "source_commit": manifest.get("source_commit"),
                    "model_id": cfg["model_id"], "model_revision": cfg.get("revision"),
                    "configuration_sha256": helper.digest(helper.canonical(cfg)),
                    "lora_adapter": manifest.get("worker", {}).get("lora_adapter"),
                    "public_evaluation_settings": {k: cfg.get(k) for k in
                        ["budget", "max_turns", "max_context_tokens", "image_max_pixels",
                         "decoding", "tool_adapter", "assistant_prefix"]},
                    "admitted_obtainable_rows": 0
                }
                with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    for raw_result, raw_trace in db.execute(
                            "SELECT result, trace FROM episodes WHERE arm='interactive_agent' AND case_id IN (" + marks + ")", ids):
                        row, trace = json.loads(raw_result), json.loads(raw_trace)
                        family = row["family_id"]
                        assert row["case_id"] == family + "-Obtainable"
                        assert row["variant"] == "Obtainable" and row["arm"] == "interactive_agent"
                        assert row["task"] == cases[family]["task"]
                        assert row["source_label"] == cases[family]["provenance_truth"]
                        item = {"result": row, "trace": trace, "directory": db_path.parent,
                                "api": False, "decisions": False,
                                "provenance": [{"database": str(db_path.relative_to(runtime)),
                                                "trace_sha256": hashlib.sha256(raw_trace.encode()).hexdigest()}]}
                        if not helper.admit(indexed, family, item):
                            raise ValueError("Unexpected replay in exact trained-policy source")
                        provenance["admitted_obtainable_rows"] += 1
                manifests.append(provenance)
            complete = set(indexed) == families
            summaries = {task: helper.summarize(
                [x for f, x in indexed.items() if task == "overall" or cases[f]["task"] == task], count)
                if complete else None for task, count in TASKS.items()}
            if not complete:
                raise ValueError(f"Incomplete policy {model_label} {policy}: {len(indexed)}/209")
            # Each all-case measure must equal its task-count-weighted counterpart.
            for key in ["source_label_accuracy_pct", "finished_pct", "new_pixel_image_pct", "request_pct"]:
                weighted = sum(summaries[t][key] * TASKS[t] for t in ["state", "identity"]) / 209
                assert abs(weighted - summaries["overall"][key]) < 1e-9
            for key in ["mean_episode_wall_s", "mean_input_tokens", "mean_output_tokens", "mean_model_calls"]:
                weighted = sum(summaries[t]["efficiency"][key] * TASKS[t] for t in ["state", "identity"]) / 209
                assert abs(weighted - summaries["overall"]["efficiency"][key]) < 1e-6
            overall = summaries["overall"]
            counts = overall["counts"]
            assert counts["finished"] == counts["definite"] + counts["abstained"]
            assert counts["source_label_correct"] <= counts["definite"] <= counts["finished"] <= 209
            entry = {"model": model_label, "policy": policy, "run_name": run_name or base_name,
                     "complete": complete, "expected": 209, "present": len(indexed),
                     "checkpoint": checkpoint,
                     "report_path": str(report_path), "report_sha256": sha(report_path),
                     "manifests": manifests, "by_task": summaries}
            result["policies"].append(entry)
            print(model_label, policy, counts["source_label_correct"], counts["finished"],
                  counts["episodes_with_new_pixels"], flush=True)
    assert len(result["policies"]) == 12
    lookup = {(x["model"], x["policy"]): x for x in result["policies"]}
    for model, policy, correct in [
            ("Qwen3-VL-8B", "Base", 110), ("Qwen3-VL-2B", "Base", 56),
            ("Qwen2.5-VL-7B", "Base", 0)]:
        assert lookup[model, policy]["by_task"]["overall"]["counts"]["source_label_correct"] == correct
    for model, policy, new_images in [
            ("Qwen3-VL-2B", "SFT+RL", 1), ("Qwen3-VL-8B", "SFT+RL", 92),
            ("Qwen3-VL-8B", "RL", 14), ("Qwen2.5-VL-7B", "SFT+RL", 0)]:
        assert lookup[model, policy]["by_task"]["overall"]["counts"]["episodes_with_new_pixels"] == new_images
    json_path = output / "adaptation_obtainable_results.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Post-training on Obtainable",
        "",
        "All rows use the same 209 families (153 condition, 56 identity) and the interactive interface. Accuracy is source-label agreement, not grounded accuracy. Missing/invalid final answers and abstentions score zero. Finish includes explicit abstentions; efficiency includes every attempt. New image counts requests returning previously unseen RGB pixels, not sufficient evidence. Sufficient scores are untouched.",
        "",
        "| Model | Policy | Condition acc. % | Identity acc. % | Overall acc. % | New image: condition % | New image: identity % | Request % | Finish % | Time s | Output tokens | Model calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for entry in result["policies"]:
        groups = entry["by_task"]
        o, s, i = [groups[x] for x in ["overall", "state", "identity"]]
        e = o["efficiency"]
        nums = [s["source_label_accuracy_pct"], i["source_label_accuracy_pct"], o["source_label_accuracy_pct"],
                s["new_pixel_image_pct"], i["new_pixel_image_pct"], o["request_pct"], o["finished_pct"],
                e["mean_episode_wall_s"]]
        cells = [entry["model"], entry["policy"]] + [f"{x:.1f}" for x in nums]
        cells += [f"{e['mean_output_tokens']:.0f}", f"{e['mean_model_calls']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "adaptation_obtainable_table.md").write_text("\n".join(lines) + "\n")
    print("verified all twelve complete policies; weighted task and efficiency invariants pass", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    aggregate(options.repository, options.runtime, options.output)
