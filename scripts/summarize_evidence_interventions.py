"""Complete-only summaries for paired photo-access interventions.

No inference or evaluation database writes. Optional publication updates only the
appendix. V2 can be summarized alone; joined comparisons wait for the complete V1
matrix and verify paired native initial prompts and tool schemas.
"""
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

MODELS = ("qwen3_8b_base", "qwen3_8b_sftrl")
NEW_ARMS = ("adaptive_no_new_photo", "restore_preview", "full_available_finish")
OLD_ARMS = ("adaptive", "finish_only", "fixed_original", "fixed_order", "random_order")
VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")
LABELS = {
    "adaptive": "Adaptive", "adaptive_no_new_photo": "Adaptive, no new photo",
    "finish_only": "Finish only", "fixed_original": "Request original",
    "fixed_order": "Fixed order", "random_order": "Random order",
    "restore_preview": "Restore preview", "full_available_finish": "Full available",
}
COMPARE_FIELDS = (
    "dataset_sha256", "cohort_sha256", "source_sha256", "storage_script_sha256",
    "terminal_script_sha256", "seed", "budget", "max_turns", "max_new_tokens",
    "max_context_tokens", "image_max_pixels", "tool_adapter", "assistant_prefix",
    "grounding_manifest",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers():
    path = Path(__file__).with_name("aggregate_obtainable.py")
    spec = importlib.util.spec_from_file_location("intervention_pixel_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def read_database(path, columns="case_id,arm"):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT " + columns + " FROM episodes ORDER BY case_id,arm").fetchall()


def weight_identity(config):
    model = config["model"]
    return {key: model.get(key) for key in (
        "model_id", "base_revision", "config_sha256", "adapter_sha256", "adapter_config_sha256")}


def scan(root, plan, cohort, arms, variants, shards):
    """Reject extra keys/settings even before complete; expose no partial scores."""
    payload = (Path(plan["data"]) / "cases.json").read_bytes()
    cases = {c["case_id"]: c for c in json.loads(payload)
             if c["family_id"] in cohort["family_ids"] and c["variant"] in variants}
    expected_pairs = {(f, v) for f in cohort["family_ids"] for v in variants}
    if {(c["family_id"], c["variant"]) for c in cases.values()} != expected_pairs:
        raise ValueError("Frozen case matrix is incomplete")
    status, entries = {}, {}
    for model in MODELS:
        common, seen = None, set()
        for shard in range(shards):
            directory = root / model / ("budget" + str(plan["budget"])) / ("shard-" + str(shard))
            families = cohort["family_ids"][shard::shards]
            expected = {(cid, arm) for cid, case in cases.items()
                        if case["family_id"] in families for arm in arms}
            key = model + "/" + str(shard)
            manifest, database = directory / "manifest.json", directory / "episodes.sqlite"
            if not manifest.exists() or not database.exists():
                status[key] = {"completed": 0, "expected": len(expected)}
                continue
            record = json.loads(manifest.read_text())
            config = record["configuration"]
            if (config["families"], config["shards"], config["shard"],
                config["arms"], config["variants"], config["expected_episodes"]) != (
                    families, shards, shard, list(arms), list(variants), len(expected)):
                raise ValueError("Unexpected shard/arm/case assignment: " + key)
            if (config["dataset_sha256"], config["budget"], config["max_turns"],
                config["max_new_tokens"], config["max_context_tokens"], config["image_max_pixels"]) != (
                    plan["dataset_sha256"], plan["budget"], plan["max_turns"], 384, 8192, 262144):
                raise ValueError("Inference settings differ: " + key)
            if config["cohort_sha256"] != cohort["_file_sha256"]:
                raise ValueError("Cohort file changed")
            if config["cases"] != len(families) * len(variants):
                raise ValueError("Manifest case count differs")
            comparison = {k: v for k, v in config.items()
                          if k not in {"families", "cases", "shard", "expected_episodes"}}
            comparison["_source_commit"] = record["source_commit"]
            if common is not None and comparison != common:
                raise ValueError("Shard configurations differ")
            common = comparison
            actual = set(read_database(database))
            if not actual <= expected or seen & actual:
                raise ValueError("Unexpected or overlapping recorded keys")
            seen.update(actual)
            status[key] = {"completed": len(actual), "expected": len(expected)}
            entries[key] = {"directory": directory, "manifest": record,
                            "manifest_sha256": sha256(manifest)}
    return all(v["completed"] == v["expected"] for v in status.values()), status, entries, cases


def load_complete(entries, cases, arms, plan, helper):
    indexed, provenance = {}, {}
    for key, entry in entries.items():
        model = key.split("/")[0]
        directory = entry["directory"]
        digest, count = hashlib.sha256(), 0
        for cid, arm, raw, trace_raw in read_database(directory / "episodes.sqlite", "case_id,arm,result,trace"):
            case = cases[cid]
            if case["variant"] != "Obtainable":
                continue
            row, trace = json.loads(raw), json.loads(trace_raw)
            if (row["case_id"], row["family_id"], row["task"], row["variant"], row["arm"]) != (
                    cid, case["family_id"], case["task"], "Obtainable", arm):
                raise ValueError("Recorded episode identity differs")
            if row["source_label"] != case["provenance_truth"]:
                raise ValueError("Source label differs")
            if row["source_label_match"] != (row["verdict"] == row["source_label"]):
                raise ValueError("Source-label comparison differs")
            decision = trace.get("decision")
            if (decision.get("verdict") if decision else None) != row["verdict"]:
                raise ValueError("Recorded decision differs")
            if trace["metadata"]["case_id"] != cid or trace["metadata"]["arm"] != arm:
                raise ValueError("Trace identity differs")
            if row["request_cost"] + row["tool_cost"] > plan["budget"] or row["assistant_turns"] > plan["max_turns"]:
                raise ValueError("Recorded allowance exceeded")
            for field in ("input_tokens", "output_tokens"):
                if row[field] != sum(g.get(field, 0) for g in trace["generation"]):
                    raise ValueError("Recorded token accounting differs")
            identity = (model, arm, case["family_id"])
            if identity in indexed:
                raise ValueError("Duplicate episode")
            trace_sha = hashlib.sha256(trace_raw.encode()).hexdigest()
            indexed[identity] = {
                "result": row, "trace": trace, "directory": directory,
                "provenance": [{"database": str(directory / "episodes.sqlite"), "trace_sha256": trace_sha}],
            }
            digest.update(canonical([cid, arm, raw, trace_raw]).encode())
            count += 1
        provenance[key] = {
            "manifest_sha256": entry["manifest_sha256"], "obtainable_rows_sha256": digest.hexdigest(),
            "obtainable_episodes": count, "configuration": entry["manifest"]["configuration"],
        }
    return indexed, provenance


def initial_input(item, helper):
    """Compare the actual pre-answer input, independent of PNG filenames."""
    messages = []
    for message in item["trace"]["messages"]:
        if message.get("role") == "assistant":
            break
        current = json.loads(json.dumps(message))
        if isinstance(current.get("content"), list):
            for part in current["content"]:
                if part.get("type") == "image":
                    fingerprint = helper.image_fingerprint(part, item["directory"])
                    part.pop("image_ref", None)
                    part["decoded_rgb_sha256"] = fingerprint
        messages.append(current)
    return canonical({"messages": messages, "tools": item["trace"]["tools"]})


def metric_groups(indexed, model, arm, families, cases, helper):
    output = {}
    for task in ("overall", "state", "identity"):
        selected = [f for f in families if task == "overall" or cases[f + "-Obtainable"]["task"] == task]
        items = [indexed[(model, arm, family)] for family in selected]
        value = helper.summarize(items, len(selected))
        if value is None:
            raise ValueError("Incomplete metric group")
        # Include offered input tokens even when a context-limit check rejects a call.
        value["efficiency"].update(
            mean_episode_wall_s=sum(x["result"]["latency_s"] for x in items) / len(items),
            mean_input_tokens=sum(x["result"]["input_tokens"] for x in items) / len(items),
            mean_output_tokens=sum(x["result"]["output_tokens"] for x in items) / len(items),
            input_accounting="All recorded offered input tokens, including context-limit rejections; unlike executed-input counts, rejected calls are retained",
        )
        output[task] = value
    return output


def collect(plan, cohort, new_root, old_root, shards, annotation_audit=None):
    helper, helper_path = helpers()
    ready, progress, entries, cases = scan(new_root, plan, cohort, NEW_ARMS, ("Obtainable",), shards)
    if not ready:
        return None, {"v2": progress}
    shared = None
    for entry in entries.values():
        config = entry["manifest"]["configuration"]
        identity = {k: config[k] for k in (*COMPARE_FIELDS, "script_sha256", "experiment")}
        identity["source_commit"] = entry["manifest"]["source_commit"]
        if shared is not None and identity != shared:
            raise ValueError("V2 models do not share source/configuration")
        shared = identity
    new, provenance = load_complete(entries, cases, NEW_ARMS, plan, helper)
    annotation_reference = {"verified_action_equivalence": False, "independent_oracle_timing": None}
    if annotation_audit is not None:
        audit = json.loads(Path(annotation_audit).read_text())
        records = audit["obtainable_records"]
        if (audit["dataset_sha256"] != plan["dataset_sha256"]
            or audit["controller_sha256"] != shared["script_sha256"]
            or len(records) != len(cohort["family_ids"])
            or {r["family_id"] for r in records} != set(cohort["family_ids"])
            or not all(r["byte_identical_calls"] for r in records)):
            raise ValueError("Annotation action-equivalence audit differs")
        annotation_reference.update(verified_action_equivalence=True,
            families=len(records), audit_sha256=sha256(annotation_audit))
    report = {
        "schema_version": "resolveai-evidence-interventions-v1",
        "complete_v2": True, "joined_reference_complete": False,
        "variant": "Obtainable", "family_ids": cohort["family_ids"],
        "scope": "Source-label agreement and observed requests; independent sufficiency metrics are unavailable",
        "new_image_definition": "Request-returned decoded RGB shape and pixel hash absent from all prior observations",
        "pixel_helper_sha256": sha256(helper_path), "models": {}, "provenance_v2": provenance,
        "annotation_reference": annotation_reference,
        "comparisons": {"no_new_photo": "Photo-release availability ablation; request feedback can change",
                        "full_available": "Privileged access reference, not an equal-access policy",
                        "annotation_reference": "Action equivalence is claimed only when its versioned structural audit is supplied; no separate oracle timing"},
    }
    for model in MODELS:
        report["models"][model] = {arm: metric_groups(
            new, model, arm, cohort["family_ids"], cases, helper) for arm in NEW_ARMS}
    old_ready, old_progress, old_entries, old_cases = scan(
        old_root, plan, cohort, OLD_ARMS, VARIANTS, shards)
    progress = {"v2": progress, "v1": old_progress}
    if not old_ready:
        return report, progress
    old, old_provenance = load_complete(old_entries, old_cases, OLD_ARMS, plan, helper)
    parity = {}
    for model in MODELS:
        for shard in range(shards):
            a = old_entries[model + "/" + str(shard)]["manifest"]["configuration"]
            b = entries[model + "/" + str(shard)]["manifest"]["configuration"]
            if any(a[k] != b[k] for k in COMPARE_FIELDS) or weight_identity(a) != weight_identity(b):
                raise ValueError("V1/V2 source, weights, cohort or inference settings differ")
            wa = old_entries[model + "/" + str(shard)]["manifest"].get("worker", {})
            wb = entries[model + "/" + str(shard)]["manifest"].get("worker", {})
            if any(wa.get(k) != wb.get(k) for k in (
                "gpu", "model_type", "processor", "tool_adapter", "max_pixels",
                "min_pixels", "processor_options")):
                raise ValueError("V1/V2 worker settings differ")
        matched = 0
        for family in cohort["family_ids"]:
            a = old[(model, "adaptive", family)]
            b = new[(model, "adaptive_no_new_photo", family)]
            if initial_input(a, helper) != initial_input(b, helper):
                raise ValueError("Paired native initial prompts/tools differ: " + family)
            matched += 1
        parity[model] = {"families": matched, "identical_initial_inputs": matched}
        for arm in OLD_ARMS:
            report["models"][model][arm] = metric_groups(
                old, model, arm, cohort["family_ids"], cases, helper)
    report.update(joined_reference_complete=True, provenance_v1=old_provenance,
                  native_initial_input_parity=parity)
    return report, progress


def render(report):
    if not report["joined_reference_complete"]:
        raise ValueError("Joined table requires complete references")
    lines = [
        r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Photo-access controls on Obtainable cases.}",
        r"Accuracy and rates are percentages.}",
        r"\label{tab:evidence-interventions}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}\toprule",
        r"& & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{New image} & & & &\\",
        r"Model & Policy & Cond. & Ident. & Cond. & Ident. & Finish & Time (s) & Input & Output\\\midrule",
    ]
    order = ("adaptive", "adaptive_no_new_photo", "finish_only", "fixed_original",
             "fixed_order", "random_order", "restore_preview", "full_available_finish")
    for model in MODELS:
        for i, arm in enumerate(order):
            values = report["models"][model][arm]
            overall, state, identity = (values[t] for t in ("overall", "state", "identity"))
            costs = overall["efficiency"]
            numbers = [state["source_label_accuracy_pct"], identity["source_label_accuracy_pct"],
                       state["new_pixel_image_pct"], identity["new_pixel_image_pct"],
                       overall["finished_pct"], costs["mean_episode_wall_s"],
                       costs["mean_input_tokens"], costs["mean_output_tokens"]]
            formatted = [f"{x:.1f}" if j < 6 else str(round(x)) for j, x in enumerate(numbers)]
            name = ("8B Base" if model.endswith("base") else "8B SFT+RL") if i == 0 else ""
            lines.append(name + " & " + LABELS[arm] + " & " + " & ".join(formatted) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines + [r"\end{tabular}\end{table*}"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--annotation-audit", type=Path)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--publish-repository", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if bool(args.publish_repository) != bool(args.runtime):
        parser.error("Publication requires repository and runtime")
    plan, cohort = json.loads(args.plan.read_text()), json.loads(args.cohort.read_text())
    cohort["_file_sha256"] = sha256(args.cohort)
    if sha256(Path(plan["data"]) / "cases.json") != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    if cohort["dataset_sha256"] != plan["dataset_sha256"] or len(set(cohort["family_ids"])) != len(cohort["family_ids"]):
        raise ValueError("Invalid frozen cohort")
    if not set(cohort["family_ids"]) <= set(plan["family_ids"]) or not 1 <= args.shards <= len(cohort["family_ids"]):
        raise ValueError("Invalid cohort/shard assignment")
    previous = None
    while True:
        report, progress = collect(plan, cohort, args.results, args.reference_root,
                                   args.shards, args.annotation_audit)
        if report is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            if report["joined_reference_complete"]:
                table = render(report)
                args.output.with_suffix(".tex").write_text(table)
                if args.publish_repository:
                    try:
                        sys.path.insert(0, str(args.publish_repository))
                        from scripts.watch_adaptation_results import publish_changes
                        from scripts.sync_api_paper import synchronize
                        repo = args.publish_repository
                        appendix = (repo / "paper/appendix.tex").read_text()
                        marker = "% evidence-interventions-complete"
                        if marker not in appendix:
                            appendix += ("\n" + marker + "\n"
                                         + r"\section{Controlling Photo Access}" + "\n"
                                         + r"\input{tables/evidence_interventions}" + "\n")
                        publish_changes(repo, args.runtime, {
                            "docs/results/cvpr_20261105_evidence_interventions.json": json.dumps(report, indent=2) + "\n",
                            "paper/tables/evidence_interventions.tex": table,
                            "paper/appendix.tex": appendix,
                        }, "Report complete matched photo-access interventions")
                        synchronize(repo, args.runtime, "6ae80679a4c7a64df04ac5b8")
                    except (RuntimeError, ValueError) as error:
                        print(json.dumps({"publication_waiting": str(error)}), flush=True)
                        if args.once:
                            return
                        time.sleep(30)
                        continue
                print(json.dumps({"status": "complete", "progress": progress}), flush=True)
                return
        if progress != previous:
            print(json.dumps({"status": "waiting", "v2_complete": report is not None,
                              "progress": progress}), flush=True)
            previous = progress
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
