"""Read-only paired evidence-control summaries; publish only complete model matrices.

Writes complete JSON/LaTeX artifacts. Optional publication compiles the paper,
updates the supplementary results, and synchronizes Overleaf without editing
model weights, images, evaluation databases, or running task state.
"""
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import time

ARMS = ("adaptive", "finish_only", "fixed_original", "fixed_order", "random_order")
VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")
LABELS = {"Supported", "Refuted"}
ARM_NAMES = {
    "adaptive": "Adaptive", "finish_only": "Finish only",
    "fixed_original": "Request original", "fixed_order": "Fixed order",
    "random_order": "Random order",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT case_id,arm,result,trace FROM episodes ORDER BY case_id,arm").fetchall()


def configuration_key(config):
    return {k: v for k, v in config.items()
            if k not in {"families", "cases", "shard", "expected_episodes"}}


def matrix_status(root, models, plan, cohort, shards):
    """Validate exact keys before exposing metrics from any model."""
    cases = json.loads((Path(plan["data"]) / "cases.json").read_bytes())
    by_id = {c["case_id"]: c for c in cases if c["family_id"] in cohort["family_ids"]}
    status, manifests = {}, {}
    for model in models:
        common = None
        all_keys = set()
        for shard in range(shards):
            directory = root / model / ("budget" + str(plan["budget"])) / ("shard-" + str(shard))
            expected_families = cohort["family_ids"][shard::shards]
            expected = {(c["case_id"], arm) for c in by_id.values()
                        if c["family_id"] in expected_families for arm in ARMS}
            path, dbpath = directory / "manifest.json", directory / "episodes.sqlite"
            key = model + "/" + str(shard)
            if not path.exists() or not dbpath.exists():
                status[key] = {"completed": 0, "expected": len(expected)}
                continue
            manifest = json.loads(path.read_text())
            config = manifest["configuration"]
            if (config["families"], config["shards"], config["shard"]) != (
                    expected_families, shards, shard):
                raise ValueError("Unexpected cohort/shard assignment: " + key)
            if (config["arms"], config["variants"], config["budget"], config["max_turns"],
                config["max_new_tokens"], config["dataset_sha256"]) != (
                    list(ARMS), list(VARIANTS), plan["budget"], plan["max_turns"],
                    384, plan["dataset_sha256"]):
                raise ValueError("Control settings differ: " + key)
            if config["expected_episodes"] != len(expected):
                raise ValueError("Unexpected denominator: " + key)
            identity = configuration_key(config)
            if common is not None and identity != common:
                raise ValueError("Model shards have different settings: " + model)
            common = identity
            with closing(sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                actual = set(db.execute("SELECT case_id,arm FROM episodes"))
            if not actual <= expected or all_keys & actual:
                raise ValueError("Unexpected or overlapping result keys: " + key)
            all_keys.update(actual)
            status[key] = {"completed": len(actual), "expected": len(expected)}
            manifests[key] = {"path": path, "directory": directory, "value": manifest}
    ready = all(v["completed"] == v["expected"] for v in status.values())
    return ready, status, manifests, by_id


def content_parts(m):
    value=m.get('content',[])
    return value if isinstance(value,list) else []

def pixel_helpers():
    path = Path(__file__).with_name("aggregate_obtainable.py")
    spec = importlib.util.spec_from_file_location("evidence_control_pixel_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def request_info(tr, directory, fingerprint):
    seen=set()
    records=[]
    names={}
    for m in tr['messages']:
        if m.get('role')=='assistant':
            for c in m.get('tool_calls',[]):
                fun=c.get('function',{})
                names[c.get('id')]=fun
        image_refs={Path(x['image_ref']).name for x in content_parts(m) if x.get('type')=='image' and x.get('image_ref')}
        imgs={fingerprint(x,directory) for x in content_parts(m) if x.get('type')=='image' and x.get('image_ref')}
        if m.get('role')=='tool' and (m.get('name')=='request_photo' or names.get(m.get('tool_call_id'),{}).get('name')=='request_photo'):
            objs=[]
            for part in content_parts(m):
                if part.get('type')=='text':
                    try:objs.append(json.loads(part['text']))
                    except (ValueError,TypeError):pass
            status=next((x.get('status') for x in objs if isinstance(x,dict) and 'status' in x),None)
            arg=names.get(m.get('tool_call_id'),{}).get('arguments',{})
            if isinstance(arg,str):
                try:arg=json.loads(arg)
                except ValueError:arg={}
            records.append({'status':status,'novel_images':len(imgs-seen),'returned_images':len(imgs),
               'repeated_only':bool(imgs) and not bool(imgs-seen),'query':arg.get('query',{}),
               'image_refs':sorted(image_refs)})
        seen.update(imgs)
    return records


def measure(rows):
    n = len(rows)
    if not n:
        raise ValueError("Empty score group")
    return {
        "episodes": n,
        "source_label_accuracy": sum(r["verdict"] == r["source_label"] for r in rows) / n,
        "completion_rate": sum(r["termination"] == "finished" for r in rows) / n,
        "request_rate": sum(r["requests"] > 0 for r in rows) / n,
        "new_image_rate": sum(r["new_image"] for r in rows) / n,
        "new_released_record_rate": sum(r["new_released_record"] for r in rows) / n,
        "repeated_image_request_rate": sum(r["repeated_image_request"] for r in rows) / n,
        "failed_request_rate": sum(r["failed_request"] for r in rows) / n,
        "definite_after_failed_request_rate": sum(
            r["failed_request"] and r["verdict"] in LABELS for r in rows) / n,
        "definite_given_failed_request": (
            sum(r["failed_request"] and r["verdict"] in LABELS for r in rows)
            / sum(r["failed_request"] for r in rows)
            if any(r["failed_request"] for r in rows) else None),
        "definite_rate": sum(r["verdict"] in LABELS for r in rows) / n,
        "need_more_evidence_rate": sum(r["verdict"] == "Need more evidence" for r in rows) / n,
        "failure_rate": sum(r["termination"] != "finished" for r in rows) / n,
        "mean_wall_s": sum(r["latency_s"] for r in rows) / n,
        "mean_input_tokens": sum(r["input_tokens"] for r in rows) / n,
        "mean_output_tokens": sum(r["output_tokens"] for r in rows) / n,
        "mean_assistant_turns": sum(r["assistant_turns"] for r in rows) / n,
        "mean_request_calls": sum(r["requests"] for r in rows) / n,
        "fixed_original_not_offered_rate": sum(r["original_not_offered"] for r in rows) / n,
    }


def collect(root, models, plan, cohort, shards):
    ready, status, manifests, cases = matrix_status(root, models, plan, cohort, shards)
    if not ready:
        return None, status
    helper, helper_path = pixel_helpers()
    results, provenance = {}, {}
    shared = None
    for model in models:
        rows = []
        for shard in range(shards):
            key = model + "/" + str(shard)
            entry = manifests[key]
            config = entry["value"]["configuration"]
            comparable = {k: config[k] for k in (
                "experiment", "plan_sha256", "dataset_sha256", "cohort_sha256",
                "source_sha256", "script_sha256", "storage_script_sha256",
                "terminal_script_sha256", "seed", "max_turns", "budget",
                "max_new_tokens", "max_context_tokens", "grounding_manifest")}
            comparable["source_commit"] = entry["value"]["source_commit"]
            if shared is not None and comparable != shared:
                raise ValueError("Control comparison settings differ across models")
            shared = comparable
            row_hash = hashlib.sha256()
            saved_rows = read_rows(entry["directory"] / "episodes.sqlite")
            for cid, arm, raw, trace_raw in saved_rows:
                row, trace, case = json.loads(raw), json.loads(trace_raw), cases[cid]
                if (row["case_id"], row["arm"], row["family_id"], row["task"], row["variant"]) != (
                        cid, arm, case["family_id"], case["task"], case["variant"]):
                    raise ValueError("Saved row metadata differs")
                if row["source_label"] != case["provenance_truth"]:
                    raise ValueError("Saved source label differs")
                if row["source_label_match"] != (row["verdict"] == row["source_label"]):
                    raise ValueError("Saved source-label comparison differs")
                if row["tool_cost"] + row["request_cost"] > plan["budget"] or row["assistant_turns"] > plan["max_turns"]:
                    raise ValueError("Recorded allowance exceeded")
                if trace["metadata"]["case_id"] != cid or trace["metadata"]["arm"] != arm:
                    raise ValueError("Trace identity differs")
                if row["input_tokens"] != sum(g.get("input_tokens", 0) for g in trace["generation"]):
                    raise ValueError("Input-token accounting differs")
                if row["output_tokens"] != sum(g.get("output_tokens", 0) for g in trace["generation"]):
                    raise ValueError("Output-token accounting differs")
                requests = request_info(trace, entry["directory"], helper.image_fingerprint)
                row["new_image"] = any(r["novel_images"] > 0 for r in requests)
                row["new_released_record"] = bool(set(row["released"]) - set(case["initial"]))
                row["repeated_image_request"] = any(r["repeated_only"] for r in requests)
                row["failed_request"] = any(r["status"] in (
                    "not_found", "unable_to_provide", "unavailable", "missing")
                    for r in requests)
                row["original_not_offered"] = (
                    trace["scripted_prefix"].get("skip_reason") == "original_view_not_offered")
                rows.append(row)
                row_hash.update(json.dumps([cid, arm, raw, trace_raw]).encode())
            provenance[key] = {
                "manifest_sha256": digest(entry["path"]), "rows_sha256": row_hash.hexdigest(),
                "episodes": len(saved_rows),
                "configuration": config,
            }
        results[model] = {}
        for arm in ARMS:
            results[model][arm] = {}
            for variant in VARIANTS:
                selected = [r for r in rows if r["arm"] == arm and r["variant"] == variant]
                results[model][arm][variant] = {
                    "overall": measure(selected),
                    "by_task": {task: measure([r for r in selected if r["task"] == task])
                                for task in ("state", "identity")},
                }
    return {
        "complete": True, "families": cohort["family_ids"],
        "source_label_scope": "Unreviewed source-derived labels; not grounded accuracy",
        "denominator": "All planned cases; abstentions and unfinished attempts remain in denominators",
        "request_scope": "Public request options only; fixed_original skips tasks without original",
        "new_image_definition": "Request returned decoded RGB pixels and dimensions absent from prior observations; not evidence sufficiency",
        "new_released_record_definition": "New released image_id relative to the initial set; distinct from pixel novelty",
        "novelty_method_source_sha256": digest(helper_path),
        "summary_source_sha256": digest(Path(__file__)),
        "input_token_scope": "Offered input tokens, including rejected context-limit calls; distinct from executed-input counts",
        "token_scope": "Generated model tokens; scripted calls remain in input but cost zero generated tokens",
        "settings": shared, "models": results, "provenance": provenance,
    }, status


def render(report):
    lines = [
        r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{minipage}{.94\textwidth}\centering",
        r"\caption{\textbf{Evidence-request controls at budget " + str(report["settings"]["budget"]) + r".}",
        r"Accuracy and rates are percentages.}",
        r"\label{tab:evidence-controls}",
        r"\begin{tabular}{@{}llrrrrrr@{}}\toprule",
        r"\multicolumn{8}{l}{\emph{(a) Source-label accuracy (\%)}}\\",
        r"& & \multicolumn{3}{c}{Sufficient} & \multicolumn{3}{c}{Obtainable}\\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
        r"Model & Policy & Condition & Identity & Overall & Condition & Identity & Overall\\\midrule",
    ]
    names = {"qwen3_8b_base": "8B Base", "qwen3_8b_sftrl": "8B SFT+RL"}
    for model, arms in report["models"].items():
        for index, arm in enumerate(ARMS):
            values = []
            for variant in ("Sufficient", "Obtainable"):
                block = arms[arm][variant]
                values += [100 * block["by_task"][task]["source_label_accuracy"]
                           for task in ("state", "identity")]
                values.append(100 * block["overall"]["source_label_accuracy"])
            lines.append((names.get(model, model.replace("_", r"\_")) if index == 0 else "")
                         + " & " + ARM_NAMES[arm] + " & "
                         + " & ".join(f"{x:.1f}" for x in values) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}\par\vspace{6pt}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}\toprule",
        r"\multicolumn{10}{l}{\emph{(b) Search, stopping and interaction cost}}\\",
        r"& & \multicolumn{2}{c}{Obtainable (\%)} & \multicolumn{3}{c}{Unavailable (\%)} & & &\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
        r"Model & Policy & Request & New image & Definite & Need more & Failure & Time (s) & Input & Output\\\midrule",
    ]
    for model, arms in report["models"].items():
        for index, arm in enumerate(ARMS):
            obtainable = arms[arm]["Obtainable"]["overall"]
            unavailable = arms[arm]["Unavailable"]["overall"]
            values = [
                100 * obtainable["request_rate"], 100 * obtainable["new_image_rate"],
                100 * unavailable["definite_rate"],
                100 * unavailable["need_more_evidence_rate"], 100 * unavailable["failure_rate"],
                obtainable["mean_wall_s"], obtainable["mean_input_tokens"],
                obtainable["mean_output_tokens"],
            ]
            cells = [f"{x:.1f}" if i < 6 else str(round(x)) for i, x in enumerate(values)]
            lines.append((names.get(model, model.replace("_", r"\_")) if index == 0 else "")
                         + " & " + ARM_NAMES[arm] + " & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines + [r"\end{tabular}\end{minipage}\end{table*}"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path, help="Output stem for complete JSON and LaTeX")
    parser.add_argument("--models", nargs="+", default=["qwen3_8b_base", "qwen3_8b_sftrl"])
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--publish-repository", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if bool(args.publish_repository) != bool(args.runtime):
        parser.error("Publication requires both --publish-repository and --runtime")
    plan, cohort = json.loads(args.plan.read_text()), json.loads(args.cohort.read_text())
    if digest(Path(plan["data"]) / "cases.json") != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    if cohort["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("Cohort dataset differs")
    previous = None
    while True:
        report, status = collect(args.results, args.models, plan, cohort, args.shards)
        if report is not None:
            if report["settings"]["plan_sha256"] != digest(args.plan):
                raise ValueError("Plan content differs from evaluation")
            if report["settings"]["cohort_sha256"] != digest(args.cohort):
                raise ValueError("Cohort content differs from evaluation")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            args.output.with_suffix(".tex").write_text(render(report))
            if args.publish_repository:
                try:
                    from scripts.watch_adaptation_results import publish_changes
                    from scripts.sync_api_paper import synchronize
                    repo = args.publish_repository
                    appendix = (repo / "paper/appendix.tex").read_text()
                    marker = "% evidence-controls-complete"
                    if marker not in appendix:
                        appendix += (
                            "\n" + marker
                            + "\n" + r"\section{Fixed and Adaptive Evidence Requests}" + "\n"
                            + "These comparisons use the same frozen 209 families and four access versions. "
                            + "All policies share the model, native finish schema and 384-token call allowance; "
                            + "the adaptive policy retains the full tool menu, while scheduled policies end with finish only. "
                            + "Scheduled requests use public options and count against the same eight turns and total cost of three. "
                            + "Random order is seeded by family so all versions share its order. "
                            + "Fixed original skips cases without a publicly offered original view. "
                            + "Accuracy uses source labels and does not establish sufficient evidence.\n"
                            + r"\input{tables/evidence_controls}" + "\n"
                        )
                    changes = {
                        "docs/results/cvpr_20261105_evidence_controls.json":
                            json.dumps(report, indent=2) + "\n",
                        "paper/tables/evidence_controls.tex": render(report),
                        "paper/appendix.tex": appendix,
                    }
                    publish_changes(repo, args.runtime, changes,
                                    "Publish complete fixed and adaptive evidence-request controls")
                    synchronize(repo, args.runtime, "6ae80679a4c7a64df04ac5b8")
                except (RuntimeError, ValueError) as error:
                    print(json.dumps({"publication_waiting": str(error)}), flush=True)
                    if args.once:
                        return
                    time.sleep(30)
                    continue
            print(json.dumps({"status": "complete", "counts": status}), flush=True)
            return
        if status != previous:
            print(json.dumps({"status": "waiting", "counts": status}), flush=True)
            previous = status
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
