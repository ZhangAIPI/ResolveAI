"""Admit models only after all three settings cover the unchanged paper cohort."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.score_candidate_labels import ARMS, read_rows, summarize


def completed_worker(spec, directory, reference, cases, expected_episodes):
    if not (directory/"manifest.json").exists() or not (directory/"episodes.sqlite").exists():
        return None, {"name":spec["name"],"status":"not_started"}
    manifest = json.loads((directory/"manifest.json").read_text())
    config = manifest["configuration"]
    for key, expected in (("model_id",spec["model_id"]),("revision",spec["revision"]),
                          ("dataset_sha256",reference["dataset_sha256"])):
        if config[key] != expected:
            raise ValueError("Checkpoint or dataset mismatch: "+key)
    rows = read_rows(directory/"episodes.sqlite")
    families = set(reference["family_ids"])
    selected = [row for row in rows if row["variant"]=="Sufficient"
                and row["family_id"] in families and row["arm"] in ARMS]
    indexed = {(row["family_id"],row["arm"]):row for row in selected}
    if len(indexed) != len(selected):
        raise ValueError("Duplicate cohort episode")
    wanted = {(family,arm) for family in reference["family_ids"] for arm in ARMS}
    progress = {
        "name":spec["name"],"raw_episodes":len(rows),"expected_raw_episodes":expected_episodes,
        "paper_cases_by_setting":{arm:sum((family,arm) in indexed
             for family in reference["family_ids"]) for arm in ARMS},
    }
    if set(indexed) != wanted:
        return None, {**progress,"status":"paper_cohort_incomplete"}
    for (family,_),row in indexed.items():
        if row["source_label"] != cases[family]["provenance_truth"]:
            raise ValueError("Stored source label differs from frozen case")
    observed = {(row["family_id"],row["arm"]) for row in rows if row["variant"]=="Sufficient"}
    available = sum(all((family,arm) in observed for arm in ARMS) for family,case in cases.items()
                    if case["provenance_truth"] in {"Supported","Refuted"})
    worker = {
        "available_known_families":available,
        "name":spec["name"],"model_id":spec["model_id"],"revision":spec["revision"],
        "source_commit":manifest["source_commit"],
        "paper_cohort_complete":True,"full_study_complete":len(rows)==expected_episodes,
        "raw_episodes":len(rows),"expected_raw_episodes":expected_episodes,
        "arms":{arm:{
            **summarize([indexed[family,arm] for family in reference["family_ids"]]),
            "by_task":{task:summarize([indexed[family,arm] for family in reference["family_ids"]
                                     if cases[family]["task"]==task])
                       for task in reference["task_counts"]},
        } for arm in ARMS},
    }
    return worker,{**progress,"status":"paper_cohort_complete"}


def refresh(reference, plan, priority, specs):
    payload = (Path(plan["data"])/"cases.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    cases = {c["family_id"]:c for c in json.loads(payload) if c["variant"]=="Sufficient"}
    result = deepcopy(reference)
    status = []
    for spec in specs:
        worker,progress = completed_worker(spec,priority/spec["name"],reference,cases,
                                           len(plan["family_ids"])*4*len(ARMS))
        status.append(progress)
        if worker is None:
            continue
        existing = next((i for i,w in enumerate(result["workers"])
                         if w["model_id"]==worker["model_id"]),None)
        if existing is None:
            result["workers"].append(worker)
        else:
            for arm in ARMS:
                if result["workers"][existing]["arms"][arm] != worker["arms"][arm]:
                    raise ValueError("Previously published scores changed")
            result["workers"][existing] = worker
    if result != reference:
        result["as_of_utc"] = datetime.now(timezone.utc).isoformat()
    return result,status


def markdown_results(report):
    from scripts.paper_result_table import NAMES, ORDER
    lines = [
        "已完成 **"+str(len(report["workers"]))+" 个开源模型**的 Sufficient（初始证据充足）对照，使用同一固定 209 族。",
        "下表是来源标签下的答对率（%），并非独立审核后的证据充分性成绩。",
        "",
        "| 模型 | 无工具 | 有工具 |",
        "| --- | ---: | ---: |",
    ]
    for worker in sorted(report["workers"],key=lambda w:ORDER.index(w["model_id"])):
        values = [100*worker["arms"][arm]["label_agreement"]
                  for arm in ("initial_text","interactive_agent")]
        lines.append("| "+NAMES[worker["model_id"]]+f" | {values[0]:.1f} | {values[1]:.1f} |")
    lines += ["","下表仅报告 Sufficient。Obtainable、任务拆分、训练和效率见[当前实验](docs/experiments/reviewer_evidence_search.zh-CN.md)。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("plan",type=Path)
    parser.add_argument("priority",type=Path)
    parser.add_argument("specs",nargs="+",type=Path)
    args = parser.parse_args()
    original = args.report.read_bytes()
    reference = json.loads(original)
    updated,status = refresh(reference,json.loads(args.plan.read_text()),args.priority,
                             [json.loads(path.read_text()) for path in args.specs])
    if updated != reference:
        if args.report.read_bytes() != original:
            raise ValueError("Report changed during refresh")
        temporary = args.report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(updated,indent=2)+"\n")
        temporary.replace(args.report)
    readme = Path("README.md")
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- benchmark-results-start -->", "<!-- benchmark-results-end -->"
        if start in text and end in text:
            first = text.index(start)+len(start)
            last = text.index(end,first)
            readme.write_text(text[:first]+"\n"+markdown_results(updated)+text[last:])
    print(json.dumps({"workers":len(updated["workers"]),"readiness":status},indent=2))


if __name__ == "__main__":
    main()
