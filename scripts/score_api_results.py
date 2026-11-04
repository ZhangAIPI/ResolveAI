"""Admit complete API cohorts to Table 1; never reduce the frozen denominator."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts.score_candidate_labels import summarize as summarize_accuracy
from scripts.score_efficiency import summarize as summarize_cost


def api_usage(row, trace, pilot=None):
    generations = [g for g in trace["generation"] if "output_tokens" in g]
    rejected = len(trace["generation"])-len(generations)
    wall = trace["wall_latency_s"]
    if abs(wall-row["latency_s"])>1e-6:
        raise ValueError("Stored latency mismatch")
    replayed = [g for g in generations if g.get("response_reused")]
    if replayed:
        if all("replay_wall_s" in g for g in replayed):
            wall += sum(g["latency_s"]-g["replay_wall_s"] for g in replayed)
        elif pilot is not None and trace["messages"]==pilot["messages"]:
            wall = pilot["wall_latency_s"]
        else:
            raise ValueError("Cached episode needs its original timing provenance")
    return {"task":row["task"],"correct":row["verdict"]==row["source_label"],
            "finished":row["termination"]=="finished" and row["verdict"] is not None,
            "wall_s":wall,"model_s":sum(g["latency_s"] for g in generations),
            "input_tokens":sum(g["input_tokens"] for g in generations),
            "output_tokens":sum(g["output_tokens"] for g in generations),
            "model_calls":len(generations),"rejected_calls":rejected,
            "tool_calls":row["tool_calls"],"requests":row["requests"],
            "reasoning_tokens":sum(g.get("reasoning_tokens",g.get("api_usage",{}).get("output_tokens_details",{}).get("reasoning_tokens",0)) for g in generations)}


def collect(directory, reference, cases):
    manifest=json.loads((directory/"manifest.json").read_text())
    config=manifest["configuration"]
    if config["dataset_sha256"]!=reference["dataset_sha256"]:
        raise ValueError("API dataset changed")
    if set(config["family_ids"])!=set(reference["family_ids"]):
        raise ValueError("API run is not the complete fixed paper cohort")
    arms=config.get("arms",["initial_text","full_available_text","interactive_agent"])
    wanted={(family,arm) for family in reference["family_ids"] for arm in arms}
    selected={}
    with sqlite3.connect((directory/"episodes.sqlite").resolve().as_uri()+"?mode=ro",uri=True) as db:
        for raw,raw_trace in db.execute("SELECT result,trace FROM episodes"):
            row,trace=json.loads(raw),json.loads(raw_trace)
            if row["variant"]!="Sufficient" or row["family_id"] not in reference["family_ids"]:
                continue
            key=row["family_id"],row["arm"]
            if key in selected or key not in wanted:
                raise ValueError("Unexpected or duplicate API attempt")
            case=cases[row["family_id"]]
            if row["source_label"]!=case["provenance_truth"] or row["task"]!=case["task"]:
                raise ValueError("API label or task changed")
            selected[key]=(row,trace)
    if set(selected)!=wanted:
        return None,{"name":directory.name,"attempts":len(selected),"expected":len(wanted),"ready":False}
    pilot={}
    pilot_db=directory.parent/"terra-pilot/episodes.sqlite"
    if directory.name=="terra-eval" and pilot_db.exists():
        with sqlite3.connect(pilot_db.resolve().as_uri()+"?mode=ro",uri=True) as db:
            pilot={(json.loads(raw)["family_id"],json.loads(raw)["arm"]):json.loads(trace)
                   for raw,trace in db.execute("SELECT result,trace FROM episodes")}
    scores,costs={},{}
    for arm in arms:
        data=[selected[family,arm] for family in reference["family_ids"]]
        scores[arm]={**summarize_accuracy([r for r,t in data]),
                     "by_task":{task:summarize_accuracy([r for r,t in data if r["task"]==task])
                                for task in ("state","identity")}}
        usage=[api_usage(r,t,pilot.get((r["family_id"],arm))) for r,t in data]
        costs[arm]={**summarize_cost(usage),
                    "mean_reasoning_tokens":sum(u["reasoning_tokens"] for u in usage)/len(usage),
                    "by_task":{task:summarize_cost([u for u in usage if u["task"]==task])
                               for task in ("state","identity")}}
    worker={"name":directory.name,"model_id":config["model_id"],"arms":scores,"costs":costs,
            "api_config":config["api_config"],"source_commit":manifest["source_commit"],
            "manifest_sha256":hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
            "endpoint":config["api_config"].get("endpoint","responses"),
            "paper_cohort_complete":True,"formal_grounded_accuracy":None}
    return worker,{"name":directory.name,"ready":True,"attempts":len(wanted),"expected":len(wanted)}


def refresh(repository, root, plan_path):
    reference=json.loads((repository/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    plan=json.loads(plan_path.read_text())
    payload=(Path(plan["data"])/"cases.json").read_bytes()
    if hashlib.sha256(payload).hexdigest()!=reference["dataset_sha256"]:
        raise ValueError("Frozen cases changed")
    cases={c["family_id"]:c for c in json.loads(payload) if c["variant"]=="Sufficient"}
    names=["terra-eval"]+[p["name"] for p in json.loads((repository/"docs/experiments/openai-models.json").read_text())["profiles"]]+["luna_decisions_tools"]
    workers,status=[],[]
    for name in names:
        directory=root/name
        if not (directory/"manifest.json").exists() or not (directory/"episodes.sqlite").exists():
            continue
        worker,state=collect(directory,reference,cases)
        status.append(state)
        if worker: workers.append(worker)
    report={"as_of_utc":datetime.now(timezone.utc).isoformat(),"dataset_sha256":reference["dataset_sha256"],
            "family_ids":reference["family_ids"],"scope":reference["scope"],
            "timing":"API network and CPU tools; cached generations retain original model time. Not an equal-hardware GPU speed ranking.",
            "workers":workers,"readiness":status,"formal_grounded_accuracy":None}
    output=repository/"docs/results/openai_candidate_scores.json"
    previous=json.loads(output.read_text()) if output.exists() else None
    if previous:
        old={w["name"]:w for w in previous["workers"]}
        new={w["name"]:w for w in workers}
        if not set(old)<=set(new) or any(old[name]!=new[name] for name in old):
            raise ValueError("Published API measurements changed")
    if previous is None or previous["workers"]!=workers:
        output.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("plan",type=Path)
    args=p.parse_args()
    print(json.dumps(refresh(args.repository,args.root,args.plan)["readiness"],indent=2))
