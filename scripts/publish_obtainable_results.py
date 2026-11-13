"""Publish complete recorded Obtainable cohorts and the paired-access main table."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from scripts.aggregate_obtainable import aggregate
from scripts.paper_result_table import display_api_workers
from scripts.watch_adaptation_results import publish_changes
from scripts.sync_api_paper import synchronize


def value(group,key,integer=False):
    if group is None or group.get(key) is None:
        return "--"
    return f"{group[key]:.0f}" if integer else f"{group[key]:.1f}"


def model_sort_key(model):
    """API first; rank each group by Obtainable overall accuracy, then label."""
    overall=(model.get("arms", {}).get("interactive_agent", {})
             .get("by_task", {}).get("overall") or {})
    accuracy=overall.get("source_label_accuracy_pct")
    return (0 if model["group"]=="API" else 1,
            accuracy is None, -accuracy if accuracy is not None else 0,
            model["label"])


def render(sufficient, closed, obtainable):
    """One row per model; each access condition retains its own denominator."""
    if sufficient["dataset_sha256"] != obtainable["dataset_sha256"] or set(sufficient["family_ids"]) != set(obtainable["family_ids"]):
        raise ValueError("Access-setting cohorts differ")
    primary={w["name"]:w for w in sufficient["workers"]}
    for w in display_api_workers(closed["workers"]):
        name="luna_decisions_tools" if w["name"]=="luna_decisions" else w["name"]
        primary[name]=w
    models=sorted(obtainable["models"],key=model_sort_key)
    lines=[
      '\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{2.2pt}',
      '\\begin{minipage}{.98\\textwidth}\\centering',
      '\\caption{\\textbf{Zero-shot model comparison.} Sufficient supplies the relevant photographs initially; Obtainable withholds a requestable photograph. Accuracy matches source labels. Photo retrieval is the percentage of Obtainable cases receiving an unseen image or higher-resolution original; repeated images do not count. Retrieval is split by task, while completion and cost include all Obtainable attempts, including abstentions and failed submissions. Within each group, models are ordered by Obtainable overall accuracy.}',
      '\\label{tab:benchmark}',
      '\\begin{tabular}{@{}lrrrrrrrrrr@{}}\\toprule',
      ' & \\multicolumn{2}{c}{Sufficient accuracy} & \\multicolumn{3}{c}{Obtainable accuracy} & \\multicolumn{2}{c}{Photo retrieval} & \\multicolumn{3}{c}{Obtainable interaction}\\\\',
      '\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-11}',
      ' & \\multicolumn{2}{c}{Overall} & \\multicolumn{3}{c}{With tools} & \\multicolumn{2}{c}{(\\%)} & & &\\\\',
      'Model & No tools & With tools & Condition & Identity & Overall & Condition & Identity & Finished (\\%) & Time (s) & Output tokens\\\\\\midrule',
    ]
    previous=None
    for model in models:
        if previous!=model["group"]:
            if previous is not None:lines.append(r"\midrule")
            lines.append(r"\multicolumn{11}{l}{\emph{"+model["group"]+r"}}\\")
            previous=model["group"]
        source=primary[model["name"]]
        sufficient_scores=[]
        for arm in ("initial_text","interactive_agent"):
            row=source["arms"][arm]
            if row["episodes"]!=209:raise ValueError("Sufficient row denominator changed")
            sufficient_scores.append(f"{100*row['label_matches']/209:.1f}")
        agent=model["arms"]["interactive_agent"]["by_task"]
        initial=model["arms"]["initial_text"]["by_task"]["overall"]
        cells=sufficient_scores
        cells += [value(agent[t],"source_label_accuracy_pct") for t in ("state","identity","overall")]
        cells += [value(agent[t],"new_pixel_image_pct") for t in ("state","identity")]
        efficiency=agent["overall"]["efficiency"] if agent["overall"] else None
        cells += [value(agent["overall"],"finished_pct"),value(efficiency,"mean_episode_wall_s"),
                  "--" if model["name"]=="luna_decisions_tools" else value(efficiency,"mean_output_tokens",True)]
        label=model["label"]+(r"$^{\dagger}$" if model["name"]=="luna_decisions_tools" else "")
        lines.append(label+" & "+" & ".join(cells)+r"\\")
    lines += [r"\bottomrule\end{tabular}\end{minipage}\end{table*}"]
    return "\n".join(lines)+"\n"


def changes_for(repo,stage):
    result=json.loads((stage/"obtainable_results.json").read_text())
    reference=json.loads((repo/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    closed=json.loads((repo/"docs/results/openai_candidate_scores.json").read_text())
    return {
      "docs/results/obtainable_results.json":(stage/"obtainable_results.json").read_text(),
      "docs/results/obtainable_coverage.json":(stage/"obtainable_coverage.json").read_text(),
      "docs/experiments/obtainable_results.md":(stage/"obtainable_table.md").read_text(),
      "paper/tables/benchmark_results.tex":render(reference,closed,result),
      "paper/tables/obtainable_results.tex":(stage/"obtainable_table.tex").read_text()
    }


def ready_signature(root):
    """Poll cheap completed keys; do not republish partial API scores."""
    counts={}
    for name in ("gpt61_sol","terra-eval","gpt54_mini","gpt41_mini","luna"):
        database=root/name/"episodes.sqlite"
        if not database.exists():
            counts[name]=False
            continue
        from contextlib import closing
        with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True)) as db:
            n=db.execute("SELECT count(*) FROM episodes WHERE arm='interactive_agent' AND case_id LIKE '%-Obtainable'").fetchone()[0]
        counts[name]=(n==209)
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository",type=Path)
    p.add_argument("runtime",type=Path)
    p.add_argument("--watch",action="store_true")
    args=p.parse_args()
    repo,rt=args.repository,args.runtime
    root=rt/"evaluation/cvpr-paper-priority/reviewer-api-obtainable"
    stage=root/"publication"
    stage.mkdir(exist_ok=True,parents=True)
    lock=(root/"publication.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        signature=ready_signature(root)
        if signature!=previous:
            aggregate(repo,rt,stage)
            changes=changes_for(repo,stage)
            publish_changes(repo,rt,changes,"Report complete Obtainable cohorts alongside supplied-evidence accuracy")
            # synchronize obtains its own paper lock; never hold that lock here.
            for attempt in range(3):
                try:
                    synchronize(repo,rt,"6ae80679a4c7a64df04ac5b8")
                    break
                except RuntimeError as exc:
                    if str(exc)!="Overleaf operation failed: pdf" or attempt==2:raise
                    time.sleep(10)
            previous=signature
            (root/"publication-state.json").write_text(json.dumps({"complete_models":signature},indent=2)+"\n")
            print(json.dumps({"published_complete_models":signature}),flush=True)
        status=json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
        if not args.watch or all(signature.values()) or status.get("phase")=="finished":
            return
        time.sleep(60)


if __name__=="__main__":
    main()
