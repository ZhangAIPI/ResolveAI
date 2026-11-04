"""Reproduce paired answer changes and selected examples from existing episode records."""
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np


def rows(directory, families):
    files = [directory/"episodes.sqlite"] if (directory/"episodes.sqlite").exists() else sorted(directory.glob("shard-*/episodes.sqlite"))
    output = {}
    for path in files:
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as db:
            for (raw,) in db.execute("SELECT result FROM episodes"):
                row = json.loads(raw)
                if row["family_id"] not in families:
                    continue
                key = row["family_id"],row["variant"],row["arm"]
                if key in output:
                    raise ValueError("Duplicate episode")
                output[key] = row,path
    return output


def paired(first, second, families, task=None):
    changes = Counter()
    differences = []
    for family in sorted(families):
        a = first[family]
        b = second[family]
        if task and a["task"] != task:
            continue
        before,after = bool(a["source_label_match"]),bool(b["source_label_match"])
        changes[("correct" if before else "wrong")+"_"+("correct" if after else "wrong")] += 1
        differences.append(int(after)-int(before))
    x = np.array(differences,dtype=np.int8)
    rng = np.random.default_rng(20261104)
    boot = x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)*100
    return {"families":len(x),"counts":dict(changes),"difference_pp":float(100*x.mean()),
            "paired_bootstrap_95_pct":np.percentile(boot,[2.5,97.5]).tolist(),"bootstrap_seed":20261104}


def analyze(repo, runtime):
    root = runtime/"evaluation/cvpr-paper-priority"
    reference = json.loads((repo/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    families = set(reference["family_ids"])
    cases = {c["case_id"]:c for c in json.loads((runtime/"benchmark-v041/cases.json").read_text())}
    directories = {"base2":root/"qwen3_2b","base8":runtime/"evaluation/cvpr-20261101/qwen3",
                   "sft2":root/"qwen3_2b_mixed_sft/benchmark","rl2":root/"qwen3_2b_grpo_final/benchmark",
                   "sft7":root/"qwen25_mixed_sft/benchmark","rl7":root/"qwen25_grpo_final/benchmark",
                   "rl8":root/"qwen3_8b_grpo_final/benchmark"}
    data = {name:rows(directory,families) for name,directory in directories.items()}
    result = {"scope":"Paired source-label outcomes; no inference rerun or reviewed-sufficiency claim",
              "dataset_sha256":reference["dataset_sha256"],"family_ids":sorted(families),"paired":{},"examples":{}}
    for name in ("base2","base8"):
        a = {f:data[name][(f,"Sufficient","initial_text")][0] for f in families}
        b = {f:data[name][(f,"Sufficient","interactive_agent")][0] for f in families}
        result["paired"][name] = {task:paired(a,b,families,None if task=="overall" else task)
                                  for task in ("overall","state","identity")}
    for size in ("2","7"):
        a = {f:data["sft"+size][(f,"Sufficient","interactive_agent")][0] for f in families}
        b = {f:data["rl"+size][(f,"Sufficient","interactive_agent")][0] for f in families}
        result["paired"]["rl_after_sft"+size] = paired(a,b,families)
    targets = [("bottle2","rl2","case-52a69b7c898d12a8e0a5","Obtainable"),
               ("bottle8","rl8","case-52a69b7c898d12a8e0a5","Obtainable"),
               ("vehicle","base8","case-cc1dcd4afe1b98c554c6","Sufficient"),
               ("screw","base8","case-035e4ca9730fd6a3fcab","Sufficient")]
    for name,model,family,variant in targets:
        key=family,variant,"interactive_agent"
        row,path=data[model][key]
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as db:
            raw=db.execute("SELECT trace FROM episodes WHERE case_id=? AND arm=?",(row["case_id"],key[2])).fetchone()[0]
        trace=json.loads(raw)
        actions=[]
        for message in trace["messages"]:
            if message.get("role")!="assistant":
                continue
            for call in message.get("tool_calls",[]):
                function=call["function"]
                arguments=function.get("arguments",{})
                arguments=json.loads(arguments) if isinstance(arguments,str) else arguments
                actions.append({"name":function["name"],"arguments":arguments})
        case=cases[row["case_id"]]
        result["examples"][name]={"case_id":row["case_id"],"claim":case["claim"],"result":row,
                                 "actions":actions,"trace_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                                 "source_database":str(path.relative_to(runtime))}
    assert result["paired"]["base8"]["overall"]["counts"]["wrong_correct"]==38
    assert result["paired"]["base8"]["overall"]["counts"]["correct_wrong"]==22
    assert result["paired"]["base2"]["overall"]["counts"]["wrong_correct"]==7
    assert result["paired"]["base2"]["overall"]["counts"]["correct_wrong"]==48
    return result,cases


def figure_assets(runtime, analysis, cases):
    """Use original source bytes and a vector plot; no generated or retouched photographs."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    assets={}
    selected = [
        ("case-52a69b7c898d12a8e0a5-Obtainable","preview","case-bottle-preview.png"),
        ("case-52a69b7c898d12a8e0a5-Obtainable","original","case-bottle-original.png"),
        ("case-cc1dcd4afe1b98c554c6-Sufficient","photo-6f948d8055dc1737c205","case-vehicle-a.jpg"),
        ("case-cc1dcd4afe1b98c554c6-Sufficient","photo-0a79f3148c6f7adf7798","case-vehicle-b.jpg")]
    analysis["figure_sources"]=[]
    for case_id,image_id,name in selected:
        case=cases[case_id]
        image=next(e for e in case["evidence"] if e["id"]==image_id)
        path=Path(case["asset_root"])/image["path"]
        content=path.read_bytes()
        assets["paper/figures/"+name]=content
        analysis["figure_sources"].append({"figure_file":name,"case_id":case_id,"image_id":image_id,
                                          "source":image["path"],"sha256":hashlib.sha256(content).hexdigest(),
                                          "transform":"none"})
    plt.rcParams.update({"font.size":8.5,"font.family":"DejaVu Sans","pdf.fonttype":42,"axes.linewidth":.5})
    fig,ax=plt.subplots(figsize=(3.25,1.85))
    entries=[("base2","state","2B / Condition"),("base2","identity","2B / Identity"),
             ("base8","state","8B / Condition"),("base8","identity","8B / Identity")]
    for y,(model,task,label) in enumerate(entries):
        counts=analysis["paired"][model][task]["counts"]
        lost,corrected=counts.get("correct_wrong",0),counts.get("wrong_correct",0)
        ax.barh(y,-lost,height=.52,color="#AE5E40",zorder=3)
        ax.barh(y,corrected,height=.52,color="#47798D",zorder=3)
        ax.text(-lost-.8,y,str(lost),ha="right",va="center",fontsize=8)
        ax.text(corrected+.8,y,str(corrected),ha="left",va="center",fontsize=8)
    ax.set_yticks(range(4),[x[2] for x in entries])
    ax.set_xlim(-49,45);ax.set_ylim(3.65,-.95)
    ax.xaxis.set_major_locator(FixedLocator([-40,-20,0,20,40]))
    ax.xaxis.set_major_formatter(FixedFormatter(["40","20","0","20","40"]))
    ax.axvline(0,color="#555555",linewidth=.55,zorder=2)
    ax.grid(axis="x",color="#dddddd",linewidth=.45,zorder=0)
    ax.set_xlabel("Number of cases",labelpad=2)
    ax.text(-24,-.66,r"Correct $\rightarrow$ wrong",ha="center",fontsize=8,color="#763C27")
    ax.text(23,-.66,r"Wrong $\rightarrow$ correct",ha="center",fontsize=8,color="#335A6A")
    for spine in ["top","right","left"]:ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y",length=0,pad=5);ax.tick_params(axis="x",length=2)
    fig.subplots_adjust(left=.29,right=.99,top=.99,bottom=.22)
    buffer=io.BytesIO()
    fig.savefig(buffer,format="pdf",metadata={"Author":None,"CreationDate":None,"ModDate":None})
    plt.close(fig)
    assets["paper/figures/paired-answer-changes.pdf"]=buffer.getvalue()
    return assets


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--figure-directory",type=Path)
    args=parser.parse_args()
    analysis,cases=analyze(args.repository,args.runtime)
    if args.figure_directory:
        args.figure_directory.mkdir(parents=True,exist_ok=True)
        for name,content in figure_assets(args.runtime,analysis,cases).items():
            (args.figure_directory/Path(name).name).write_bytes(content)
    args.output.write_text(json.dumps(analysis,indent=2)+"\n")
