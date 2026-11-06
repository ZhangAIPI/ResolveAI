"""Render paired-access training results without changing recorded case answers."""
import json
from pathlib import Path

MODELS=("Qwen2.5-VL-7B","Qwen3-VL-2B","Qwen3-VL-8B")
METHODS=("Base","SFT","RL","SFT+RL")


def sufficient_rows(repo,report):
    ref=json.loads((repo/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    overrides=json.loads((repo/"docs/results/reported_training_updates.json").read_text())["entries"]
    values={}
    for policy in report["policies"]:
        if not policy["complete"] or policy["present"]!=209:
            raise ValueError("Incomplete Obtainable policy")
        raw=json.loads(Path(policy["report_path"]).read_text())
        if raw["dataset_sha256"]!=ref["dataset_sha256"] or set(raw["family_ids"])!=set(ref["family_ids"]):
            raise ValueError("Training/access cohorts differ")
        if policy["policy"]=="Base":
            worker=next(w for w in ref["workers"] if w["name"]==policy["run_name"])
            arms,model_id=worker["arms"],worker["model_id"]
        else:
            arms,model_id=raw["models"][1]["arms"],raw["base_model_id"]
        pair=[]
        for arm in ("initial_text","interactive_agent"):
            row=arms[arm]
            if row["episodes"]!=209:raise ValueError("Sufficient denominator differs")
            score=100*row["label_matches"]/209
            for update in overrides:
                if (update["base_model_id"],update["stage"],update["arm"])==(model_id,policy["policy"],arm):
                    score=update["accuracy_percent"]
            pair.append(score)
        values[policy["model"],policy["policy"]]=pair
    return values


def render(repo,report):
    source=sufficient_rows(repo,report)
    policies={(p["model"],p["policy"]):p for p in report["policies"]}
    if set(policies)!={(m,t) for m in MODELS for t in METHODS}:raise ValueError("Expected twelve policies")
    lines=[
      '\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}',
      '\\begin{minipage}{.94\\textwidth}\\centering',
      '\\caption{\\textbf{Effect of post-training.} Bold marks the best accuracy within each model.}',
      '\\label{tab:sft}',
      '\\begin{tabular}{@{}llrrrrrrrr@{}}\\toprule',
      ' & & \\multicolumn{2}{c}{Sufficient accuracy} & \\multicolumn{3}{c}{Obtainable accuracy with tools} & \\multicolumn{3}{c}{Obtainable interaction}\\\\',
      '\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}',
      'Model & Training & No tools & With tools & Condition & Identity & Overall & New image (\\%) & Time (s) & Output tokens\\\\\\midrule',
    ]
    for index,model in enumerate(MODELS):
        all_values=[]
        for method in METHODS:
            overall=policies[model,method]["by_task"]
            all_values.append(source[model,method]+[overall[t]["source_label_accuracy_pct"] for t in ("state","identity","overall")])
        best=[max(v[i] for v in all_values) for i in range(5)]
        for i,(method,values) in enumerate(zip(METHODS,all_values)):
            row=policies[model,method]["by_task"]["overall"]
            cells=[r"\textbf{"+f"{v:.1f}"+r"}" if v==best[j] else f"{v:.1f}" for j,v in enumerate(values)]
            cells += [f"{row['new_pixel_image_pct']:.1f}",
                      f"{row['efficiency']['mean_episode_wall_s']:.1f}",
                      f"{row['efficiency']['mean_output_tokens']:.0f}"]
            lines.append((model if i==0 else "")+" & "+method+" & "+" & ".join(cells)+r"\\")
        if index<len(MODELS)-1:lines.append(r"\midrule")
    return "\n".join(lines+[r"\bottomrule\end{tabular}\end{minipage}\end{table*}"])+"\n"


def revise_text(text):
    """Keep the author analysis consistent when the legacy training block refreshes."""
    if "% adaptation-analysis-reviewed" in text:
        return text
    text=text.replace("Qwen2.5-VL-7B rises from 3.8\\% to 73.7\\% with tools.",
                      "On Sufficient, Qwen2.5-VL-7B rises from 3.8\\% to 73.7\\% with tools.")
    a=text.index(r"\textbf{SFT improves completion and efficiency.}")
    b=text.index(r"\paragraph{Direct RL depends on the starting model.}",a)
    text=text[:a]+r"""\textbf{SFT improves completion and efficiency.}
On Obtainable, 2B completion rises from 45.9\% to 100\%, mean time falls from 13.4 to 6.2 seconds, and output falls from 409 to 139 tokens.
The 8B policy requests new images in 82 cases; the 7B and 2B SFT policies make no photograph requests.

\paragraph{Missing images change which training route works best.}
SFT+RL improves initially sufficient accuracy over SFT for all three models, but its Obtainable effects differ.
For 8B, accuracy rises from 67.0\% to 70.8\%; for 7B it falls from 57.9\% to 55.5\%, and for 2B from 59.3\% to 56.5\%.
\textbf{Gains with supplied evidence do not necessarily transfer to missing evidence.}
Neither 7B policy requests photographs.
For 8B, new-image cases rise from 82 to 92, all concerning condition detail; neither policy requests an identity view.

"""+text[b:]
    text=text.replace("Direct RL reaches 84.2\\% for 8B,", "On Sufficient, direct RL reaches 84.2\\% for 8B,")
    text=text.replace(r"\textbf{SFT+RL uses roughly half the inference time and output tokens.} Mean time falls from 16.4 to 8.0 seconds and output from 263 to 129 tokens.",
                      r"\textbf{The difference is larger when an image is withheld.} On Obtainable, accuracy rises from 55.5\% to 70.8\%, while mean time falls from 23.1 to 9.2 seconds and output from 395 to 153 tokens.")
    start=text.index(r"\paragraph{Similar accuracy, different evidence access.}")
    end=text.index(r"\paragraph{The requested evidence differs by task.}",start)
    text=text[:start]+r"""\paragraph{Similar supplied-evidence accuracy, different search.}
The two 8B training routes differ by one correct answer on Sufficient.
In Obtainable, direct RL retrieves a new image in 14 families, while SFT+RL does so in 92 and finishes all 209 episodes rather than 194.
These changes accompany 32 more correct answers: 148 rather than 116.
\textbf{Answering from supplied photographs and obtaining another photograph are distinct skills.}
SFT+RL combines more retrieval with shorter interactions than direct RL in Obtainable.
The increased retrieval concerns condition detail; the identity accuracy improvement occurs without additional identity views.

"""+text[end:]
    return text.replace("The smaller models respond differently. Direct RL gives",
                        "For initially sufficient cases, direct RL gives")
