"""Compare a completed adapter and its base on the fixed paper cohort."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.score_candidate_labels import ARMS, read_rows, summarize
from scripts.paper_result_table import NAMES, comparison_header, comparison_values, comparison_cells


def acquisition_behavior(rows, cases, families):
    """Describe actions, without treating a returned photo as sufficient evidence."""
    variants = ("Sufficient", "Obtainable", "Missing", "Unavailable")
    selected = [row for row in rows if row["arm"] == "interactive_agent"
                and row["family_id"] in families and row["variant"] in variants]
    wanted = {(family, variant) for family in families for variant in variants}
    keys = [(row["family_id"], row["variant"]) for row in selected]
    if set(keys) != wanted or len(keys) != len(wanted):
        raise ValueError("Matched four-version agent cohort is incomplete or duplicated")

    def counts(group):
        new_photos = [set(row["released"]) - set(cases[row["case_id"]]["initial"]) for row in group]
        if any(images and not row["requests"] for row, images in zip(group, new_photos)):
            raise ValueError("A new photo appeared without an actual request")
        return {
            "episodes": len(group),
            "finished": sum(row["termination"] == "finished" for row in group),
            "episodes_with_request": sum(row["requests"] > 0 for row in group),
            "episodes_with_new_photo": sum(bool(images) for images in new_photos),
            "new_photos": sum(len(images) for images in new_photos),
            "requests": sum(row["requests"] for row in group),
            "abstained": sum(row["verdict"] == "Need more evidence" for row in group),
            "mean_tool_calls": sum(row["tool_calls"] for row in group) / len(group) if group else None,
        }

    return {
        "scope": "Observed behavior only; new-photo release does not establish sufficiency or useful acquisition",
        "by_variant": {variant: {
            **counts([row for row in selected if row["variant"] == variant]),
            "by_task": {task: counts([row for row in selected
                                     if row["variant"] == variant and row["task"] == task])
                        for task in sorted({row["task"] for row in selected})},
        } for variant in variants},
    }


def compare(cohort, base, adapted, training, plan_path=None):
    reference = json.loads(cohort.read_text())
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in (base, adapted)]
    model_id = manifests[0]["configuration"]["model_id"]
    source = next(w for w in reference["workers"] if w["model_id"] == model_id)
    for key in ("dataset_sha256", "budget", "max_turns", "decoding", "image_max_pixels", "max_context_tokens",
                "tool_adapter", "assistant_prefix", "grounding_manifest"):
        if manifests[0]["configuration"][key] != manifests[1]["configuration"][key]:
            raise ValueError("Unmatched evaluation configuration: " + key)
    if manifests[0]["configuration"]["dataset_sha256"] != reference["dataset_sha256"]:
        raise ValueError("Paper cohort belongs to a different dataset")
    plan = json.loads((plan_path or base.parent / "plan.json").read_text())
    case_bytes = (Path(plan["data"]) / "cases.json").read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != reference["dataset_sha256"]:
        raise ValueError("Frozen case pool changed")
    cases = {case["case_id"]: case for case in json.loads(case_bytes)}
    training_manifest = json.loads((training / "training_manifest.json").read_text())
    adapted_name = training_manifest["data"].get("adaptation_name", "Acquisition SFT")
    results, behaviors = [], []
    wanted = {(family, arm) for family in reference["family_ids"] for arm in ARMS}
    for name, directory in (("Base", base), (adapted_name, adapted)):
        all_rows = read_rows(directory / "episodes.sqlite")
        behaviors.append({"name": name, **acquisition_behavior(
            all_rows, cases, set(reference["family_ids"]))})
        rows = [r for r in all_rows
                if r["variant"] == "Sufficient" and (r["family_id"], r["arm"]) in wanted]
        indexed = {(r["family_id"], r["arm"]): r for r in rows}
        if set(indexed) != wanted or len(indexed) != len(rows):
            raise ValueError("The fixed paper cohort is incomplete or duplicated")
        results.append({"name": name, "arms": {
            arm: {**summarize([r for r in rows if r["arm"] == arm]),
                  "by_task": {task: summarize([r for r in rows if r["arm"] == arm and r["task"] == task])
                              for task in reference["task_counts"]}}
            for arm in ARMS}})
    for arm in ARMS:
        if results[0]["arms"][arm]["label_matches"] != source["arms"][arm]["label_matches"]:
            raise ValueError("Stored base results differ from the frozen paper snapshot")
    from scripts.score_efficiency import summarize_run
    efficiency = [summarize_run(directory, reference["family_ids"], reference["dataset_sha256"])
                  for directory in (base, adapted)]
    return {
        "efficiency": efficiency,
        "base_model_id": model_id,
        "scope": reference["scope"], "common_families": reference["common_families"],
        "task_counts": reference["task_counts"],
        "dataset_sha256": reference["dataset_sha256"], "family_ids": reference["family_ids"],
        "selection": reference["selection"], "formal_grounded_accuracy": None,
        "training": training_manifest,
        "checkpoint": json.loads((training / "complete.json").read_text()),
        "evaluation_manifests": manifests, "models": results, "acquisition_behavior": behaviors,
    }


def render(report):
    model_id = report.get("base_model_id", "Qwen/Qwen3-VL-8B-Instruct")
    lines = [
        r"\begin{table*}[t]",r"\centering\small",r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{minipage}{.88\textwidth}\centering",
        r"\caption{\textbf{Before and after supervised fine-tuning of "+NAMES[model_id]+r".}",
        r"The two column groups compare the same test cases without and with access to tools.",
        r"Accuracy (\%), time and output tokens use the definitions in Table~\ref{tab:benchmark}.",
        r"Access to tools does not imply that the model uses them; actual calls are reported in the supplement.}",
        r"\label{tab:sft}",*comparison_header("Training"),
    ]
    for index,row in enumerate(report["models"]):
        values = comparison_values(row["arms"],report["efficiency"][index]["arms"])
        label = "Before SFT" if index==0 else "After SFT"
        lines.append(label+" & "+" & ".join(comparison_cells(values))+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}",r"\end{minipage}",r"\end{table*}"])+"\n"



def initial_rl_checkpoint(report):
    """Validation may retain the SFT initialization despite later policy updates."""
    state = report["checkpoint"]["state"]
    return ("groups" in report["training"]["configuration"]
            and report["checkpoint"].get("evaluated_group", state.get("best_group")) == 0)


def adaptation_groups(reports):
    if not reports:
        raise ValueError("At least one complete adaptation comparison is required")
    groups = {}
    for report in reports:
        if any(report[key] != reports[0][key] for key in ("dataset_sha256","family_ids")):
            raise ValueError("Adaptation comparisons do not share the same cohort")
        group = groups.setdefault(report["base_model_id"],[])
        if group and (report["models"][0] != group[0]["models"][0]
                      or report["efficiency"][0] != group[0]["efficiency"][0]):
            raise ValueError("Adaptation comparisons use different base measurements")
        group.append(report)
    return groups


def core_stages(group):
    """Choose predetermined training stages, never the highest benchmark score."""
    selected = {}
    for report in group:
        training = report["training"]
        if "evaluation_checkpoint_policy" in training:
            continue  # Fixed-final diagnostics have their own supplementary table.
        config = training["configuration"]
        if "groups" in config:
            stage = "RL" if config.get("initialization","sft") == "base" else "SFT+RL"
        elif report["models"][1]["name"] == "Mixed-source SFT":
            stage = "SFT"
        else:
            continue
        if stage in selected:
            raise ValueError("More than one comparison for a core training stage")
        selected[stage] = report
    return selected


def render_adaptations(reports):
    """One model per row; separate panels compare accuracy and interaction cost."""
    from scripts.paper_result_table import ORDER
    groups = adaptation_groups(reports)
    stages = ("Base","SFT","RL","SFT+RL")
    headers = ["Base","SFT","RL",r"SFT$\to$RL"]
    lines = [
        r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
        r"\begin{minipage}{.68\textwidth}\centering",
        r"\caption{\textbf{Effect of policy training.}",
        r"SFT uses balanced identity and condition data; RL applies GRPO directly to the base; SFT$\to$RL continues the SFT adapter.",
        r"Checkpoints use held-out validation data. Accuracy and cost follow Table~\ref{tab:benchmark}; best accuracy is bold within each model and tool setting.",
        r"-- denotes an unfinished comparison. $^{*}$Validation retained the initial policy for that RL run.",
        r"Training-data and action-target ablations are reported separately in the supplement.}",
        r"\label{tab:sft}",r"\begin{tabular}{@{}l*{8}{r}@{}}",r"\toprule",
    ]
    for panel in ("accuracy","cost"):
        if panel == "accuracy":
            lines += [r"\multicolumn{9}{l}{\emph{(a) Overall accuracy (\%)}}\\",
                      r" & \multicolumn{4}{c}{No tools} & \multicolumn{4}{c}{With tools}\\"]
        else:
            lines += [r"\addlinespace[6pt]",
                      r"\multicolumn{9}{l}{\emph{(b) Interaction cost with tools}}\\",
                      r" & \multicolumn{4}{c}{Time per case (s)} & \multicolumn{4}{c}{Output tokens per case}\\"]
        lines += [r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
                  "Model & "+" & ".join(headers+headers)+r"\\",r"\midrule"]
        for model in sorted(groups,key=ORDER.index):
            group = groups[model]
            selected = core_stages(group)
            base = group[0]
            cells = []
            for measurement in (("initial_text","interactive_agent") if panel=="accuracy"
                                else ("mean_wall_s","mean_output_tokens")):
                for stage in stages:
                    report = base if stage=="Base" else selected.get(stage)
                    if report is None:
                        cells.append("--")
                        continue
                    index = 0 if stage=="Base" else 1
                    if panel=="accuracy":
                        value = report["models"][index]["arms"][measurement]["label_agreement"]*100
                        marker = r"$^{*}$" if stage!="Base" and initial_rl_checkpoint(report) else ""
                        available = [base["models"][0]["arms"][measurement]["label_agreement"]]
                        available += [r["models"][1]["arms"][measurement]["label_agreement"] for r in selected.values()]
                        number = f"{value:.1f}"
                        if abs(value - 100*max(available)) < 1e-8:
                            number = r"\textbf{" + number + "}"
                        cells.append(number+marker)
                    else:
                        value = report["efficiency"][index]["arms"]["interactive_agent"][measurement]
                        cells.append(f"{value:.1f}" if measurement=="mean_wall_s" else str(round(value)))
            lines.append(NAMES[model]+" & "+" & ".join(cells)+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}\end{minipage}\end{table*}"])+"\n"


def render_training_ablation(reports):
    """A compact paired ablation, rather than a catalogue of every training run."""
    by_name = {r["models"][1]["name"]:r for r in reports
               if r["base_model_id"]=="Qwen/Qwen3-VL-8B-Instruct"}
    variants = [
        ("Identity","Final decisions","Terminal-only SFT"),
        ("","Actions + decisions","Acquisition SFT"),
        ("Mixed-task","Final decisions","Mixed terminal-only SFT"),
        ("","Actions + decisions","Mixed-source SFT"),
    ]
    lines = [
        r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
        r"\begin{minipage}{.86\textwidth}\centering",
        r"\caption{\textbf{Training data and action supervision.} All rows use Qwen3-VL-8B.",
        r"Condition, Identity and Overall are accuracy (\%) with tools on initially sufficient cases.",
        r"Request (\%) is the fraction requesting another photograph in Obtainable.",
        r"Final-decision supervision retains the same public histories but omits acquisition-action targets.",
        r"The update limit is shared; target-token compute is not matched.}",
        r"\label{tab:training-ablation}",
        r"\begin{tabular}{@{}llrrrr@{}}\toprule",
        r"Training data & Supervised targets & Condition & Identity & Overall & Request (\%)\\\midrule",
    ]
    for data,targets,name in variants:
        if name not in by_name:
            continue
        report = by_name[name]
        scores = report["models"][1]["arms"]["interactive_agent"]
        behavior = report["acquisition_behavior"][1]["by_variant"]["Obtainable"]
        values = [100*scores["by_task"][task]["label_agreement"] for task in ("state","identity")]
        values += [100*scores["label_agreement"],100*behavior["episodes_with_request"]/behavior["episodes"]]
        if data=="Mixed-task":
            lines.append(r"\midrule")
        lines.append(data+" & "+targets+" & "+" & ".join(f"{v:.1f}" for v in values)+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}\end{minipage}\end{table*}"])+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    report = compare(args.cohort, args.base, args.adapted, args.training, args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sft_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "sft_results.tex").write_text(render(report))
