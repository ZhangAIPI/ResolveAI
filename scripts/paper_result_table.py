"""Render task-grouped benchmark results and agent outcomes from one fixed cohort."""
import argparse
import json
from pathlib import Path

NAMES = {
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL3.5-8B",
    "OpenGVLab/InternVL3_5-14B-HF": "InternVL3.5-14B",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "SmolVLM2-2.2B",
    "mistral-experimental/pixtral-12b": "Pixtral-12B",
}
ORDER = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-2B-Instruct",
         "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
         "Qwen/Qwen3.5-9B", "OpenGVLab/InternVL3_5-8B-HF",
         "OpenGVLab/InternVL3_5-14B-HF",
         "llava-hf/llava-onevision-qwen2-7b-ov-hf",
         "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "mistral-experimental/pixtral-12b"]
MODES = ("initial_text", "interactive_agent")


def task_scores(arms):
    result = []
    for task in ("state", "identity", None):
        for arm in MODES:
            row = arms[arm]["by_task"][task] if task else arms[arm]
            result.append(100 * row["label_matches"] / row["episodes"])
    return result


def outcomes(row):
    abstained = row["verdicts"].get("Need more evidence", 0)
    missing = row["episodes"] - row["submitted"]
    wrong = row["submitted"] - row["label_matches"] - abstained
    result = [row["label_matches"], wrong, abstained, missing]
    if min(result) < 0 or sum(result) != row["episodes"]:
        raise ValueError("Agent outcomes do not partition the fixed cohort")
    return result


def accuracy_rows(report, sft=None):
    rows = []
    if sft is not None and (sft["dataset_sha256"] != report["dataset_sha256"]
                            or sft["family_ids"] != report["family_ids"]):
        raise ValueError("SFT and base cohorts differ")
    for worker in sorted(report["workers"], key=lambda w: ORDER.index(w["model_id"])):
        rows.append({"label":NAMES[worker["model_id"]], "model_id":worker["model_id"],
                     "arms":worker["arms"], "adapted":False})
        if sft is not None and worker["model_id"] == sft["base_model_id"]:
            for arm in MODES:
                if worker["arms"][arm]["label_matches"] != sft["models"][0]["arms"][arm]["label_matches"]:
                    raise ValueError("SFT base differs from the main table")
            method = sft["models"][1]["name"]
            method = {"Acquisition SFT":"identity SFT"}.get(method, method)
            rows.append({"label":r"\quad + " + method, "model_id":worker["model_id"],
                         "arms":sft["models"][1]["arms"], "adapted":True})
    if sft is not None and not any(row["adapted"] for row in rows):
        raise ValueError("SFT base is absent from the table")
    return rows


def cost_rows(report, efficiency, sft=None):
    if efficiency["dataset_sha256"] != report["dataset_sha256"] or efficiency["family_ids"] != report["family_ids"]:
        raise ValueError("Efficiency and accuracy cohorts differ")
    costs = {row["model_id"]:row for row in efficiency["workers"]}
    if set(costs) != {w["model_id"] for w in report["workers"]}:
        raise ValueError("Efficiency model list differs")
    rows = accuracy_rows(report, sft)
    for row in rows:
        cost = sft["efficiency"][1] if row["adapted"] else costs[row["model_id"]]
        if cost["gpu"] != "NVIDIA RTX A6000":
            raise ValueError("Update the hardware caption for this run")
        row["cost"] = cost["arms"]["interactive_agent"]
        score = row["arms"]["interactive_agent"]
        if row["cost"]["correct"] != score["label_matches"] or row["cost"]["episodes"] != score["episodes"]:
            raise ValueError("Efficiency accuracy mismatch")
        if score["episodes"] != report["common_families"]:
            raise ValueError("Different model denominators")
    return rows


def comparison_header(label):
    return [
        r"\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{5}{c}{No tools} & \multicolumn{5}{c}{With tools}\\",
        r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
        label+r" & Condition & Identity & Overall & \shortstack{Time\\(s)} & \shortstack{Output\\tokens} & Condition & Identity & Overall & \shortstack{Time\\(s)} & \shortstack{Output\\tokens}\\",
        r"\midrule",
    ]


def comparison_values(arms, costs):
    values = []
    for arm in MODES:
        score, usage = arms[arm], costs[arm]
        if usage["correct"] != score["label_matches"] or usage["episodes"] != score["episodes"]:
            raise ValueError("Setting-specific cost differs from accuracy")
        values += [100*x["label_matches"]/x["episodes"] for x in
                   (score["by_task"]["state"],score["by_task"]["identity"],score)]
        values += [usage["mean_wall_s"], usage["mean_output_tokens"]]
    return values


def comparison_cells(values, best=None):
    cells = []
    for index,value in enumerate(values):
        text = "--" if value is None else (f"{value:.0f}" if index%5==4 else f"{value:.1f}")
        if best is not None and value is not None and index in (0,1,2,5,6,7) and value==best[index]:
            text = r"\textbf{"+text+"}"
        cells.append(text)
    return cells


def display_api_workers(workers):
    """Combine the unchanged direct baseline with its separately audited menu run."""
    menu=next((w for w in workers if w["name"]=="luna_decisions_tools"),None)
    if menu is None:
        return list(workers)
    baseline=next((w for w in workers if w["name"]=="luna_decisions"),None)
    if (baseline is None or menu["model_id"]!=baseline["model_id"]
            or menu["endpoint"]!="decisions"
            or menu["api_config"].get("decision_tools")!="public-menu-v1"
            or set(menu["arms"])!={"interactive_agent"}):
        raise ValueError("Decisions menu run does not match its direct baseline")
    merged={**baseline,
            "arms":{**baseline["arms"],**menu["arms"]},
            "costs":{**baseline["costs"],**menu["costs"]},
            "action_adapter":"public-menu-v1","tool_source":menu["name"]}
    return [merged if w["name"]=="luna_decisions" else w
            for w in workers if w["name"]!="luna_decisions_tools"]


def render(report, efficiency, closed=None):
    cost_rows(report, efficiency)
    costs = {row["model_id"]:row for row in efficiency["workers"]}
    workers = sorted(report["workers"],key=lambda w:ORDER.index(w["model_id"]))
    values = [comparison_values(w["arms"],costs[w["model_id"]]["arms"]) for w in workers]
    best = [max(column) for column in zip(*values)]
    api = display_api_workers(closed["workers"]) if closed else []
    has_menu = any(w.get("action_adapter")=="public-menu-v1" for w in api)
    if api and (closed["dataset_sha256"]!=report["dataset_sha256"] or closed["family_ids"]!=report["family_ids"]):
        raise ValueError("API and local-model cohorts differ")
    api_labels = {"terra-eval":"GPT-6.1 Terra", "gpt61_sol":"GPT-6.1 Sol",
                  "gpt56_sol":"GPT-5.6 Sol", "gpt41":"GPT-4.1", "gpt41_mini":"GPT-4.1 mini",
                  "gpt54_mini":"GPT-5.4 mini", "luna":"GPT-6 Luna",
                  "luna_decisions":"GPT-6 Luna (Decisions)"}
    api_order = ["gpt61_sol","terra-eval","gpt56_sol","gpt54_mini","gpt41","gpt41_mini","luna","luna_decisions"]
    api = sorted(api,key=lambda w:api_order.index(w["name"]))
    api_values = []
    for worker in api:
        values_row = []
        for arm in MODES:
            if arm not in worker["arms"]:
                values_row += [None]*5
                continue
            scores,cost = worker["arms"][arm],worker["costs"][arm]
            if scores["episodes"]!=report["common_families"] or cost["correct"]!=scores["label_matches"]:
                raise ValueError("API accuracy/cost denominator changed")
            values_row += [100*s["label_matches"]/s["episodes"] for s in
                           (scores["by_task"]["state"],scores["by_task"]["identity"],scores)]
            values_row += [cost["mean_wall_s"],None if worker["endpoint"]=="decisions" else cost["mean_output_tokens"]]
        api_values.append(values_row)
    api_best = [max((row[i] for row in api_values if row[i] is not None),default=None) for i in range(10)]
    lines = [
        r"\begin{table*}[t]",r"\centering\small",r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{minipage}{.96\textwidth}\centering",
        (r"\caption{\textbf{Open-weight and API-served models with and without tools.}" if api else r"\caption{\textbf{Open-source models with and without tools.}"),
        r"No tools answers directly from the initial photographs; With tools starts from the same photographs and permits interaction.",
        r"Condition, Identity and Overall report correct answers (\%) against unreviewed source labels; abstentions and missing decisions receive zero.",
        (r"Time and output tokens are means over all attempts. Local models run on RTX A6000; API time includes network round trips and CPU tools." if api else r"Time and output tokens are means over all attempts on RTX A6000 GPUs."),
        (r"Output includes API reasoning tokens; inputs are excluded. $^{\dagger}$Decisions selects supplied action parameters and citation regions; its generated-text counts are inapplicable." if has_menu else (r"Output includes API reasoning tokens where present; input images and prompts are excluded. Decisions returns fixed choices; unmeasured tool results and generated-text counts are marked --." if api else r"Output tokens count generated text and end markers, excluding input images and prompts.")),
        (r"Best accuracy is bold within each group; -- marks inapplicable or unmeasured entries. API settings are detailed in the supplement.}" if api else r"Best accuracy results are bold; unmeasured baseline costs are marked --.}"),
        r"\label{tab:benchmark}",*comparison_header("Model"),
    ]
    if api:
        lines.append(r"\multicolumn{11}{l}{\emph{Open weights}}\\")
    for worker,row in zip(workers,values):
        lines.append(NAMES[worker["model_id"]]+" & "+" & ".join(comparison_cells(row,best))+r"\\")
    if api:
        lines += [r"\midrule",r"\multicolumn{11}{l}{\emph{API-served models}}\\"]
        for worker,row in zip(api,api_values):
            label=api_labels[worker["name"]]+(r"$^{\dagger}$" if worker.get("action_adapter") else "")
            lines.append(label+" & "+" & ".join(comparison_cells(row,api_best))+r"\\")
    prior = []
    for task in ("state","identity",None):
        counts = report["task_label_counts"][task] if task else report["label_counts"]
        prior.append(100*counts.get("Refuted",0)/sum(counts.values()))
    prior_cells = [f"{v:.1f}" for v in prior]+["--","--"]
    lines += [r"\midrule","Always reject the claim & "+" & ".join(prior_cells*2)+r"\\",
              r"\bottomrule",r"\end{tabular}",r"\end{minipage}",r"\end{table*}"]
    return "\n".join(lines)+"\n"


def render_efficiency_reference(report, efficiency, sft=None):
    rows = cost_rows(report, efficiency, sft)
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{\textbf{Completion and latency diagnostics.}",
        r"The same attempts as the main table. Finished includes valid uncertain decisions.",
        r"Model calls count actual generations. Median and 90th-percentile times include unfinished episodes.}",
        r"\label{tab:efficiency-reference}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"Model & Finished (\%) & Input tokens (k) & Model calls & Tool calls & Median (s) & 90th percentile (s)\\", r"\midrule",
    ]
    for row in rows:
        cost = row["cost"]
        cells = [f"{cost['finished_pct']:.1f}", f"{cost['mean_input_tokens']/1000:.1f}",
                 f"{cost['mean_model_calls']:.2f}", f"{cost['mean_tool_calls']:.2f}",
                 f"{cost['p50_wall_s']:.1f}", f"{cost['p90_wall_s']:.1f}"]
        lines.append(row["label"]+" & "+" & ".join(cells)+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}",r"\end{table*}"])+"\n"


def render_agent_responses(report):
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\begin{minipage}{.88\textwidth}\centering",
        r"\caption{\textbf{Final responses when tools are available (\%).}",
        r"Each row sums to 100\%, up to rounding. Correct and Incorrect are definite verdicts compared with the dataset label.",
        r"Uncertain means the model submitted Need more evidence.",
        r"No answer means no valid final verdict was submitted within the turn and context limits.",
        r"This breakdown explains the missing credit in the main accuracy table.}",
        r"\label{tab:agent-responses}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Model & Correct & Incorrect & Uncertain & No answer\\", r"\midrule",
    ]
    for worker in sorted(report["workers"], key=lambda w: ORDER.index(w["model_id"])):
        row = worker["arms"]["interactive_agent"]
        values = [100 * value / row["episodes"] for value in outcomes(row)]
        lines.append(NAMES[worker["model_id"]] + " & " +
                     " & ".join(f"{value:.1f}" for value in values) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}",
                               r"\end{minipage}", r"\end{table*}"]) + "\n"


def render_full_reference(report, sft=None):
    lines = [
        r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{minipage}{.96\textwidth}\centering",
        r"\caption{\textbf{Direct and interactive verification by task (\%).}",
        r"Initial uses the initially supplied photographs; All images supplies every obtainable source record before a direct verdict; Tools permits interaction.",
        r"All rows use the same fixed cohort and source-label metric.}",
        r"\label{tab:full-reference}", r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{Object condition} & \multicolumn{3}{c}{Object identity} & \multicolumn{3}{c}{Overall}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"Model & Initial & All images & Tools & Initial & All images & Tools & Initial & All images & Tools\\",
        r"\midrule",
    ]
    for row in accuracy_rows(report, sft):
        values = []
        for task in ("state","identity",None):
            for arm in ("initial_text","full_available_text","interactive_agent"):
                score = row["arms"][arm]["by_task"][task] if task else row["arms"][arm]
                values.append(100*score["label_matches"]/score["episodes"])
        lines.append(row["label"]+" & "+" & ".join(f"{v:.1f}" for v in values)+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}",r"\end{minipage}",r"\end{table*}"])+"\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-reference", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--closed",type=Path,default=Path("docs/results/openai_candidate_scores.json"))
    parser.add_argument("--efficiency-reference", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    efficiency = json.loads(args.efficiency.read_text())
    sft = json.loads(args.sft.read_text()) if args.sft else None
    args.output.write_text(render(report, efficiency, json.loads(args.closed.read_text()) if args.closed.exists() else None))
    if args.full_reference:
        args.full_reference.write_text(render_full_reference(report, sft))
    if args.responses:
        args.responses.write_text(render_agent_responses(report))
    if args.efficiency_reference:
        args.efficiency_reference.write_text(render_efficiency_reference(report, efficiency, sft))
