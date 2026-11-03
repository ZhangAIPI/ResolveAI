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
}
ORDER = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-2B-Instruct",
         "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
         "Qwen/Qwen3.5-9B", "OpenGVLab/InternVL3_5-8B-HF",
         "llava-hf/llava-onevision-qwen2-7b-ov-hf"]
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


def render(report):
    n = report["common_families"]
    if not n or report["formal_grounded_accuracy"] is not None:
        raise ValueError("Expected a source-label report")
    workers = sorted(report["workers"], key=lambda w: ORDER.index(w["model_id"]))
    for worker in workers:
        for arm in MODES:
            if worker["arms"][arm]["episodes"] != n:
                raise ValueError("Different model denominators")
            if sum(v["episodes"] for v in worker["arms"][arm]["by_task"].values()) != n:
                raise ValueError("Task counts do not partition the cohort")
    scores = [task_scores(worker["arms"]) for worker in workers]
    best = [max(column) for column in zip(*scores)]
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{minipage}{.88\textwidth}\centering",
        r"\caption{\textbf{Correct answers with and without tools (\%).}",
        r"A correct answer confirms a true claim or rejects a false one, according to the dataset label.",
        r"Images only answers from the initial photographs; With tools can inspect images or request photographs before answering.",
        r"Need more evidence and failure to submit a final verdict earn no credit.",
        r"The source-derived labels have not been independently reviewed for evidence sufficiency.}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{Object condition} & \multicolumn{2}{c}{Object identity} & \multicolumn{2}{c}{Overall}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Model & Images only & With tools & Images only & With tools & Images only & With tools\\",
        r"\midrule",
    ]
    for worker, values in zip(workers, scores):
        cells = [rf"\textbf{{{value:.1f}}}" if value == top else f"{value:.1f}"
                 for value, top in zip(values, best)]
        lines.append(NAMES[worker["model_id"]] + " & " + " & ".join(cells) + r"\\")
    priors = []
    for task in ("state", "identity", None):
        counts = report["task_label_counts"][task] if task else report["label_counts"]
        prior = 100 * counts.get("Refuted", 0) / sum(counts.values())
        priors.extend([prior, prior])
    lines += [
        r"\midrule", "Always reject the claim & " + " & ".join(f"{v:.1f}" for v in priors) + r"\\",
        r"\bottomrule", r"\end{tabular}", r"\end{minipage}", r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


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


def render_full_reference(report):
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\textbf{Additional full-access reference.}",
        r"Source-label agreement (\%) on the same fixed cohort as the main task comparison.",
        r"All images means all obtainable source records are supplied before the direct decision.}",
        r"\label{tab:full-reference}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Model & Initial & All images & With tools\\",
        r"\midrule",
    ]
    for worker in sorted(report["workers"], key=lambda w: ORDER.index(w["model_id"])):
        arms = worker["arms"]
        values = [100 * arms[a]["label_matches"] / arms[a]["episodes"]
                  for a in ("initial_text", "full_available_text", "interactive_agent")]
        lines.append(NAMES[worker["model_id"]] + " & " +
                     " & ".join(f"{v:.1f}" for v in values) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-reference", type=Path)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report))
    if args.full_reference:
        args.full_reference.write_text(render_full_reference(report))
    if args.responses:
        args.responses.write_text(render_agent_responses(report))
