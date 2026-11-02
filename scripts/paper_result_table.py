"""Render task-grouped benchmark results and agent outcomes from one fixed cohort."""
import argparse
import json
from pathlib import Path

NAMES = {
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL3.5-8B",
}
ORDER = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
         "Qwen/Qwen3.5-9B", "OpenGVLab/InternVL3_5-8B-HF"]
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
        raise ValueError("Expected an unreviewed source-label report")
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
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{minipage}{.88\textwidth}\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{Object verification across tasks and interaction outcomes.}",
        r"(a) Source-label agreement (\%, higher is better), broken down by task.",
        r"(b) The same agent episodes split into four mutually exclusive outcomes (counts).",
        f"All models use the same {n} candidate families, with one constructed Sufficient version per family.",
        r"Source labels and initial sufficiency have not been independently reviewed; these are not grounded-accuracy scores.}",
        r"\label{tab:benchmark}",
        r"\textbf{(a) Verification scores by task (\%)}\par\smallskip",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        " & " + " & ".join([
            rf"\multicolumn{{2}}{{c}}{{Object condition ({report['task_counts']['state']} cases)}}",
            rf"\multicolumn{{2}}{{c}}{{Cross-view identity ({report['task_counts']['identity']} cases)}}",
            rf"\multicolumn{{2}}{{c}}{{Overall ({n} cases)}}"]) + r"\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Model & Answer only & With tools & Answer only & With tools & Answer only & With tools\\",
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
        r"\midrule",
        "Always Refuted & " + " & ".join(f"{v:.1f}" for v in priors) + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{5pt}",
        rf"\textbf{{(b) Agent outcomes ({n} attempts per model)}}\par\smallskip",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Correct & Wrong & Need more evidence & Did not finish\\",
        r"\midrule",
    ]
    for worker in workers:
        values = outcomes(worker["arms"]["interactive_agent"])
        lines.append(NAMES[worker["model_id"]] + " & " + " & ".join(map(str, values)) + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Tasks:} condition asks whether the photographed item has the stated defect;",
        r"identity asks whether the foreground objects in two submissions are the same physical item.",
        r"\textit{Settings:} Answer only judges the initial images directly; With tools starts with the same images",
        r"and permits inspection and photo requests before a terminal decision.",
        r"Overall pools all cases; abstentions and missing verdicts score zero.",
        r"In (b), Correct/Wrong compare definite verdicts with the source label;",
        r"Need more evidence is a terminal abstention, while Did not finish means no terminal decision was submitted.",
        r"\end{minipage}",
        r"\end{minipage}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


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
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report))
    if args.full_reference:
        args.full_reference.write_text(render_full_reference(report))
