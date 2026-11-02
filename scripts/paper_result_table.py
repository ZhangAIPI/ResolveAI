"""Generate the paper's first result table from a fixed, measured common cohort."""
import argparse
import json
from pathlib import Path

NAMES = {
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL3.5-8B",
}
ARMS = ("initial_text", "full_available_text", "interactive_agent")


def render(report):
    n = report["common_families"]
    if not n or report["formal_grounded_accuracy"] is not None:
        raise ValueError("Expected a nonempty, unreviewed source-label report")
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\textbf{Initial benchmark results on a fixed common cohort.}",
        f"Source-label agreement (\\%) on {n} candidate families: "
        f'{report["task_counts"]["state"]} condition and {report["task_counts"]["identity"]} identity.',
        r"One constructed \emph{Sufficient} version is used per family.",
        r"All attempts count; abstentions and missing decisions are non-matches.",
        r"Finish is the percentage submitting an agent verdict.",
        r"These unreviewed labels do not establish evidence sufficiency or grounded accuracy.}",
        r"\label{tab:benchmark}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Initial & Full & Agent & Finish\\",
        r"\midrule",
    ]
    for worker in sorted(report["workers"], key=lambda w: NAMES[w["model_id"]]):
        arms = worker["arms"]
        if any(arms[a]["episodes"] != n for a in ARMS):
            raise ValueError("Different denominators in paper table")
        scores = [100 * arms[a]["label_matches"] / n for a in ARMS]
        finish = 100 * arms["interactive_agent"]["submitted"] / n
        lines.append(NAMES[worker["model_id"]] + " & " +
                     " & ".join(f"{s:.1f}" for s in [*scores, finish]) + r"\\")
    lines += [
        r"\midrule",
        "Always Refuted & " + " & ".join(
            [f'{100 * report["label_counts"]["Refuted"] / n:.1f}'] * 3) + r" & ---\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(args.report.read_text())))
