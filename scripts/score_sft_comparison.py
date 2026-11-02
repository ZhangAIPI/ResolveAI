"""Compare a completed adapter and its base on the fixed paper cohort."""
import argparse
import json
from pathlib import Path
from scripts.score_candidate_labels import ARMS, read_rows, summarize
from scripts.paper_result_table import task_scores, outcomes


def compare(cohort, base, adapted, training):
    reference = json.loads(cohort.read_text())
    source = next(w for w in reference["workers"] if w["model_id"] == "Qwen/Qwen3-VL-8B-Instruct")
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in (base, adapted)]
    for key in ("dataset_sha256", "budget", "max_turns", "decoding", "image_max_pixels", "max_context_tokens",
                "tool_adapter", "assistant_prefix", "grounding_manifest"):
        if manifests[0]["configuration"][key] != manifests[1]["configuration"][key]:
            raise ValueError("Unmatched evaluation configuration: " + key)
    if manifests[0]["configuration"]["dataset_sha256"] != reference["dataset_sha256"]:
        raise ValueError("Paper cohort belongs to a different dataset")
    results = []
    wanted = {(family, arm) for family in reference["family_ids"] for arm in ARMS}
    for name, directory in (("Base", base), ("Acquisition SFT", adapted)):
        rows = [r for r in read_rows(directory / "episodes.sqlite")
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
    return {
        "scope": reference["scope"], "common_families": reference["common_families"],
        "task_counts": reference["task_counts"],
        "dataset_sha256": reference["dataset_sha256"], "family_ids": reference["family_ids"],
        "selection": reference["selection"], "formal_grounded_accuracy": None,
        "training": json.loads((training / "training_manifest.json").read_text()),
        "checkpoint": json.loads((training / "complete.json").read_text()),
        "evaluation_manifests": manifests, "models": results,
    }


def render(report):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{minipage}{.88\textwidth}\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{Verification before and after source-supervised acquisition SFT.}",
        f"Source-label agreement (\\%) uses the same tasks, settings and {report['common_families']}-family cohort as Table~\\ref{{tab:benchmark}}.",
        r"Training and validation source sequences are disjoint from the frozen evaluation pool.",
        r"The labels remain unreviewed for evidence sufficiency.}",
        r"\label{tab:sft}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        " & " + " & ".join([
            rf"\multicolumn{{2}}{{c}}{{Object condition ({report['task_counts']['state']} cases)}}",
            rf"\multicolumn{{2}}{{c}}{{Cross-view identity ({report['task_counts']['identity']} cases)}}",
            rf"\multicolumn{{2}}{{c}}{{Overall ({report['common_families']} cases)}}"]) + r"\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Qwen3-VL-8B & Answer only & With tools & Answer only & With tools & Answer only & With tools\\",
        r"\midrule",
    ]
    for row in report["models"]:
        lines.append(row["name"] + " & " + " & ".join(f"{v:.1f}" for v in task_scores(row["arms"])) + r"\\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule", r"Agent outcomes & Correct & Wrong & Need more evidence & Did not finish\\",
        r"\midrule",
    ]
    for row in report["models"]:
        lines.append(row["name"] + " & " + " & ".join(map(str, outcomes(row["arms"]["interactive_agent"]))) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{minipage}", r"\end{table*}"]) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare(args.cohort, args.base, args.adapted, args.training)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sft_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "sft_results.tex").write_text(render(report))
