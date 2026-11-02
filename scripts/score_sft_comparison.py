"""Compare a completed adapter and its base on the fixed paper cohort."""
import argparse
import json
from pathlib import Path
from scripts.score_candidate_labels import ARMS, read_rows, summarize


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
        results.append({"name": name, "arms": {arm: summarize([r for r in rows if r["arm"] == arm])
                                             for arm in ARMS}})
    for arm in ARMS:
        if results[0]["arms"][arm]["label_matches"] != source["arms"][arm]["label_matches"]:
            raise ValueError("Stored base results differ from the frozen paper snapshot")
    return {
        "scope": reference["scope"], "common_families": reference["common_families"],
        "dataset_sha256": reference["dataset_sha256"], "family_ids": reference["family_ids"],
        "selection": reference["selection"], "formal_grounded_accuracy": None,
        "training": json.loads((training / "training_manifest.json").read_text()),
        "checkpoint": json.loads((training / "complete.json").read_text()),
        "evaluation_manifests": manifests, "models": results,
    }


def render(report):
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\textbf{Source-supervised acquisition SFT.}",
        f'Qwen3-VL-8B before and after LoRA, evaluated on the same {report["common_families"]}',
        r"families and source-label agreement (\%) as Table~\ref{tab:benchmark}.",
        r"Train and validation use CO3D development sequences disjoint from the frozen evaluation.",
        r"Finish is agent submission rate; these scores do not establish reviewed evidence sufficiency.}",
        r"\label{tab:sft}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Training & Initial & Full & Agent & Finish\\",
        r"\midrule",
    ]
    for row in report["models"]:
        arms = row["arms"]
        scores = [100 * arms[a]["label_agreement"] for a in ARMS]
        scores.append(100 * arms["interactive_agent"]["submitted"] / report["common_families"])
        lines.append(row["name"] + " & " + " & ".join(f"{value:.1f}" for value in scores) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


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
