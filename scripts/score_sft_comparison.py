"""Compare a completed adapter and its base on the fixed paper cohort."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.score_candidate_labels import ARMS, read_rows, summarize
from scripts.paper_result_table import task_scores, NAMES


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
    model_name = NAMES[model_id]
    costs = report.get("efficiency")
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Supervised fine-tuning (SFT) of " + model_name + r".}",
        r"Entries are correct answers (\%) when tools are available, defined as in Table~\ref{tab:benchmark}.",
        r"The test images, tools and decision limits are unchanged."
        + (r" Time (seconds) and output tokens are means per attempted case." if costs else "") + r"}",
        r"\label{tab:sft}",
        r"\begin{tabular}{@{}lrrrrr@{}}" if costs else r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        (r"Training & Condition & Identity & Overall & \shortstack{Time\\(s)} & \shortstack{Output\\tokens}\\"
         if costs else r"Training & Condition & Identity & Overall\\"), r"\midrule",
    ]
    for index, row in enumerate(report["models"]):
        label = "Before SFT" if index == 0 else "After SFT"
        values = task_scores(row["arms"])[1::2]
        cells = [f"{v:.1f}" for v in values]
        if costs:
            usage = costs[index]["arms"]["interactive_agent"]
            if usage["correct"] != row["arms"]["interactive_agent"]["label_matches"]:
                raise ValueError("SFT efficiency cohort differs")
            cells += [f"{usage['mean_wall_s']:.1f}", f"{usage['mean_output_tokens']:.0f}"]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


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
