"""Compare candidate source labels on a common cohort; never claim sufficiency accuracy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sqlite3

ARMS = ("initial_text", "full_available_text", "interactive_agent")


def read_rows(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT result FROM episodes")]


def summarize(rows):
    n = len(rows)
    hits = sum(row["verdict"] == row["source_label"] for row in rows)
    return {
        "episodes": n,
        "label_matches": hits,
        "label_agreement": hits / n if n else None,
        "submitted": sum(row["verdict"] is not None for row in rows),
        "verdicts": dict(Counter(row["verdict"] or "No submitted verdict" for row in rows)),
    }


def score(experiment):
    plan = json.loads((experiment / "plan.json").read_text())
    models = json.loads((experiment / "models.json").read_text())
    payload = (Path(plan["data"]) / "cases.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    cases = json.loads(payload)
    candidates = {
        case["family_id"]: case for case in cases
        if case["variant"] == "Sufficient"
        and case["family_id"] in set(plan["family_ids"])
        and case.get("provenance_truth") in {"Supported", "Refuted"}
    }
    data, available, manifests = {}, {}, {}
    for model in models:
        name = model["name"]
        all_rows = read_rows(experiment / name / "episodes.sqlite")
        selected = [
            row for row in all_rows
            if row["variant"] == "Sufficient"
            and row["family_id"] in candidates and row["arm"] in ARMS
        ]
        indexed = {(row["family_id"], row["arm"]): row for row in selected}
        if len(indexed) != len(selected):
            raise ValueError("Duplicate family/arm rows")
        for (family, _), row in indexed.items():
            if row["source_label"] != candidates[family]["provenance_truth"]:
                raise ValueError("Stored label differs from candidate source label")
        data[name] = indexed
        available[name] = {family for family in candidates
                           if all((family, arm) in indexed for arm in ARMS)}
        manifests[name] = json.loads((experiment / name / "manifest.json").read_text())
    common = sorted(set.intersection(*available.values()))
    labels = Counter(candidates[family]["provenance_truth"] for family in common)
    report = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Literal agreement with unreviewed construction labels, not grounded accuracy",
        "selection": "Sufficient variant; known source label; all models completed all three arms",
        "variant": "Sufficient",
        "dataset_sha256": plan["dataset_sha256"],
        "eligible_families": len(candidates),
        "common_families": len(common),
        "family_ids": common,
        "task_counts": dict(Counter(candidates[family]["task"] for family in common)),
        "label_counts": dict(labels),
        "constant_majority_label_agreement": max(labels.values()) / len(common) if common else None,
        "unknown_identity_families_excluded": sum(
            case["variant"] == "Sufficient" and case["family_id"] in set(plan["family_ids"])
            and case.get("provenance_truth") not in {"Supported", "Refuted"} for case in cases
        ),
        "denominator": "All matched episodes; abstentions and missing decisions count as non-matches",
        "formal_grounded_accuracy": None,
        "workers": [],
    }
    for model in models:
        name = model["name"]
        report["workers"].append({
            "name": name, "model_id": model["model_id"],
            "revision": model["revision"],
            "source_commit": manifests[name]["source_commit"],
            "available_known_families": len(available[name]),
            "arms": {
                arm: {
                    **summarize([data[name][(family, arm)] for family in common]),
                    "by_task": {
                        task: summarize([data[name][(family, arm)] for family in common
                                         if candidates[family]["task"] == task])
                        for task in sorted(report["task_counts"])
                    },
                }
                for arm in ARMS
            },
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(score(args.experiment), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
