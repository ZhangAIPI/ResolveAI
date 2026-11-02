"""Read-only progress for a model matrix; no provisional labels become accuracy."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def summarize_worker(directory, expected):
    path = directory / "episodes.sqlite"
    rows = []
    if path.exists():
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = [
                json.loads(row[0]) for row in db.execute("SELECT result FROM episodes")
            ]
    by_arm = {}
    for arm in ("initial_text", "full_available_text", "interactive_agent"):
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "episodes": len(selected),
            "verdicts": dict(
                Counter(
                    row.get("verdict") or "No submitted verdict" for row in selected
                )
            ),
            "terminations": dict(Counter(row["termination"] for row in selected)),
            "errors": sum(row["errors"] for row in selected),
            "photo_requests": sum(row["requests"] for row in selected),
            "tool_calls": sum(row.get("tool_calls", 0) for row in selected),
            "wall_latency_s": round(
                sum(row.get("latency_s", 0) for row in selected), 1
            ),
        }
    return {
        "completed": len(rows),
        "expected": expected,
        "complete": len(rows) == expected,
        "arms": by_arm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.experiment / "plan.json").read_text())
    jobs = {
        j["name"]: j for j in json.loads((args.experiment / "jobs.json").read_text())
    }
    models = json.loads((args.experiment / "models.json").read_text())
    report = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "source": json.loads((args.experiment / "source.json").read_text()),
        "families": len(plan["family_ids"]),
        "dataset_sha256": plan["dataset_sha256"],
        "scope": "Behavior and interface diagnosis; independently reviewed sufficiency scores pending",
        "formal_scores": None,
        "workers": [],
    }
    for model in models:
        progress = summarize_worker(
            args.experiment / model["name"], plan["expected_episodes_per_model"]
        )
        manifest_path = args.experiment / model["name"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        report["workers"].append(
            {
                "source_commit": manifest.get("source_commit", model.get("source_commit", report["source"]["source_commit"])),
                "versions": manifest.get("versions"),
                "name": model["name"],
                "model_id": model["model_id"],
                "revision": model["revision"],
                "job_id": jobs[model["name"]]["job_id"],
                **progress,
            }
        )
    report["completed"] = sum(w["completed"] for w in report["workers"])
    report["expected"] = sum(w["expected"] for w in report["workers"])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
