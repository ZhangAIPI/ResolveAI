"""Admit actual independent human reviews and regrade frozen model/human traces."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from resolveai.environment import Environment
from resolveai.review import admit, reviewed_outcome, full_truth_consensus


class NoOCR:
    available = False


def grade(case, decision, released, *, citation_scoring_available=True):
    env = Environment(case, case["asset_root"], ocr_backend=NoOCR())
    if not set(released) <= env._evidence.keys():
        raise ValueError("Trace refers to evidence outside the reviewed case")
    env._released = set(released)
    verdict = decision.get("verdict") if decision else None
    result = {"submitted": verdict is not None, **reviewed_outcome(env, verdict)}
    result["unsupported_verdict"] = (
        verdict in {"Supported", "Refuted"}
        and result["current_evidence_verdict"] == "Need more evidence"
    )
    result["grounded_correct"] = (
        env.evaluate(decision)["grounded_correct"]
        if citation_scoring_available and decision and "citations" in decision
        else None
    )
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["actor"], row["condition"], row.get("ui_protocol", "model-frozen"))
        ].append(row)
    return [
        {
            "actor": actor,
            "condition": condition,
            "ui_protocol": ui_protocol,
            "reviewed_episodes": len(items),
            "submitted": sum(r["submitted"] for r in items),
            "reasonable_verdicts": sum(
                r["decision_matches_current_evidence"] for r in items
            ),
            "unsupported_verdicts": sum(r["unsupported_verdict"] for r in items),
            "grounded_correct": sum(r["grounded_correct"] is True for r in items)
            if any(r["grounded_correct"] is not None for r in items)
            else None,
            "citation_scoring_available": sum(
                r["grounded_correct"] is not None for r in items
            ),
            "unused_obtainable_evidence": sum(
                r["unused_obtainable_evidence"] for r in items
            ),
            "requests": sum(r.get("requests", 0) for r in items),
        }
        for (actor, condition, ui_protocol), items in sorted(groups.items())
    ]


def full_material_agreement(models, cases, consensus, dataset_hash):
    """Only direct all-material baselines with exactly the reviewed evidence pool."""
    if not consensus["consensus"]:
        return []
    by_case = {c["case_id"]: c for c in cases}
    agreed = {r["family_id"]: r for r in consensus["consensus"]}
    totals = defaultdict(lambda: {"episodes": 0, "submitted": 0, "matches": 0})
    for path in sorted(models.glob("*/episodes.sqlite")):
        metadata = json.loads((path.parent / "manifest.json").read_text())
        if metadata["dataset_sha256"] != dataset_hash:
            raise ValueError("Model used a different dataset")
        db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        for result, trace in db.execute(
            "SELECT result,trace FROM episodes WHERE arm='full_available_text'"
        ):
            row = json.loads(result)
            case = by_case[row["case_id"]]
            human = agreed.get(case["family_id"])
            if (
                human is None
                or sorted(e["id"] for e in case["evidence"] if e["available"])
                != human["full_pool_ids"]
            ):
                continue
            stats = totals[metadata["model_id"]]
            stats["episodes"] += 1
            verdict = (json.loads(trace).get("decision") or {}).get("verdict")
            stats["submitted"] += verdict is not None
            stats["matches"] += verdict == human["verdict"]
        db.close()
    return [
        {
            "model": model,
            **counts,
            "label_agreement": counts["matches"] / counts["episodes"],
        }
        for model, counts in sorted(totals.items())
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("models", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh analysis snapshot directory")
    plan = json.loads((args.study / "plan.json").read_text())
    db = sqlite3.connect(
        "file:" + str(args.study / "responses.sqlite") + "?mode=ro", uri=True
    )
    answers = [
        {"actor": actor, "index": index, **json.loads(payload)}
        for actor, index, payload in db.execute("SELECT actor,idx,payload FROM answers")
    ]
    db.close()
    all_votes = [r["vote"] for r in answers if r.get("vote")]
    votes = [v for v in all_votes if v["protocol"] == "visual-review-v1"]
    truth_votes = [
        v
        for v in all_votes
        if v["protocol"] in {"visual-truth-review-v2", "visual-truth-review-v3"}
    ]
    cases = json.loads((Path(plan["data"]) / "cases.json").read_text())
    consensus = full_truth_consensus(truth_votes, cases, plan["dataset_sha256"])
    truth_report = {
        **consensus,
        "model_full_material_label_agreement": full_material_agreement(
            args.models, cases, consensus, plan["dataset_sha256"]
        ),
        "scope_note": "Single-answer reviews support full-material label agreement only. No initial/pool sufficiency, unsupported-decision, acquisition or grounded scores are inferred.",
    }
    args.output.mkdir(parents=True)
    (args.output / "full_truth_reviews.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in truth_votes)
    )
    (args.output / "full_truth_consensus.json").write_text(
        json.dumps(truth_report, indent=2) + "\n"
    )
    if not votes:
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "waiting_for_independent_reviews",
                    "recorded_answers": len(answers),
                    "review_votes": len(truth_votes),
                    "full_material_review": truth_report,
                    "formal_scores": None,
                },
                indent=2,
            )
            + "\n"
        )
        return
    path = args.output / "reviews.jsonl"
    path.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in votes) + "\n")
    manifest = admit(Path(plan["data"]), path, args.output / "reviewed")
    reviewed = {
        c["case_id"]: c
        for c in json.loads((args.output / "reviewed/cases.json").read_text())
    }
    graded = []
    for row in answers:
        if row["task"]["mode"] != "search" or row["status"] != "finished":
            continue
        case = reviewed.get(row["task"]["case_id"])
        if not case:
            continue
        graded.append(
            {
                "actor": "human",
                "ui_protocol": row.get("ui_protocol", "human-ui-v2-limited"),
                "participant": row["actor"],
                "condition": row["task"]["condition"],
                "case_id": case["case_id"],
                "requests": row["requests"],
                **grade(
                    case,
                    row["decision"],
                    row["released"],
                    citation_scoring_available=row.get(
                        "citation_scoring_available", True
                    ),
                ),
            }
        )
    for path in sorted(args.models.glob("*/episodes.sqlite")):
        metadata = json.loads((path.parent / "manifest.json").read_text())
        if metadata["dataset_sha256"] != plan["dataset_sha256"]:
            raise ValueError("Model used a different dataset")
        db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        for result, trace in db.execute("SELECT result,trace FROM episodes"):
            row = json.loads(result)
            record = json.loads(trace)
            case = reviewed.get(row["case_id"])
            if not case:
                continue
            graded.append(
                {
                    "actor": metadata["model_id"],
                    "condition": row["arm"],
                    "case_id": row["case_id"],
                    "requests": row["requests"],
                    **grade(case, record.get("decision"), row["released"]),
                }
            )
        db.close()
    report = {
        "status": "reviewed_subset_only",
        "study_reviewed_families": len({c["family_id"] for c in reviewed.values()}),
        "study_target_families": len(plan["family_ids"]),
        "review_votes": len(all_votes),
        "full_material_review": truth_report,
        "search_timeouts": sum(
            r["status"] == "timeout" and r["task"]["mode"] == "search" for r in answers
        ),
        "review_timeouts": sum(
            r["status"] == "timeout" and r["task"]["mode"] == "review" for r in answers
        ),
        "metrics": summarize(graded),
        "models_matched_human_cases": summarize(
            [
                r
                for r in graded
                if r["actor"] != "human"
                and r["case_id"]
                in {
                    a["task"].get("case_id")
                    for a in answers
                    if a["task"]["mode"] == "search"
                }
            ]
        ),
        "scope": "20-person feasibility pilot; text-only models have no citation score",
        "comparison": "Human UI protocols are reported separately. Humans are unrestricted; model tools and budgets remain frozen. Do not directly compare their latencies",
    }
    (args.output / "graded.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in graded) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
