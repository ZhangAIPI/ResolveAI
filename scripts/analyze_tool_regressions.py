"""Audit frozen Sufficient direct/tool pairs from recorded SQLite; no inference."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

VALID = ("Supported", "Refuted", "Need more evidence")
ARMS = ("initial_text", "interactive_agent")
CASE_MODELS = {"qwen3_2b", "internvl35", "llava_onevision", "smolvlm2", "gpt41_mini", "qwen25", "gpt61_sol"}

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

@lru_cache(None)
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def text_parts(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return [content]
    return [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]

def inspect_trace(trace):
    user = next(m for m in trace["messages"] if m.get("role") == "user")
    refs = [p["image_ref"] for p in user.get("content", []) if isinstance(p, dict) and p.get("type") == "image"]
    claim = ""
    for text in text_parts(user):
        try:
            claim = json.loads(text).get("claim", "")
        except (ValueError, AttributeError):
            pass
        if claim:
            break
    calls, errors, finish_errors = [], Counter(), 0
    format_errors = 0
    for m in trace["messages"]:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls", []):
                calls.append(call.get("function", {}).get("name"))
        for text in text_parts(m):
            if m.get("role") == "user" and text.startswith("Invalid tool format."):
                format_errors += 1
            if m.get("role") != "tool":
                continue
            for line in text.splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict) and "error" in value:
                    errors[str(value["error"])] += 1
                    finish_errors += int(m.get("name") == "finish")
    return {"initial_image_refs": refs, "claim": claim, "action_sequence": calls,
            "tool_error_responses": dict(errors), "invalid_format_corrections": format_errors,
            "finish_attempts": calls.count("finish"), "rejected_finish_responses": finish_errors}

def read_arm(directory, arm, families):
    dbpath = directory / "episodes.sqlite"
    stat = (dbpath.stat().st_size, dbpath.stat().st_mtime_ns)
    ids = [f + "-Sufficient" for f in sorted(families)]
    sql = "SELECT result,trace FROM episodes WHERE arm=? AND case_id IN (" + ",".join("?" for _ in ids) + ")"
    with sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True) as db:
        raw = db.execute(sql, [arm, *ids]).fetchall()
    rows = {}
    for result, trace in raw:
        row = json.loads(result)
        assert row["variant"] == "Sufficient" and row["family_id"] in families
        assert row["family_id"] not in rows
        parsed = json.loads(trace)
        rows[row["family_id"]] = {"result": row, "facts": inspect_trace(parsed),
            "trace_sha256": hashlib.sha256(trace.encode()).hexdigest()}
    assert set(rows) == families
    provenance = {"database": dbpath, "database_bytes": stat[0], "database_sha256": file_sha(str(dbpath)),
        "manifest_sha256": file_sha(str(directory / "manifest.json")),
        "ordered_result_digest": digest([rows[f]["result"] for f in sorted(families)]),
        "ordered_trace_digest": digest([(f, rows[f]["trace_sha256"]) for f in sorted(families)])}
    assert stat == (dbpath.stat().st_size, dbpath.stat().st_mtime_ns), "Source database changed"
    return rows, provenance

def transition(a, b):
    before = a["verdict"] == a["source_label"]
    after = b["verdict"] == b["source_label"]
    if after:
        return "maintained_correct" if before else "corrected"
    if not before:
        return "remained_wrong"
    return ("lost_no_valid_final" if b["verdict"] not in VALID else
            "lost_explicit_NME" if b["verdict"] == VALID[2] else "lost_valid_wrong_label")

def analyze(repo, runtime, output):
    paths = ["docs/results/cvpr_20261102_candidate_scores.json", "docs/results/openai_candidate_scores.json", "docs/results/obtainable_results.json"]
    reference, closed, obtainable = [json.loads((repo / p).read_text()) for p in paths]
    families = set(reference["family_ids"])
    assert len(families) == 209 and set(closed["family_ids"]) == families
    references = {w["name"]: w for w in reference["workers"] + closed["workers"]}
    report = {"schema": "resolveai-paired-tool-regression-v2", "variant": "Sufficient",
        "scope": "Descriptive paired source-label agreement; not grounded correctness or a causal effect of tools.",
        "arms": {"initial_text": "One direct answer, initially supplied photographs, 96 output tokens.",
                 "interactive_agent": "Same initial photographs; tools and requests; eight turns, budget 12, 384 output tokens per call."},
        "selection": "Exactly the frozen 209 Sufficient families; no success-based filtering.",
        "dataset_sha256": reference["dataset_sha256"], "family_ids": sorted(families),
        "task_counts": reference["task_counts"], "analysis_script_sha256": file_sha(str(Path(__file__).resolve())),
        "source_reports": {p: file_sha(str(repo / p)) for p in paths},
        "definitions": {"corrected": "Direct non-match -> interactive source-label match",
            "lost_no_valid_final": "Direct match -> absent or invalid interactive verdict",
            "lost_explicit_NME": "Direct match -> valid Need more evidence",
            "lost_valid_wrong_label": "Direct match -> valid opposite definite label",
            "trace_flags": "Observed, potentially overlapping events; not exclusive causal attributions."}, "models": []}
    candidates = []
    for model in obtainable["models"]:
        name = model["name"]
        tool_dir = (runtime / model["manifests"][0]["path"]).parent
        direct_dir = runtime / "evaluation/openai/luna_decisions" if name == "luna_decisions_tools" else tool_dir
        a, pa = read_arm(direct_dir, ARMS[0], families)
        b, pb = read_arm(tool_dir, ARMS[1], families)
        for source in (pa, pb):
            source["database"] = str(source["database"].relative_to(runtime))
        buckets, groups, flags = {}, {}, Counter()
        selected_examples = {}
        for f in sorted(families):
            x, y = a[f]["result"], b[f]["result"]
            assert x["source_label"] == y["source_label"] and y["source_label"] in VALID[:2]
            refs0, refs1 = a[f]["facts"]["initial_image_refs"], b[f]["facts"]["initial_image_refs"]
            assert refs0 and refs0 == refs1, (name, f, "Different or absent initial image references")
            if direct_dir != tool_dir:
                assert all(file_sha(str(direct_dir / p)) == file_sha(str(tool_dir / p)) for p in refs0)
            kind = transition(x, y)
            buckets.setdefault(kind, []).append(f)
            if not kind.startswith("lost_"):
                continue
            facts = b[f]["facts"]
            for key, value in {"with_finish_attempt": facts["finish_attempts"] > 0,
                "with_rejected_finish": facts["rejected_finish_responses"] > 0,
                "with_format_correction": facts["invalid_format_corrections"] > 0,
                "with_tool_error_response": bool(facts["tool_error_responses"])}.items():
                flags[kind + ":" + key] += int(value)
            flags[kind + ":termination:" + str(y.get("termination"))] += 1
            for code, count in facts["tool_error_responses"].items():
                flags[kind + ":tool_error_responses:" + code] += count
            flags[kind + ":invalid_format_corrections"] += facts["invalid_format_corrections"]
            if name in CASE_MODELS:
                sample_key = kind + (":error_free" if not y.get("errors") else ":with_errors")
                selected_examples.setdefault(sample_key, f)
        for task in ("overall", "state", "identity"):
            fs = [f for f in sorted(families) if task == "overall" or b[f]["result"]["task"] == task]
            counts = Counter(transition(a[f]["result"], b[f]["result"]) for f in fs)
            for key in ("maintained_correct", "corrected", "remained_wrong", "lost_no_valid_final", "lost_explicit_NME", "lost_valid_wrong_label"):
                counts.setdefault(key, 0)
            lost = sum(counts[k] for k in counts if k.startswith("lost_"))
            counts.update(episodes=len(fs), direct_correct=counts["maintained_correct"] + lost,
                interactive_correct=counts["maintained_correct"] + counts["corrected"], lost=lost,
                net_correct_change=counts["corrected"] - lost)
            assert sum(counts[k] for k in ("maintained_correct", "corrected", "remained_wrong", "lost_no_valid_final", "lost_explicit_NME", "lost_valid_wrong_label")) == len(fs)
            assert counts["interactive_correct"] - counts["direct_correct"] == counts["net_correct_change"]
            groups[task] = dict(counts)
        direct_reference = references["luna_decisions"] if name == "luna_decisions_tools" else references[name]
        assert groups["overall"]["direct_correct"] == direct_reference["arms"][ARMS[0]]["label_matches"]
        assert groups["overall"]["interactive_correct"] == references[name]["arms"][ARMS[1]]["label_matches"]
        report["models"].append({"name": name, "label": model["label"], "by_task": groups,
            "initial_image_pairs_equal": 209,
            "ordered_initial_image_reference_digest": digest([(f, a[f]["facts"]["initial_image_refs"]) for f in sorted(families)]),
            "provenance": {ARMS[0]: pa, ARMS[1]: pb},
            "transition_case_ids": buckets, "lost_trace_observations": dict(flags)})
        for f in sorted(set(selected_examples.values())):
            x, y = a[f]["result"], b[f]["result"]
            candidates.append({"model": name, "family_id": f, "task": y["task"], "transition": transition(x, y),
                "source_label": x["source_label"], "direct_verdict": x["verdict"], "interactive_verdict": y["verdict"],
                "termination": y["termination"], "errors": y["errors"], "latency_s": y.get("latency_s"),
                "trace_sha256": b[f]["trace_sha256"], **b[f]["facts"]})
        print(name, groups["overall"], flush=True)
    assert len(report["models"]) == 16
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_tool_regressions.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "case_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    lines = ["| Model | Direct correct | Tools correct | Corrected | Lost: no final | Lost: NME | Lost: wrong label | Net |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in report["models"]:
        c = m["by_task"]["overall"]
        cells = [m["label"]] + [str(c[k]) for k in ("direct_correct", "interactive_correct", "corrected", "lost_no_valid_final", "lost_explicit_NME", "lost_valid_wrong_label", "net_correct_change")]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "paired_tool_regressions.md").write_text("\n".join(lines) + "\n")
    print("Audit complete:", output, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.repository.resolve(), args.runtime.resolve(), args.output.resolve())
