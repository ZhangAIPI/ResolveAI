"""Summarize only the complete 68-by-3 selected-history content probe."""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np


def collect(repo, runtime, root):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from scripts import evaluate_photo_content_probe as probe
    from scripts.summarize_evidence_control_pairs import paired
    selected, source = probe.sources(repo, runtime)
    families = sorted(selected)
    expected = {(f, c) for f in families for c in probe.CONDITIONS}
    indexed, inputs, common = {}, {}, None
    audit_fields = ("nonpixel_tensors", "pixel_values_shape", "pixel_values_dtype",
                    "pixel_values_sha256", "pixel_blocks_sha256", "image_grid_thw",
                    "input_tokens", "source_native_input_tokens", "replacement_message_and_part")
    for shard in range(4):
        directory = root / f"shard-{shard}"
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists() or not (directory / "episodes.sqlite").exists():
            raise ValueError("Probe is incomplete")
        manifest = json.loads(manifest_path.read_text())
        cfg = manifest["configuration"]
        if (cfg["families"] != families[shard::4] or cfg["shard"] != shard or cfg["shards"] != 4
                or cfg["script_sha256"] != probe.file_sha(repo / "scripts/evaluate_photo_content_probe.py")
                or cfg["source"] != source or cfg["conditions"] != list(probe.CONDITIONS)
                or (cfg["max_new_tokens"], cfg["max_context_tokens"], cfg["continuation_calls"],
                    cfg["do_sample"]) != (384, 8192, 1, False)):
            raise ValueError("Shard source, cohort or configuration differs")
        invariant = {k: v for k, v in cfg.items() if k not in ("families", "shard")}
        if common is not None and invariant != common:
            raise ValueError("Shard configuration mismatch")
        common = invariant
        source_worker = source["source_worker"]
        if any(manifest.get("worker", {}).get(k) != v for k, v in source_worker.items()):
            raise ValueError("Frozen visual worker configuration differs")
        recorded = hashlib.sha256()
        actual = set()
        with closing(sqlite3.connect(f"file:{directory}/episodes.sqlite?mode=ro", uri=True)) as db:
            for family, condition, raw, trace_raw in db.execute("SELECT * FROM episodes ORDER BY family_id,condition"):
                key = family, condition
                if key in indexed or key not in expected or family not in cfg["families"]:
                    raise ValueError("Unexpected or overlapping result")
                row, payload = json.loads(raw), json.loads(trace_raw)
                if (row["family_id"], row["condition"]) != key:
                    raise ValueError("Recorded result key differs")
                if (row["source_record_sha256"] != selected[family]["record_sha256"]
                        or row["family_seed"] != probe.family_seed(family, cfg["seed"])
                        or row["source_label"] != selected[family]["result"]["source_label"]
                        or row["source_label_match"] != (row["schema_valid_finish"] and row["verdict"] == row["source_label"])):
                    raise ValueError("Source labels or records differ")
                for _, _, part in probe.image_parts(payload["messages"]):
                    part["image_ref"] = str((directory / part["image_ref"]).resolve())
                if (probe.digest(probe.canonical(probe.transport(payload)).encode()) != row["input_sha256"]
                        or probe.nonpixel_hash(payload) != row["nonpixel_public_sha256"]
                        or row["generation"]["input_tokens"] != row["input_tokens"]
                        or row["generation"].get("output_tokens", 0) > 384):
                    raise ValueError("Actual input hash or token accounting differs")
                original, target = probe.selected_prefix(selected[family]["trace"], selected[family]["directory"])
                original["messages"][0]["content"] += probe.INSTRUCTION
                original.update(mode="tools", max_new_tokens=384, tool_adapter="native", assistant_prefix=True)
                if probe.nonpixel_hash(payload) != probe.nonpixel_hash(original):
                    raise ValueError("Nonpixel source prefix changed")
                slots = list(probe.image_parts(payload["messages"]))
                original_slots = list(probe.image_parts(original["messages"]))
                if [(a, b) for a, b, _ in slots] != [(a, b) for a, b, _ in original_slots]:
                    raise ValueError("Image ordering changed")
                for (mi, pi, part), (_, _, old) in zip(slots, original_slots):
                    pixels = probe.rgb(part["image_ref"])
                    old_pixels = probe.rgb(old["image_ref"])
                    intended = (probe.transformed(old_pixels, condition, row["family_seed"])
                                if (mi, pi) == target else old_pixels)
                    if not np.array_equal(pixels, intended):
                        raise ValueError("Changed non-target pixels or wrong intervention")
                    if (mi, pi) == target and probe.rgb_sha(pixels) != row["returned_rgb_sha256"]:
                        raise ValueError("Returned pixel hash differs")
                indexed[key] = row
                actual.add(key)
                recorded.update(probe.canonical([family, condition, raw, trace_raw]).encode())
        if actual != {(f, c) for f in cfg["families"] for c in probe.CONDITIONS}:
            raise ValueError("Probe shard is incomplete")
        inputs[str(shard)] = {"manifest_sha256": probe.file_sha(manifest_path),
                             "input_audit_sha256": probe.file_sha(directory / "input-audit.json"),
                             "result_trace_sha256": recorded.hexdigest(), "episodes": len(actual)}
    if set(indexed) != expected or len(indexed) != 204:
        raise ValueError("Complete 204-cell matrix is required")
    for family in families:
        rows = [indexed[family, c] for c in probe.CONDITIONS]
        audits = [{k: r[k] for k in audit_fields} for r in rows]
        location = tuple(rows[0]["replacement_message_and_part"])
        prefix = selected[family]["trace"]["messages"][:location[0] + 1]
        target = [(mi, pi) for mi, pi, _ in probe.image_parts(prefix)].index(location)
        probe.matched(audits, target)
        if (len({r["nonpixel_public_sha256"] for r in rows}) != 1
                or len({r["returned_rgb_sha256"] for r in rows}) != 3
                or len({r["pixel_blocks_sha256"][target] for r in rows}) != 3):
            raise ValueError("Three-way pixel matching failed")
    groups = {}
    for c in probe.CONDITIONS:
        rows = [indexed[f, c] for f in families]
        n = len(rows)
        groups[c] = {"episodes": n, "source_label_matches": sum(r["source_label_match"] for r in rows),
            "source_label_agreement_pct": 100 * sum(r["source_label_match"] for r in rows) / n,
            "schema_valid_finish_pct": 100 * sum(r["schema_valid_finish"] for r in rows) / n,
            "abstained_pct": 100 * sum(r["verdict"] == "Need more evidence" for r in rows) / n,
            "definite_pct": 100 * sum(r["verdict"] in ("Supported", "Refuted") for r in rows) / n,
            "verdict_counts": dict(Counter(r["verdict"] if r["schema_valid_finish"] else "Invalid finish" for r in rows)),
            "mean_final_call_s": sum(r["generation"].get("latency_s", 0) for r in rows) / n,
            "mean_input_tokens": sum(r["generation"].get("input_tokens", 0) for r in rows) / n,
            "mean_output_tokens": sum(r["generation"].get("output_tokens", 0) for r in rows) / n}
    comparisons = {}
    for c in ("gray", "shuffled"):
        comparisons["real_minus_" + c] = paired(
            [indexed[f, "real"]["source_label_match"] for f in families],
            [indexed[f, c]["source_label_match"] for f in families], families)
        comparisons["real_minus_" + c]["answer_transitions"] = dict(Counter(
            str(indexed[f, "real"]["verdict"]) + " -> " + str(indexed[f, c]["verdict"])
            for f in families))
    return {"complete": True, "scope": "Selected-history synthetic image-content probe; not native benchmark scores.",
        "histories": 68, "tasks": {"state": 68}, "episodes": 204, "families": families,
        "source_label_counts": dict(Counter(selected[f]["result"]["source_label"] for f in families)),
        "denominator": "All 68 selected histories in every condition, including schema failures and abstentions.",
        "matching": {"identical_nonpixel_public_inputs": 68, "identical_nonpixel_processor_tensors": 68,
                     "unchanged_initial_image_blocks": 68, "distinct_processed_target_blocks": 68,
                     "worker_token_counts_equal_preflight": 204},
        "conditions": groups, "paired_comparisons": comparisons,
        "bootstrap": {"seed": 20261106, "replicates": 20000, "unit": "paired case family",
                      "method": "central 95% percentile interval", "numpy": np.__version__},
        "source": source, "runner_configuration": common, "shards": inputs,
        "native_initial_scores_reused": False, "reviewed_grounded_accuracy": None,
        "interpretation": "Sensitivity to returned pixels or their spatial structure at histories chosen by one trained policy. Gray/shuffled images are artificial degradations. This does not measure adaptive view selection or independently sufficient evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "results", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = collect(args.repo, args.runtime, args.results)
    report["summarizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "conditions": report["conditions"]}))


if __name__ == "__main__":
    main()
