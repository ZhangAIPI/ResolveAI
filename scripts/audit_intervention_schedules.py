"""Audit request-path coverage and public restoration/oracle action equivalence.

Uses provisional record-set annotations for structural checks, never visual
sufficiency judgments or model inference. Only the requested audit JSON is saved.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.repository), str(args.repository / "src")]
    path = Path(__file__).with_name("evaluate_evidence_controls.py")
    spec = importlib.util.spec_from_file_location("staged_controls_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class NoOCR:
        available = False

    class CaptureOnly:
        def __init__(self):
            self.payload = None

        def ask_conversation(self, payload, max_context_tokens=8192):
            self.payload = payload
            return {"halt": "serialization_probe_without_inference"}

    plan = json.loads(args.plan.read_text())
    cohort = json.loads(args.cohort.read_text())
    data = Path(plan["data"]) / "cases.json"
    if sha256(data) != plan["dataset_sha256"] or cohort["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("Frozen data differs")
    selected = [c for c in json.loads(data.read_text()) if c["family_id"] in cohort["family_ids"]]
    if len(selected) != len(cohort["family_ids"]) * 4:
        raise ValueError("Incomplete family/access matrix")

    def public_metadata(case, budget):
        content = [{"type": "text", "text": json.dumps({
            "request_options": case["request_options"], "budget": budget,
            "tool_costs": {"request_photo": 3}})}]
        evidence = {e["id"]: e for e in case["evidence"]}
        for image_id in case["initial"]:
            e = evidence[image_id]
            with Image.open(Path(case["asset_root"]) / e["path"]) as image:
                width, height = image.size
            box = e.get("source_bbox", [0, 0, width, height])
            metadata = {
                "image_id": image_id, "view_id": image_id, "source_id": e["source_id"],
                "party": e["party"], "time": e["time"], "object": e["object"],
                "camera_view": e.get("camera_view", e["view"]),
                "source_bbox": [math.floor(box[0]), math.floor(box[1]),
                                math.ceil(box[2]), math.ceil(box[3])],
                "source_size": e.get("source_size", [width, height]),
                "display_size": [width, height],
            }
            if "target_bbox" in e:
                metadata["target_bbox"] = e["target_bbox"]
            content.append({"type": "text", "text": json.dumps(metadata)})
        return {"messages": [{"role": "user", "content": content}], "tools": []}

    counts, equality, records, witnesses = Counter(), Counter(), [], {}
    for budget in (3, 6, 12):
        for case in selected:
            env = module.Environment(case, case["asset_root"], budget, ocr_backend=NoOCR())
            public = public_metadata(case, budget)
            calls, info = module.annotation_oracle_schedule(env, public, 8)
            counts[(budget, case["task"], case["variant"],
                    info["annotation_coverage_reachable"], info["minimum_request_count"])] += 1
            if budget != 3 or case["variant"] != "Obtainable":
                continue
            restored = module.request_schedule(public, "restore_preview", 0, case["case_id"])
            encode = lambda value: json.dumps(value, separators=(",", ":")).encode()
            exact = encode(calls) == encode(restored)
            equality[(case["task"], exact)] += 1
            # Resolve the calls without loading images or executing any request.
            source_ids, state = [], set(case["initial"])
            for call in restored:
                target = env._world.select(call["arguments"]["query"], state)
                source_ids.append(target["id"] if target else None)
                if target:
                    state.add(target["id"])
            records.append({
                "family_id": case["family_id"], "task": case["task"],
                "byte_identical_calls": exact, "calls": restored,
                "query_bytes_sha256": hashlib.sha256(encode(restored)).hexdigest(),
                "resolved_record_ids": source_ids,
            })
            if exact and case["task"] not in witnesses:
                a, b = CaptureOnly(), CaptureOnly()
                first = module.Environment(case, case["asset_root"], 3, ocr_backend=NoOCR())
                second = module.Environment(case, case["asset_root"], 3, ocr_backend=NoOCR())
                module.controlled_episode(first, a, "restore_preview")
                module.controlled_episode(second, b, "annotation_oracle")
                if encode(a.payload) != encode(b.payload):
                    raise ValueError("Witness model-input histories differ")
                witnesses[case["task"]] = {
                    "family_id": case["family_id"], "actual_input_bytes_equal": True,
                    "actual_input_sha256": hashlib.sha256(encode(a.payload)).hexdigest(),
                    "model_inference_calls": 0,
                }

    report = {
        "scope": "Provisional annotation image-record coverage, not reviewed visual sufficiency",
        "dataset_sha256": plan["dataset_sha256"], "cohort_sha256": sha256(args.cohort),
        "controller_sha256": sha256(path), "audit_script_sha256": sha256(__file__),
        "families": len(cohort["family_ids"]), "cases": len(selected),
        "counts": [
            {"budget": k[0], "task": k[1], "variant": k[2], "reachable": k[3],
             "minimum_requests": k[4], "cases": n}
            for k, n in sorted(counts.items(), key=lambda x: str(x[0]))],
        "obtainable_call_equivalence": [
            {"task": k[0], "byte_identical_calls": k[1], "cases": n}
            for k, n in sorted(equality.items())],
        "serialization_witnesses": witnesses, "obtainable_records": records,
        "reuse_scope": (
            "Identical request actions with the same environment and terminal prompt imply identical "
            "model inputs. Restoration outputs can serve as the annotation-guided action reference. "
            "Oracle controller wall time is not independently measured by that reuse."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "equivalence": report["obtainable_call_equivalence"],
        "witnesses": witnesses, "families": report["families"], "model_inference_calls": 0,
    }), flush=True)


if __name__ == "__main__":
    main()
