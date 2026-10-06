"""Public-image benchmark repair pool, with explicit targets and review gates.

Source labels propose cases; they never certify visual sufficiency. All pictures
remain on p62. Reference pictures come from official training material.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image

from .co3d_data import identifier
from .public_data import BASE, VARIANTS, fetch

# These are public claim definitions, not model observations generated from truth.
CONDITIONS = {
    "bottle": {"broken_large": "a chipped or broken rim", "broken_small": "a chipped or broken rim",
               "contamination": "visible foreign material or contamination"},
    "cable": {"bent_wire": "a bent wire", "cable_swap": "a wire arrangement inconsistent with the normal reference",
              "cut_outer_insulation": "cut outer insulation", "missing_cable": "a missing cable",
              "missing_wire": "a missing wire", "poke_insulation": "a puncture in the insulation"},
    "capsule": {"crack": "a crack", "faulty_imprint": "a faulty printed marking", "poke": "a puncture",
                "scratch": "a surface scratch", "squeeze": "a squeezed or deformed body"},
    "hazelnut": {"crack": "a crack", "cut": "a cut", "hole": "a hole",
                 "print": "an abnormal surface marking"},
    "metal_nut": {"bent": "a bent or deformed body", "color": "abnormal discoloration",
                  "flip": "an orientation inconsistent with the normal reference", "scratch": "a surface scratch"},
    "pill": {"contamination": "visible contamination", "crack": "a crack",
              "faulty_imprint": "a faulty printed marking", "pill_type": "an appearance inconsistent with the normal pill reference"},
    "screw": {"manipulated_front": "a deformed tip", "scratch_head": "a scratch on the head",
               "scratch_neck": "a scratch on the neck", "thread_side": "damaged threading on the side",
               "thread_top": "damaged threading near the tip"},
    "toothbrush": {"defective": "a bristle arrangement inconsistent with the normal reference"},
    "transistor": {"bent_lead": "a bent lead", "cut_lead": "a cut lead", "damaged_case": "a damaged case",
                   "misplaced": "a placement inconsistent with the normal reference"},
    "zipper": {"broken_teeth": "broken teeth", "fabric_border": "damage to the fabric border",
               "rough": "rough or damaged teeth", "split_teeth": "split teeth", "squeezed_teeth": "deformed teeth"},
}
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def stable(value):
    return hashlib.sha256(("benchmark-v04|" + value).encode()).hexdigest()


def endpoint(evidence, bbox):
    return {"image_id": evidence["id"], "bbox": bbox, "time": evidence["time"]}


def proposal(verdict, kind, requirements):
    truth = "uncertain" if verdict == "Need more evidence" else verdict
    return {"protocol": "evidence-chain-v1", "status": "source-derived-unreviewed",
            "verdict": verdict, "subclaims": [{"id": kind, "kind": kind, "truth": truth,
                "minimal_evidence_sets": {} if truth == "uncertain" else {verdict: [requirements]}}]}


def variants(family, withheld):
    """Availability versions describe release configuration, not semantic labels."""
    for variant in VARIANTS:
        case = deepcopy(family)
        case["case_id"] = case["family_id"] + "-" + variant
        case["variant"] = variant
        if variant == "Sufficient":
            case["initial"] = [withheld if i == withheld + "-preview" else
                               "original" if i == "preview" and withheld == "original" else i
                               for i in case["initial"]]
        if variant == "Missing":
            case["evidence"] = [e for e in case["evidence"] if e["id"] != withheld]
        elif variant == "Unavailable":
            for evidence in case["evidence"]:
                if evidence["id"] == withheld:
                    evidence["available"] = False
        yield case


def reference_images(mvtec, output, runtime):
    metadata = json.loads((mvtec / "mvtec_samples.json").read_text())["samples"]
    references = {}
    for category in sorted(CONDITIONS):
        normal = [r for r in metadata if r["split"] == "train"
                  and r["category"]["label"] == category and r["defect"]["label"] == "good"]
        if not normal:
            raise ValueError("official training reference absent: " + category)
        normal.sort(key=lambda r: stable("reference|" + r["_id"]["$oid"]))
        row = normal[0]
        path = fetch(BASE + "/" + row["filepath"], output / "references" / (category + ".png"))
        with Image.open(path) as image:
            size = list(image.size)
        references[category] = {
            "id": "reference", "source_id": "reference-" + row["_id"]["$oid"],
            "path": str(path.relative_to(runtime)), "object": "normal-reference",
            "party": "reference", "time": "reference", "view": "reference", "available": True,
            "source_size": size, "source_bbox": [0, 0, *size],
            "provenance": {"sample_id": row["_id"]["$oid"], "official_split": "train",
                           "filepath": row["filepath"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        }
    return references


def state_families(mvtec, references, runtime):
    sources = [c for c in json.loads((mvtec / "cases.json").read_text()) if c["variant"] == "Obtainable"]
    families, excluded = [], []
    for case in sorted(sources, key=lambda c: stable(c["family_id"])):
        category = case["provenance"]["category"]
        label = case["provenance"]["defect_label"]
        if category not in CONDITIONS or (label != "good" and label not in CONDITIONS[category]):
            excluded.append({"family_id": case["family_id"], "category": category, "defect_label": label,
                             "reason": "texture-auxiliary" if category in TEXTURES else "condition-needs-definition"})
            continue
        choices = sorted(set(CONDITIONS[category].values()))
        criterion = choices[int(stable(case["family_id"])[:8], 16) % len(choices)] if label == "good" else CONDITIONS[category][label]
        evidence = deepcopy(case["evidence"])
        for row in evidence:
            row["path"] = str((mvtec / row["path"]).relative_to(runtime))
            row["party"] = "submitted"
            if row["id"] == "original":
                row["source_size"] = case["provenance"]["original_size"]
        ref = deepcopy(references[category])
        provenance = ref.pop("provenance")
        evidence.append(ref)
        query = next(e for e in evidence if e["id"] == "original")
        query_box = [0, 0, *query["source_size"]]
        ref_box = [0, 0, *ref["source_size"]]
        claim = f"The target {category.replace('_', ' ')} in the submitted photo shows {criterion}."
        family = {
            "family_id": "case-" + stable(case["family_id"] + "|" + criterion)[:20],
            "task": "state", "category": category, "claim": claim,
            "claim_parts": [{"id": "condition", "text": claim}],
            "task_instructions": "The reference is a separate normal specimen, not a before photo of the target. "
                 "Judge only the stated condition; other unusual features do not establish this claim.",
            "initial": ["reference", "preview"], "evidence": evidence,
            "request_options": deepcopy(case["request_options"]), "asset_root": str(runtime),
            "group_ids": ["mvtec:" + case["provenance"]["sample_id"]],
            "review_partition": "development", "official_split": "test",
            "provenance_truth": case["annotation"]["visual_verdict"],
            "provenance": {**case["provenance"], "reference": provenance, "criterion": criterion},
            "annotation": proposal(case["annotation"]["visual_verdict"], "condition",
                                   [endpoint(query, query_box), endpoint(ref, ref_box)]),
        }
        families.append((family, "original"))
    return families, excluded


def identity_families(co3d, runtime):
    sources = [c for c in json.loads((co3d / "cases.json").read_text()) if c["variant"] == "Obtainable"]
    groups = defaultdict(list)
    excluded = []
    for case in sources:
        full = [e for e in case["evidence"] if not e["id"].endswith("-preview")]
        if len(full) < 4:
            excluded.append({"family_id": case["family_id"], "reason": "fewer-than-four-distinct-views"})
            continue
        groups[case["category"]].append(case)
    for bucket in groups.values():
        bucket.sort(key=lambda c: stable(c["family_id"]))
    categories = sorted(groups)
    families = []
    for n, category in enumerate(categories):
        for index, anchor in enumerate(groups[category]):
            choices = [
                ("source-positive", anchor),
                ("easy-negative", groups[categories[(n + 1) % len(categories)]][index % len(groups[categories[(n + 1) % len(categories)]])]),
            ]
            if len(groups[category]) > 1:
                choices.append(("same-category-candidate", groups[category][(index + 1) % len(groups[category])]))
            for kind, partner in choices:
                lefts = [e for e in anchor["evidence"] if not e["id"].endswith("-preview")][:2]
                rights = [e for e in partner["evidence"] if not e["id"].endswith("-preview")][-2:]
                evidence = []
                for slot, rows in [("subject-A", lefts), ("subject-B", rights)]:
                    for view, row in enumerate(rows, 1):
                        row = deepcopy(row)
                        row["path"] = str((co3d / row["path"]).relative_to(runtime))
                        row["object"], row["view"], row["party"] = slot, f"view-{view:02d}", "submitted"
                        evidence.append(row)
                first, last = evidence[0], evidence[-1]
                original_preview = next(e for e in partner["evidence"] if e["id"] == last["id"] + "-preview")
                preview = {**deepcopy(original_preview), "path": str((co3d / original_preview["path"]).relative_to(runtime)),
                           "object": "subject-B", "view": "preview", "camera_view": "view-02", "party": "submitted"}
                evidence.append(preview)
                aedge = anchor["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0][0]
                bedge = partner["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0][0]
                first["target_bbox"] = aedge["left"]["bbox"]
                last["target_bbox"] = bedge["right"]["bbox"]
                preview["target_bbox"] = bedge["right"]["bbox"]
                verdict = "Supported" if kind == "source-positive" else "Refuted" if kind == "easy-negative" else "Need more evidence"
                relation = {"relation": "same_object" if verdict == "Supported" else "different_object",
                            "left": endpoint(first, first["target_bbox"]),
                            "right": endpoint(last, last["target_bbox"]), "min_iou": .5}
                seq_a, seq_b = anchor["scene"]["source_sequence"], partner["scene"]["source_sequence"]
                family = {
                    "family_id": "case-" + stable(anchor["family_id"] + "|" + partner["family_id"] + "|" + kind)[:20],
                    "task": "identity", "category": category,
                    "claim": "The target objects in subject-A and subject-B are the same physical item.",
                    "claim_parts": [{"id": "identity", "text":
                        "Compare the specified foreground targets. Ignore other items and background. "
                        "Subject-A and subject-B name submission slots, not known physical identities. "
                        "Other views can be requested for each slot. Similarity alone does not prove identity."}],
                    "initial": [first["id"], preview["id"]], "evidence": evidence,
                    "request_options": {"objects": ["subject-A", "subject-B"], "times": ["capture"],
                                        "views": ["view-01", "view-02"]},
                    "asset_root": str(runtime), "group_ids": sorted({"co3d:" + seq_a, "co3d:" + seq_b}),
                    "review_partition": "development", "official_split": "test",
                    "provenance_truth": None if kind == "same-category-candidate" else verdict,
                    "provenance": {"construction": kind, "source_sequences": [seq_a, seq_b],
                        "source_categories": [category, partner["category"]],
                        "note": "Different sequences do not establish different physical objects."},
                    "scene": {"source_sequences": [seq_a, seq_b], "source_type": "captured_multiview"},
                    "annotation": proposal(verdict, "identity", [relation]),
                }
                families.append((family, last["id"]))
    return families, excluded


def prepare(mvtec, co3d, output):
    if output.exists():
        raise ValueError("Use a fresh directory; never overwrite frozen datasets")
    runtime = mvtec.parent.resolve()
    if co3d.parent.resolve() != runtime or not output.resolve().is_relative_to(runtime):
        raise ValueError("Use the shared p62 runtime as the private asset root")
    output.mkdir(parents=True)
    refs = reference_images(mvtec, output, runtime)
    states, excluded_states = state_families(mvtec, refs, runtime)
    identities, excluded_identities = identity_families(co3d, runtime)
    families = states + identities
    cases = [c for family, withheld in families for c in variants(family, withheld)]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    from .environment import Environment
    checked = 0
    for family, withheld in families:
        env = Environment(family, runtime, ocr_backend=NoOCR())
        fingerprint = env._world.fingerprint
        row = next(e for e in family["evidence"] if e["id"] == withheld)
        result = env.step({"type": "request_photo", "query": {k: row[k] for k in ["object", "time", "view"]}})
        if result["status"] != "provided" or withheld not in env._released or env._world.fingerprint != fingerprint:
            raise RuntimeError("Retrieval or immutable-world preflight failed")
        checked += 1
    path = output / "cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    manifest = {
        "protocol": "benchmark-v0.4", "stage": "unreviewed-curation-pool", "formal_evaluation_ready": False,
        "families": len(families), "cases": len(cases),
        "tasks": dict(Counter(f["task"] for f, _ in families)),
        "identity_constructions": dict(Counter(f["provenance"]["construction"] for f, _ in identities)),
        "state_categories": dict(Counter(f["category"] for f, _ in states)),
        "unique_query_images": len(states), "public_train_references": len(refs),
        "unique_co3d_sequences": len({g for f, _ in identities for g in f["group_ids"]}),
        "preflight": {"correct_request_and_unchanged_world": checked},
        "excluded_states": excluded_states, "excluded_identities": excluded_identities,
        "source_sha256": {name: hashlib.sha256((root / "cases.json").read_bytes()).hexdigest()
                          for name, root in [("mvtec", mvtec), ("co3d", co3d)]},
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "Public development material. Not a fresh held-out benchmark or training supervision.",
        "grouping": "Share source-image and sequence groups across rewrites, pairs and availability versions.",
        "uncertainty": "Same-category pairs have unknown physical truth until visual review; no automatic negative label.",
        "licenses": ["MVTec AD: CC-BY-NC-SA-4.0", "CO3D: CC-BY-NC-4.0"],
        "storage": "Existing query images referenced in place; only official-train normal references downloaded.",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "review_template.json").write_text(json.dumps({
        "protocol": "visual-review-v1", "dataset_sha256": manifest["cases_sha256"], "reviewer_id": "", "reviewer_type": "human",
        "family_id": "", "claim_clear": False, "alternatives_checked": False,
        "reason": "", "initial_verdicts": {v: None for v in VARIANTS},
        "pool_verdicts": {v: None for v in VARIANTS},
        "annotation": {"protocol": "evidence-chain-v1", "verdict": None, "subclaims": []},
    }, indent=2) + "\n")
    return manifest


class NoOCR:
    available = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mvtec", type=Path)
    parser.add_argument("co3d", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.mvtec.resolve(), args.co3d.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in manifest.items() if not k.startswith("excluded_")}, indent=2))


if __name__ == "__main__":
    main()
