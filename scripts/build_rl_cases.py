"""Reconstruct private training worlds from audited source supervision, never benchmark labels."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from scripts.build_acquisition_sft import make_case as identity_case, VARIANTS
from scripts.build_condition_sft import make_case as condition_case


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_paths(messages):
    paths = {}
    for message in messages:
        reference = None
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") == "text":
                try:
                    reference = json.loads(block["text"]).get("image_id")
                except (ValueError, AttributeError):
                    reference = None
            if "image_path" in block:
                paths[reference] = block["image_path"]
    return paths


def select_balanced(rows, key, per_label, rng):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    if len(groups) != 2:
        raise ValueError("Expected balanced binary source labels")
    return [row for label in sorted(groups) for row in rng.sample(groups[label], per_label)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.runtime.resolve()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    identity_root = root / "training/co3d-acquisition-v1"
    identity_manifest = json.loads((identity_root / "manifest.json").read_text())
    state_path = root / "training/visa-source/source_manifest.json"
    state_manifest = json.loads(state_path.read_text())
    frozen_sha = sha(args.benchmark / "cases.json")
    if any(m["eval_dataset_sha256"] != frozen_sha for m in (identity_manifest, state_manifest)):
        raise ValueError("Source checks refer to a different frozen benchmark")
    rng, output, hashes, summaries = random.Random(83), {}, {}, {}
    for split, per_label in (("train", 64), ("validation", 16)):
        source_file = identity_root / (split + ".jsonl")
        if sha(source_file) != identity_manifest["files"][split]:
            raise ValueError("Audited identity source changed")
        identity_rows = [row for row in (json.loads(line) for line in source_file.open())
                         if row["group"]["variant"] == "Sufficient"]
        chosen_identity = select_balanced(identity_rows, lambda r: r["group"]["label"], per_label, rng)
        state_rows = [row for row in state_manifest["images"] if row["partition"] == split]
        chosen_state = select_balanced(state_rows, lambda r: r["label"], per_label, rng)
        cases, current_paths = [], set()
        for row in chosen_identity:
            paths = image_paths(row["messages"])
            citations = row["target"]["tool_calls"][0]["function"]["arguments"]["citations"]
            pair = [{"path": paths[citation["image_id"]], "bbox": citation["bbox"]}
                    for citation in citations]
            for variant in VARIANTS:
                case, _, _ = identity_case(*pair, row["group"]["label"], variant, root)
                case["asset_root"] = str(root)
                case["group_ids"] = ["co3d:" + s for s in row["group"]["sequences"]]
                cases.append(case)
            current_paths.update(Path(item["path"]) for item in pair)
        for row in chosen_state:
            reference = state_manifest["references"][row["object"]]["image"]
            state_root = root / "visa-source"
            original, _, _ = condition_case(row, reference, state_root)
            for variant in VARIANTS:
                case = deepcopy(original)
                case.update(case_id=original["family_id"]+"-"+variant, variant=variant,
                            asset_root=str(state_root), official_split="train",
                            group_ids=["visa:" + row["object"]])
                if variant != "Sufficient":
                    case["initial"] = [original["evidence"][1]["id"]]
                if variant == "Missing":
                    case["evidence"] = case["evidence"][1:]
                elif variant == "Unavailable":
                    case["evidence"][0]["available"] = False
                cases.append(case)
            current_paths.update((state_root / row["image"], state_root / reference))
        allowed = set(identity_manifest["source_sha256"][split])
        allowed.update(row["sha256"] for row in state_rows)
        allowed.update(state_manifest["references"][row["object"]]["sha256"] for row in state_rows)
        hashes[split] = {sha(path) for path in current_paths}
        if not hashes[split] <= allowed:
            raise ValueError("A training source image changed after audit")
        output[split] = cases
        summaries[split] = {"families": len(chosen_identity)+len(chosen_state), "cases": len(cases),
                            "identity_families": len(chosen_identity), "state_families": len(chosen_state)}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("RL train/validation pixels overlap")
    if {c["family_id"] for c in output["train"]} & {c["family_id"] for c in output["validation"]}:
        raise ValueError("RL family overlap")
    args.output.mkdir(parents=True)
    for split, cases in output.items():
        (args.output / (split+"_cases.json")).write_text(json.dumps(cases, separators=(",", ":"))+"\n")
    manifest = {
        "scope": "Source-supervised private training worlds for online RL; unreviewed visual sufficiency",
        "seed": 83, "counts": summaries, "eval_dataset_sha256": frozen_sha,
        "identity_source_sha256": sha(identity_root / "manifest.json"),
        "condition_source_sha256": sha(state_path),
        "checks": {"train_validation_pixel_overlap": 0, "train_validation_family_overlap": 0,
                   "source_images_unchanged": True},
        "files": {split: sha(args.output/(split+"_cases.json")) for split in output},
        "limits": "Easy source identity and condition supervision; no adaptive-search claim",
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"stage": "complete", "counts": summaries}), flush=True)


if __name__ == "__main__":
    main()
