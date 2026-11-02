"""Build a VisA condition-perception SFT supplement from audited training sources.

Each original target is paired with a normal reference from a different item.
This supplies condition supervision, not new-view acquisition trajectories.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding
from scripts.build_acquisition_sft import digest, public_example


def make_case(row, reference, root):
    def evidence(relative, slot):
        image_id = "photo-" + hashlib.sha256(relative.encode()).hexdigest()[:20]
        return {"id": image_id, "source_id": image_id, "party": "submitted",
                "object": slot, "time": "capture", "view": "original",
                "path": relative, "available": True}

    target, normal = evidence(row["image"], "target"), evidence(reference, "reference")
    with Image.open(root / row["image"]) as image:
        box = [0, 0, *image.size]
    label = "Supported" if row["label"] == "anomaly" else "Refuted"
    citation = {"image_id": target["id"], "bbox": box, "time": "capture"}
    family = "visa-sft-" + hashlib.sha256(row["image"].encode()).hexdigest()[:20]
    case = {
        "case_id": family, "family_id": family, "task": "state", "variant": "Sufficient",
        "claim": "The target item has a visible defect.",
        "task_instructions": "Judge the target image. The normal reference shows another physical "
                             "item of the same category, not an earlier state of the target.",
        "initial": [target["id"], normal["id"]], "evidence": [target, normal],
        "request_options": {"objects": ["target", "reference"],
                            "times": ["capture"], "views": ["original"]},
        "annotation": {"protocol": "evidence-chain-v1", "status": "source-derived-unreviewed",
                       "verdict": label, "subclaims": [{"id": "condition", "kind": "state",
                         "truth": label, "minimal_evidence_sets": {label: [[citation]]}}]},
    }
    call = {"name": "finish", "arguments": {"verdict": label, "citations": [citation], "links": []}}
    paths = {item["id"]: str(root / item["path"]) for item in case["evidence"]}
    return case, call, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("grounding", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_path = args.metadata / "source_manifest.json"
    source = json.loads(source_path.read_text())
    if args.output.exists():
        raise ValueError("Use a fresh training directory")
    if any(source["checks"].values()) or source["official_split"] != "train":
        raise ValueError("Source isolation checks failed")
    if digest(args.benchmark / "cases.json") != source["eval_dataset_sha256"]:
        raise ValueError("Frozen benchmark changed")
    root = args.assets.resolve()
    for row in source["images"]:
        if row["split"] != "train" or digest(root / row["image"]) != row["sha256"]:
            raise ValueError("Source split or image changed")
    for row in source["references"].values():
        if digest(root / row["image"]) != row["sha256"]:
            raise ValueError("Reference pixels changed")
    args.output.mkdir(parents=True)
    verified, counts = set(), {}
    backend = FrozenGrounding(args.grounding, device="cpu")
    for partition in ("train", "validation"):
        stats = Counter()
        with (args.output / (partition + ".jsonl")).open("w") as stream:
            for row in source["images"]:
                if row["partition"] != partition:
                    continue
                reference = source["references"][row["object"]]["image"]
                case, call, paths = make_case(row, reference, root)
                conversation = Conversation(Environment(case, root, grounding_backend=backend))
                prefix = deepcopy(conversation.messages)
                conversation.call(call)
                if conversation.errors or conversation.termination != "finished":
                    raise ValueError("Invalid source-supervised finish")
                example = public_example(prefix, conversation.messages[-2],
                                         conversation.tools, paths, verified)
                example["group"] = {"family_id": case["family_id"], "task": "state",
                                    "category": row["object"], "source_label": row["label"]}
                stream.write(json.dumps(example, separators=(",", ":")) + "\n")
                stats[row["label"]] += 1
                if sum(stats.values()) % 100 == 0:
                    print(json.dumps({"partition": partition, **stats}), flush=True)
        counts[partition] = dict(stats)
    manifest = {
        "scope": "Auxiliary source-supervised condition perception; not acquisition or reviewed sufficiency",
        "source_manifest_sha256": digest(source_path), "source": source["provenance"],
        "counts": counts, "category_partitions": source["category_partitions"],
        "eval_dataset_sha256": source["eval_dataset_sha256"],
        "checks": {**source["checks"], "environment_pixels_verified": len(verified)},
        "files": {split: digest(args.output / (split + ".jsonl")) for split in counts},
        "limits": "No paired state change, new viewpoints, defect-mask hints, or adaptive-search targets",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "counts": counts, "checks": manifest["checks"]}), flush=True)


if __name__ == "__main__":
    main()
