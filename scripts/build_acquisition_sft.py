"""Source-supervised CO3D acquisition pilot, disjoint from the frozen evaluation pool.

Only official manyview_dev train frames are used. Category-disjoint train/dev
partitions prevent sequence sharing within this pilot. These are weak targets,
not independent judgments of visual sufficiency or adaptive-search supervision.
"""
import argparse
import base64
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import random

from PIL import Image, ImageChops
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding

VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_categories(categories, seed=17):
    categories = sorted(categories)
    random.Random(seed).shuffle(categories)
    cut = max(2, round(len(categories) * .2))
    return {"train": sorted(categories[cut:]), "validation": sorted(categories[:cut])}


def assert_disjoint(groups, hashes, eval_groups, eval_hashes):
    if groups["train"] & groups["validation"]:
        raise ValueError("Train/validation sequence overlap")
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Train/validation source-pixel overlap")
    for split in groups:
        if groups[split] & eval_groups or hashes[split] & eval_hashes:
            raise ValueError("Frozen benchmark leakage")


def public_example(messages, target, tools, paths, verified):
    result = deepcopy(messages)
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        reference = None
        for item in message["content"]:
            if item.get("type") == "text":
                try:
                    reference = json.loads(item["text"]).get("image_id")
                except (ValueError, AttributeError):
                    reference = None
            if "image_png" in item:
                path = paths[reference]
                if path not in verified:
                    shown = Image.open(BytesIO(base64.b64decode(item["image_png"]))).convert("RGB")
                    source = Image.open(path).convert("RGB")
                    if shown.size != source.size or ImageChops.difference(shown, source).getbbox():
                        raise ValueError("Training image differs from actual environment observation")
                    verified.add(path)
                item.pop("image_png")
                item["image_path"] = path
    return {"messages": result, "target": deepcopy(target), "tools": deepcopy(tools)}


def make_case(a, b, positive, variant, root):
    label = "Supported" if positive else "Refuted"
    evidence = []
    for slot, row in zip(("subject-A", "subject-B"), (a, b)):
        opaque = "photo-" + hashlib.sha256(row["path"].encode()).hexdigest()[:20]
        evidence.append({
            "id": opaque, "source_id": opaque, "party": "submitted",
            "object": slot, "time": "capture", "view": "view-02",
            "path": str(Path(row["path"]).relative_to(root)), "available": True,
            "target_bbox": row["bbox"],
        })
    left, right = [{"image_id": e["id"], "bbox": e["target_bbox"], "time": "capture"}
                   for e in evidence]
    relation = "same_object" if positive else "different_object"
    requirement = {"relation": relation, "left": left, "right": right, "min_iou": .5}
    family = "sft-" + hashlib.sha256("\n".join(sorted([a["path"], b["path"]])).encode()).hexdigest()[:20]
    case = {
        "case_id": family + "-" + variant, "family_id": family,
        "task": "identity", "variant": variant, "official_split": "train",
        "claim": "The target objects in subject-A and subject-B are the same physical item.",
        "claim_parts": [{"id": "identity", "text": "Compare the specified foreground targets. "
            "Submission slots are not physical identities. Use visual evidence, not background alone."}],
        "task_instructions": "Judge the foreground objects; request another submission if needed.",
        "initial": [e["id"] for e in evidence] if variant == "Sufficient" else [evidence[0]["id"]],
        "evidence": evidence,
        "request_options": {"objects": ["subject-A", "subject-B"],
                            "times": ["capture"], "views": ["view-02"]},
        "annotation": {"protocol": "evidence-chain-v1", "status": "source-derived-unreviewed",
            "verdict": label, "subclaims": [{"id": "identity", "kind": "identity", "truth": label,
                "minimal_evidence_sets": {label: [[requirement]]}}]},
    }
    paths = {e["id"]: str(root / e["path"]) for e in evidence}
    if variant == "Missing":
        case["evidence"] = evidence[:1]
    elif variant == "Unavailable":
        case["evidence"][1]["available"] = False
    final = {"name": "finish", "arguments": {
        "verdict": label if variant in ("Sufficient", "Obtainable") else "Need more evidence",
        "citations": [left, right] if variant in ("Sufficient", "Obtainable") else [],
        "links": [{"relation": relation, "left": left, "right": right}]
                 if variant in ("Sufficient", "Obtainable") else [],
    }}
    return case, final, paths


class CachedEnvironment(Environment):
    """Cache immutable source encodings, while checking release on every access."""
    cache = {}

    def _image(self, reference, bbox=None):
        if bbox is not None or reference not in self._released:
            return super()._image(reference, bbox)
        key = (str(self._root), json.dumps(self._evidence[reference], sort_keys=True))
        if key not in self.cache:
            if len(self.cache) > 1024:
                self.cache.clear()
            self.cache[key] = super()._image(reference)
        return deepcopy(self.cache[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs-per-label", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh directory; never append duplicate training examples")
    root = args.root.resolve()
    pool = json.loads((args.benchmark / "cases.json").read_text())
    eval_groups = {g for c in pool for g in c["group_ids"]}
    eval_hashes = {digest(Path(c["asset_root"]) / e["path"]) for c in pool for e in c["evidence"]}
    sequences, annotation_files = {}, {}
    for path in sorted((root / "co3d").glob("*/set_lists/set_lists_manyview_dev_*.json")):
        category = path.parent.parent.name
        annotation_files[category] = path.parent.parent / "frame_annotations.jgz"
        for seq, frame, relative in json.loads(path.read_text())["train"]:
            if "co3d:" + seq in eval_groups:
                raise ValueError("Official dev sequence appears in frozen benchmark")
            image = root / "co3d" / relative
            if image.exists():
                sequences.setdefault((category, seq), {})[frame] = str(image)
    categories = split_categories({key[0] for key in sequences})
    groups, hashes = {k: set() for k in categories}, {k: set() for k in categories}
    selected = {}
    for category, path in annotation_files.items():
        with gzip.open(path, "rt") as handle:
            annotations = {row["image"]["path"]: row for row in json.load(handle)}
        for (cat, seq), frames in sequences.items():
            if cat != category or len(frames) < 12:
                continue
            ordered = sorted(frames.items())
            sampled = [ordered[round(i * (len(ordered) - 1) / 11)] for i in range(12)]
            rows = []
            for frame, image in sampled:
                rel = str(Path(image).relative_to(root / "co3d"))
                mask = root / "co3d" / annotations[rel]["mask"]["path"]
                with Image.open(mask) as source:
                    bbox = source.convert("L").point(lambda p: 255 if p > 127 else 0).getbbox()
                if not bbox:
                    continue
                rows.append({"path": image, "frame": frame, "bbox": list(bbox),
                             "category": category, "sequence": seq, "sha256": digest(Path(image))})
            selected[(category, seq)] = rows
    args.output.mkdir(parents=True)
    rng, verified, all_pairs, counts = random.Random(17), set(), set(), {}
    backend = FrozenGrounding(root / "grounding")
    for split, cats in categories.items():
        current = {key: rows for key, rows in selected.items() if key[0] in cats}
        for (_, seq), rows in current.items():
            groups[split].add("co3d:" + seq)
            hashes[split].update(row["sha256"] for row in rows)
        pairs = []
        for (cat, seq), rows in current.items():
            positive = [(a, b, True) for a, b in combinations(rows, 2)
                        if abs(a["frame"] - b["frame"]) >= 30]
            negative = [(a, b, False) for a in rows
                        for (other_cat, _), other in current.items() if other_cat != cat
                        for b in other]
            rng.shuffle(positive)
            rng.shuffle(negative)
            for candidates in (positive, negative):
                taken = 0
                for pair in candidates:
                    key = tuple(sorted([pair[0]["path"], pair[1]["path"]]))
                    if key in all_pairs:
                        continue
                    all_pairs.add(key)
                    pairs.append(pair)
                    taken += 1
                    if taken == args.pairs_per_label:
                        break
        stats = Counter()
        with (args.output / (split + ".jsonl")).open("w") as handle:
            for a, b, positive in pairs:
                for variant in VARIANTS:
                    case, final, paths = make_case(a, b, positive, variant, root)
                    convo = Conversation(CachedEnvironment(case, root, grounding_backend=backend))
                    calls = [] if variant == "Sufficient" else [{
                        "name": "request_photo", "arguments": {
                            "query": {"object": "subject-B", "time": "capture", "view": "view-02"}}}]
                    calls.append(final)
                    for call in calls:
                        prefix = deepcopy(convo.messages)
                        outcome = convo.call(call)
                        if convo.errors:
                            raise ValueError(("Expert tool failure", case["case_id"], outcome))
                        target = convo.messages[-2]
                        example = public_example(prefix, target, convo.tools, paths, verified)
                        example["group"] = {"family_id": case["family_id"], "variant": variant,
                            "sequences": [a["sequence"], b["sequence"]], "label": positive}
                        handle.write(json.dumps(example, separators=(",", ":")) + "\n")
                        stats["decision_turns"] += 1
                    stats[variant] += 1
                    stats["episodes"] += 1
                stats["families"] += 1
                if stats["families"] % 100 == 0:
                    print(json.dumps({"split": split, **stats}), flush=True)
        counts[split] = dict(stats)
    assert_disjoint(groups, hashes, eval_groups, eval_hashes)
    manifest = {
        "scope": "Weak source-supervised acquisition pilot; not reviewed sufficiency or adaptive search",
        "seed": 17, "official_source_split": "manyview_dev_[01]/train frames only",
        "category_partitions": categories,
        "sequence_groups": {k: sorted(v) for k, v in groups.items()},
        "source_sha256": {k: sorted(v) for k, v in hashes.items()},
        "eval_dataset_sha256": digest(args.benchmark / "cases.json"),
        "checks": {"eval_sequence_overlap": 0, "eval_image_hash_overlap": 0,
                   "train_validation_sequence_overlap": 0, "train_validation_hash_overlap": 0,
                   "environment_pixels_verified": len(verified)},
        "counts": counts, "files": {k: digest(args.output / (k + ".jsonl")) for k in counts},
        "teacher": "Source identity / cross-category negatives; source mask regions; actual tool responses",
        "limits": "No same-category hard negatives, independent sufficiency labels or adaptive action preferences",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "counts": counts, "checks": manifest["checks"]}), flush=True)


if __name__ == "__main__":
    main()
