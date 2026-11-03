"""Mix source-disjoint identity/acquisition and condition decisions at equal counts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity", type=Path)
    parser.add_argument("condition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    sources = {name: json.loads((path / "manifest.json").read_text())
               for name, path in (("identity", args.identity), ("state", args.condition))}
    if len({manifest["eval_dataset_sha256"] for manifest in sources.values()}) != 1:
        raise ValueError("Source checks refer to different evaluation pools")
    selected, paths, stats = {}, {}, {}
    rng = random.Random(71)
    for split in ("train", "validation"):
        groups = {}
        for task, root in (("identity", args.identity), ("state", args.condition)):
            path = root / (split + ".jsonl")
            if sha(path) != sources[task]["files"][split]:
                raise ValueError("Audited source decisions changed")
            groups[task] = [json.loads(line) for line in path.open()]
        count = len(groups["state"])
        if len(groups["identity"]) < count:
            raise ValueError("Too few identity decisions")
        rows = []
        for task in ("identity", "state"):
            subset = rng.sample(groups[task], count)
            for row in subset:
                row["group"]["task"] = task
            rows.extend(subset)
        rng.shuffle(rows)
        selected[split] = rows
        paths[split] = {Path(block["image_path"]) for row in rows for msg in row["messages"]
                        if isinstance(msg.get("content"), list) for block in msg["content"]
                        if "image_path" in block}
        stats[split] = {"decision_turns": len(rows),
                       "families": len({row["group"]["family_id"] for row in rows}),
                       "by_task": dict(Counter(row["group"]["task"] for row in rows))}
    checksums = {path: sha(path) for path in paths["train"] | paths["validation"]}
    hashes = {split: {checksums[path] for path in paths[split]} for split in paths}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Mixed train/validation pixels overlap")
    args.output.mkdir(parents=True)
    for split, rows in selected.items():
        with (args.output / (split + ".jsonl")).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "scope": "Balanced source-supervised identity/acquisition and condition SFT; unreviewed sufficiency",
        "adaptation_name": "Mixed-source SFT", "seed": 71,
        "eval_dataset_sha256": next(iter(sources.values()))["eval_dataset_sha256"],
        "source_manifests": {name: {"sha256": sha(root / "manifest.json"), "manifest": sources[name]}
                             for name, root in (("identity", args.identity), ("state", args.condition))},
        "counts": stats, "checks": {"train_validation_image_hash_overlap": 0},
        "files": {split: sha(args.output / (split + ".jsonl")) for split in selected},
        "limits": "Weak source labels; no new-view, sufficiency or adaptive-search annotation",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "counts": stats}), flush=True)


if __name__ == "__main__":
    main()
