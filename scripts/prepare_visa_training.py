"""Cache a balanced VisA auxiliary training pool from the official 2-class train split.

Streams the official archive and retains only selected images/masks. The frozen
ResolveAI evaluation pool is used solely for overlap checks, never for labels.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import urllib.request

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(csv_path):
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    categories = sorted({row["object"] for row in rows})
    random.Random(17).shuffle(categories)
    validation = set(categories[:3])
    selected, references = [], {}
    rng = random.Random(17)
    for category in sorted(categories):
        current = [row for row in rows if row["object"] == category and row["split"] == "train"]
        normal = sorted([row for row in current if row["label"] == "normal"], key=lambda r: r["image"])
        anomaly = sorted([row for row in current if row["label"] == "anomaly"], key=lambda r: r["image"])
        rng.shuffle(normal)
        reference = normal.pop()
        references[category] = reference["image"]
        if len(normal) < len(anomaly) or not anomaly:
            raise ValueError("Missing balanced official training examples")
        for row in anomaly + normal[:len(anomaly)]:
            selected.append({**row, "partition": "validation" if category in validation else "train"})
    return selected, references


def download(url, wanted, root):
    missing = {name for name in wanted if not (root / name).exists()}
    if not missing:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        print(json.dumps({"stage": "archive", "content_length": response.headers.get("Content-Length"),
                          "selected_files": len(missing)}), flush=True)
        with tarfile.open(fileobj=response, mode="r|*") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or ".." in parts:
                    continue
                name = "/".join(parts[-5:])
                if name not in missing:
                    continue
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.extractfile(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if temporary.stat().st_size != member.size:
                    raise ValueError("Incomplete archive member")
                temporary.replace(target)
                missing.remove(name)
                if len(missing) % 200 == 0:
                    print(json.dumps({"stage": "extract", "remaining_files": len(missing)}), flush=True)
                if not missing:
                    break
        if missing:
            raise ValueError("Archive missing selected files: " + repr(sorted(missing)[:5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.metadata / "source.json").read_text())
    csv_path = args.metadata / "2cls_highshot.csv"
    selected, references = select(csv_path)
    if any(row["split"] != "train" for row in selected):
        raise ValueError("Official held-out rows cannot enter the training pool")
    url = provenance["archive_url"]
    wanted = {row["image"] for row in selected} | set(references.values())
    wanted.update(row["mask"] for row in selected if row["mask"])
    args.assets.mkdir(parents=True, exist_ok=True)
    download(url, wanted, args.assets)
    payload = (args.benchmark / "cases.json").read_bytes()
    cases = json.loads(payload)
    paths = {Path(case["asset_root"]) / evidence["path"] for case in cases for evidence in case["evidence"]}
    evaluation = {sha(path) for path in paths}
    hashes = {"train": set(), "validation": set()}
    ref_hashes, image_info = {}, {}
    for category, relative in references.items():
        path = args.assets / relative
        ref_hashes[category] = sha(path)
        with Image.open(path) as image:
            image.verify()
    for row in selected:
        path = args.assets / row["image"]
        checksum = sha(path)
        if checksum in evaluation or ref_hashes[row["object"]] in evaluation:
            raise ValueError("Frozen benchmark image overlap")
        with Image.open(path) as image:
            size = image.size
            image.verify()
        bbox = None
        if row["label"] == "anomaly":
            with Image.open(args.assets / row["mask"]) as mask:
                if mask.size != size:
                    raise ValueError("Source mask/image size mismatch")
                bbox = mask.convert("L").point(lambda value: 255 if value else 0).getbbox()
            if not bbox:
                raise ValueError("Anomaly annotation has an empty source mask")
        hashes[row["partition"]].update((checksum, ref_hashes[row["object"]]))
        image_info[row["image"]] = {"sha256": checksum, "size": list(size),
                                   "anomaly_bbox": list(bbox) if bbox else None}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Train/validation source pixels overlap")
    counts = Counter((row["partition"], row["label"]) for row in selected)
    report = {
        "scope": "Auxiliary condition-training source prepared; no SFT or sufficiency result is claimed",
        "provenance": provenance, "official_setup": "2cls_highshot", "official_split": "train",
        "csv_sha256": sha(csv_path), "eval_dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "category_partitions": {split: sorted({row["object"] for row in selected if row["partition"] == split})
                                for split in hashes},
        "counts": {split: {label: counts[(split, label)] for label in ("normal", "anomaly")} for split in hashes},
        "images": [{**row, **image_info[row["image"]]} for row in selected],
        "references": {category: {"image": relative, "sha256": ref_hashes[category]}
                       for category, relative in references.items()},
        "checks": {"official_test_rows_used": 0, "eval_image_hash_overlap": 0,
                   "train_validation_image_hash_overlap": 0},
    }
    (args.metadata / "source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "counts": report["counts"], "checks": report["checks"]}), flush=True)


if __name__ == "__main__":
    main()
