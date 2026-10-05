"""Build a pinned, small MVTec pilot using only the official test split.

This is controlled resolution release, not new-camera photography. Preview
quality is changed uniformly, without using defect masks or truth labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from PIL import Image

REPO = "Voxel51/mvtec-ad"
REVISION = "30a183a3b96e3aef953f230784b123b719b09d97"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")


def fetch(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def prepare(root, per_label=4, preview_size=48):
    if per_label < 1 or preview_size < 1:
        raise ValueError("sample count and preview size must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = fetch(BASE + "/samples.json", root / "mvtec_samples.json")
    rows = json.loads(metadata_path.read_text())["samples"]
    selected = []
    for category in ("bottle", "hazelnut", "metal_nut"):
        for anomalous in (False, True):
            candidates = [r for r in rows if r["split"] == "test"
                          and r["category"]["label"] == category
                          and (r["defect"]["label"] != "good") == anomalous]
            candidates.sort(key=lambda r: hashlib.sha256(
                ("resolveai-poc-v1|" + r["_id"]["$oid"]).encode()).hexdigest())
            if len(candidates) < per_label:
                raise ValueError("insufficient test samples")
            selected.extend(candidates[:per_label])

    def build(row):
        uid = row["_id"]["$oid"]
        family = "mvtec-" + uid
        relative = Path("assets") / family
        original = fetch(BASE + "/" + row["filepath"], root / relative / "original.png")
        with Image.open(original) as image:
            image = image.convert("RGB")
            dimensions = list(image.size)
            image.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)
            image.save(root / relative / "preview.png")
        label = "Refuted" if row["defect"]["label"] == "good" else "Supported"
        category = row["category"]["label"].replace("_", " ")
        common = {"source_id": uid, "party": "public", "object": category,
                  "time": "capture", "available": True}
        full = {**common, "id": "original", "path": str(relative / "original.png"), "view": "original"}
        preview = {**common, "id": "preview", "path": str(relative / "preview.png"), "view": "overview"}
        provenance = {"repo": REPO, "revision": REVISION, "filepath": row["filepath"],
                      "sample_id": uid, "official_split": row["split"],
                      "category": row["category"]["label"], "defect_label": row["defect"]["label"],
                      "original_size": dimensions,
                      "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
        cases = []
        for variant in VARIANTS:
            evidence = [preview, dict(full)]
            if variant == "Missing":
                evidence = [preview]
            if variant == "Unavailable":
                evidence[1]["available"] = False
            cases.append({"case_id": family + "-" + variant,
                "family_id": family, "variant": variant,
                "claim": f"The photographed {category} has a visible defect.",
                "initial": ["original"] if variant == "Sufficient" else ["preview"],
                "evidence": evidence,
                "request_spec": {"object": category, "time": "capture", "view": "original"},
                "annotation": {"verdict": label, "visual_verdict": label,
                    "minimal_evidence_sets": {label: [["original"]]},
                    "protocol": "full-resolution-proxy-v1"},
                "provenance": provenance})
        return cases

    with ThreadPoolExecutor(max_workers=4) as executor:
        cases = [case for group in executor.map(build, selected) for case in group]
    (root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    manifest = {"dataset": REPO, "revision": REVISION,
        "official_source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license": "CC-BY-NC-SA-4.0", "families": len(selected), "cases": len(cases),
        "per_label_per_category": per_label, "preview_max_side": preview_size,
        "categories": ["bottle", "hazelnut", "metal_nut"],
        "sampling": "sha256(resolveai-poc-v1|sample_id) within category and binary label",
        "split": "official test; no training or policy tuning on this pilot",
        "evidence_rule": "Original image required as an operational quality proxy, not human-annotated sufficiency."}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--preview-size", type=int, default=48)
    args = parser.parse_args()
    prepare(args.root, args.per_label, args.preview_size)


if __name__ == "__main__":
    main()
