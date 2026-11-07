"""Export the frozen procedural cohort without machine-specific asset roots."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

PREFIX = "resolveai-multistep-search-v1"
PROVENANCE_FILES = (
    "scripts/export_multistep_release.py", "scripts/build_multistep_cases.py",
    "scripts/evaluate_multistep.py", "scripts/analyze_multistep.py",
    "src/resolveai/render_multistep.py", "src/resolveai/multistep.py",
    "src/resolveai/environment.py", "src/resolveai/world.py",
    "src/resolveai/grounding.py", "pyproject.toml",
)
README = """# ResolveAI: procedural two-request search
This release contains 66 scene banks, 132 families and 528 cases (four access
versions per family). It is a controlled renderer-specific suite, not real
photographs or independent human evidence adjudication.

## Files and checksums
Run `sha256sum -c SHA256SUMS` from this directory. RGB, first-hit visibility
masks and bank manifests are byte-identical to the source freeze. The only case
field removed is the machine-specific `asset_root`; JSON uses compact encoding.
Consequently `dataset_manifest.json` records a new portable case SHA256 and the
original frozen SHA256 separately. Labels, regions, identity links, encoded
diagnostic masks and finite-world certificates are unchanged.
`source_dataset_manifest.json` and `source_freeze_audit.json` describe the
original freeze, not the normalized archive's file times. Tar owner and time
metadata are normalized; original pilot preservation applies to the source.

## Development and held-out use
`splits.json` defines the first eight banks / 16 development families and the
remaining 58 banks / 116 held-out families. Both matched families and every
access version stay in one split. Do not tune on held-out examples or select
cases based on model scores. Bootstrap whole scene banks.
Only Obtainable is used for the controlled five-policy, two-request comparison.
Sufficient starts with the identity and condition photographs. Missing and
Unavailable prevent access to the identity survey and require abstention.

## Loading and scoring
Clone https://github.com/ZhangAIPI/ResolveAI and check out the repository commit
in `source_provenance.json`, then install the package with `python -m pip install
-e .`. Source file SHA256 values identify the exact builder, renderer and scorer.
From the repository, with RELEASE pointing to this extracted directory:

```python
import os
from pathlib import Path
from scripts.evaluate_multistep import load_data, NoOCR
from resolveai.environment import Environment
from resolveai.multistep import observation_token, evaluate
source, cases = load_data(Path(os.environ["RELEASE"]))
case = next(c for c in cases if c["variant"] == "Obtainable")
env = Environment(case, source.parent, budget=6, ocr_backend=NoOCR())
tokens = [observation_token(im, view=im["image_id"])
          for im in env.observation()["images"]]
assert evaluate(case, None, tokens)["certificate_valid"]
```

An agent receives only `env.observation()` and tool feedback. Keep cases, labels,
masks and certificates outside its process. Requests release recorded views and
do not mutate the physical scene. Position names refer to the front survey.
The native runner accepts this release directory via `--data`; it has no
machine-specific asset root to override the runner's data-directory fallback.

The exporter verifies all 528 cases after fresh extraction using the existing
loader, Environment and scorer. This uses private ground-truth decisions as
software fixtures; it is not an agent result. Native, interface-assisted choice
and renderer-specific image-program results must be reported separately.
"""


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or bool(PureWindowsPath(name).drive)
            or ".." in name.split("/") or "\\" in name or str(path) != name):
        raise ValueError("Unsafe archive-relative path")
    return path


def public_tree(value):
    if isinstance(value, dict):
        for key, item in value.items():
            public_tree(key)
            public_tree(item)
    elif isinstance(value, list):
        for item in value:
            public_tree(item)
    elif isinstance(value, str) and (
            value.startswith("/") or PureWindowsPath(value).is_absolute()):
        raise ValueError("Machine-specific absolute path in exported JSON")


def portable_cases(cases):
    result = []
    for case in cases:
        item = dict(case)
        item.pop("asset_root", None)
        public_tree(item)
        for evidence in item["evidence"]:
            safe_name(evidence["path"])
        result.append(item)
    return result


def checked_source(root, path):
    root, path = Path(root).resolve(strict=True), Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("Source path escapes its allowed root") from None
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink in source path")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Source must be a regular file inside its allowed root")
    return path


def write_archive(output, generated, assets, *, source_root=None):
    """Write only allowlisted regular files; normalize owner/time metadata."""
    names = set(generated) | set(assets)
    if len(names) != len(generated) + len(assets):
        raise ValueError("Duplicate archive path")
    for name in names:
        safe_name(name)
    if assets and source_root is None:
        raise ValueError("An explicit asset root is required")
    for asset in assets.values():
        checked_source(source_root, asset)
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|",
                              format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(names):
                    info = tarfile.TarInfo(PREFIX + "/" + name)
                    info.mode, info.mtime, info.uid, info.gid = 0o644, 0, 0, 0
                    if name in generated:
                        data = generated[name]
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
                    else:
                        path = Path(assets[name])
                        if path.is_symlink() or not path.is_file():
                            raise ValueError("Assets must be regular files")
                        info.size = path.stat().st_size
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)


def extract_checked(archive, destination):
    """Extract regular members without tar path/link traversal."""
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            safe_name(member.name)
            if not member.isfile() or not member.name.startswith(PREFIX + "/"):
                raise ValueError("Unexpected archive member")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return destination / PREFIX


def verify_files(root):
    root = Path(root)
    manifest = json.loads((root / "release_manifest.json").read_text())
    expected = set(manifest["files"]) | {"release_manifest.json", "SHA256SUMS"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Release file set differs from manifest")
    for name, record in manifest["files"].items():
        safe_name(name)
        path = root / name
        if path.is_symlink() or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Release file checksum mismatch")
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        safe_name(name)
        if sha(root / name) != digest:
            raise ValueError("SHA256SUMS mismatch")
    return manifest


def verify_scoring(root, repo):
    """Private-answer integration fixtures, not measured agent performance."""
    sys.path.insert(0, str(repo))
    from scripts.evaluate_multistep import load_data, NoOCR, public_queries
    from scripts.analyze_multistep import load_dataset
    from resolveai.environment import Environment
    from resolveai.multistep import evaluate, observation_token
    source, cases = load_data(root)
    load_dataset(root)  # Existing analyzer also checks the portable manifest.
    counts = Counter()
    for case in cases:
        if "asset_root" in case:
            raise ValueError("Portable cases must use the data-directory fallback")
        env = Environment(case, source.parent, 6, ocr_backend=NoOCR())
        tokens = [observation_token(im, view=im["image_id"])
                  for im in env.observation()["images"]]
        initial = evaluate(case, None, tokens)
        if not initial["certificate_valid"]:
            raise ValueError("A relocated initial certificate failed")
        if case["variant"] != "Sufficient":
            queries = public_queries(env.observation())
            for view in ("front_survey", case["multistep_certificate"]["target_rear_view"]):
                response = env.step({"type": "request_photo", "query": queries[view]})
                tokens.extend(observation_token(im, view=im["image_id"])
                              for im in response.get("images", []))
        if case["variant"] in ("Missing", "Unavailable"):
            decision = {"verdict": "Need more evidence", "citations": [], "links": []}
        else:
            regions = case["multistep_regions"]
            decision = {
                "verdict": case["provenance_truth"],
                "citations": [deepcopy(regions[k]) for k in
                              ("reference", "front_target", "rear_target", "diagnostic")],
                "links": deepcopy(case["multistep_certificate"]["required_links"]),
            }
        response = env.step({"type": "finish", **decision})
        score = evaluate(case, env.decision, tokens)
        if (response.get("error") or not env.finished or not score["certificate_valid"]
                or not score["simulator_grounded_correct"]):
            raise ValueError("A relocated scoring fixture failed: " + case["case_id"])
        counts[case["variant"]] += 1
    return {"cases_scored": len(cases), "by_variant": dict(counts),
            "all_private_answer_fixtures_passed": True,
            "scope": "Software relocation/scoring checks; not agent performance"}


def prepare(source, repo):
    data = source / "benchmark"
    for name in ("benchmark/cases.json", "benchmark/dataset_manifest.json",
                 "benchmark/construction_audit.json", "full-bank-audit.json"):
        checked_source(source, source / name)
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    original_sha = sha(data / "cases.json")
    if manifest["dataset_sha256"] != original_sha:
        raise ValueError("Frozen source cases checksum mismatch")
    original = json.loads((data / "cases.json").read_text())
    cases = portable_cases(original)
    if (len(cases) != 528 or len({c["family_id"] for c in cases}) != 132
            or Counter(c["variant"] for c in cases) !=
            {v: 132 for v in ("Sufficient", "Obtainable", "Missing", "Unavailable")}):
        raise ValueError("Expected the complete frozen 132-family cohort")
    assets, seeds = {}, []
    for relative, digest in sorted(manifest["bank_manifests"].items()):
        safe_name(relative)
        path = checked_source(source, source / relative)
        if sha(path) != digest:
            raise ValueError("Source bank manifest checksum mismatch")
        raw = json.loads(path.read_text())
        public_tree(raw)
        if relative != f"banks/seed-{raw['seed']}/bank.json":
            raise ValueError("Unexpected source bank-manifest path")
        seeds.append(raw["seed"])
        assets[relative] = path
        for row in raw["images"].values():
            for local, expected in ((row["path"], row["sha256"]),
                                    (row["visibility_checks"]["mask_path"],
                                     row["visibility_checks"]["mask_sha256"])):
                safe_name(local)
                asset = checked_source(source, path.parent / local)
                name = str(asset.relative_to(source))
                safe_name(name)
                if asset.is_symlink() or sha(asset) != expected:
                    raise ValueError("Source RGB/mask checksum mismatch")
                assets[name] = asset
    if seeds != list(range(71000, 71066)) or len(assets) != 1914:
        raise ValueError("Incomplete or unexpected scene-bank assets")
    payload = encoded(cases)
    portable_manifest = {**manifest,
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "source_dataset_sha256": original_sha,
        "path_normalization": "Removed only asset_root; compact JSON; all other case values unchanged"}
    audit = json.loads((data / "construction_audit.json").read_text())
    audit["manifest"] = portable_manifest
    splits = {}
    for label, lower, upper in (("development", 71000, 71007), ("heldout", 71008, 71065)):
        selected = [c for c in cases if lower <= int(c["evidence"][0]["path"].split("/")[1][5:]) <= upper]
        splits[label] = {"scene_seeds": list(range(lower, upper + 1)),
            "family_ids": sorted({c["family_id"] for c in selected}),
            "case_ids": [c["case_id"] for c in selected],
            "families": len(selected) // 4, "cases": len(selected)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    for name in PROVENANCE_FILES:
        committed = subprocess.check_output(["git", "show", commit + ":" + name], cwd=repo)
        if hashlib.sha256(committed).hexdigest() != sha(checked_source(repo, repo / name)):
            raise ValueError("Commit/freeze source files before exporting: " + name)
    provenance = {"repository": "https://github.com/ZhangAIPI/ResolveAI",
        "repository_commit": commit, "source_dataset_sha256": original_sha,
        "source_manifest_sha256": sha(data / "dataset_manifest.json"),
        "source_construction_audit_sha256": sha(data / "construction_audit.json"),
        "exporter_transform": portable_manifest["path_normalization"],
        "source_files": {name: sha(repo / name) for name in PROVENANCE_FILES},
        "archive_metadata": "Owner fields empty/zero, timestamps zero; no host paths"}
    generated = {
        "cases.json": payload, "dataset_manifest.json": encoded(portable_manifest),
        "construction_audit.json": encoded(audit), "splits.json": encoded(splits),
        "source_dataset_manifest.json": encoded(manifest),
        "source_freeze_audit.json": (source / "full-bank-audit.json").read_bytes(),
        "source_provenance.json": encoded(provenance), "README.md": README.encode(),
    }
    for name, value in generated.items():
        if name.endswith(".json"):
            public_tree(json.loads(value))
    index = {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
             for name, value in generated.items()}
    index.update({name: {"sha256": sha(path), "bytes": path.stat().st_size}
                  for name, path in assets.items()})
    generated["release_manifest.json"] = encoded({
        "schema_version": "resolveai-portable-multistep-v1",
        "families": 132, "cases": 528, "banks": 66, "RGB_images": 924,
        "mask_images": 924, "source_dataset_sha256": original_sha,
        "portable_dataset_sha256": portable_manifest["dataset_sha256"], "files": index})
    checks = {name: record["sha256"] for name, record in index.items()}
    checks["release_manifest.json"] = hashlib.sha256(generated["release_manifest.json"]).hexdigest()
    generated["SHA256SUMS"] = "".join(digest + "  " + name + "\n"
                                    for name, digest in sorted(checks.items())).encode()
    return generated, assets, original_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Frozen root containing benchmark/ and banks/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Output must be a new archive path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated, assets, original_sha = prepare(args.source.resolve(), args.repo.resolve())
    write_archive(args.output, generated, assets, source_root=args.source.resolve())
    # This context removes only its own newly-created verification directory.
    with tempfile.TemporaryDirectory(prefix="resolveai-release-check-",
                                     dir=args.output.parent) as temporary:
        root = extract_checked(args.output, temporary)
        verify_files(root)
        report = verify_scoring(root, args.repo.resolve())
    if sha(args.source / "benchmark/cases.json") != original_sha:
        raise ValueError("Frozen source changed during export")
    report.update(archive=args.output.name, archive_bytes=args.output.stat().st_size,
                  archive_sha256=sha(args.output), source_cases_sha256=original_sha,
                  portable_cases_sha256=json.loads(generated["dataset_manifest.json"])["dataset_sha256"],
                  source_cases_unchanged=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
