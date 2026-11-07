"""Copy frozen procedural evaluation rows into a fresh preflighted cohort/shard.

This CPU-only operation never retries a failed episode, loads model weights, or
changes the source run. It preserves SQLite text and referenced image bytes.
"""
import argparse
from contextlib import closing
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from resolveai.benchmark import source_digest
from resolveai.multistep import evaluate
from scripts.evaluate_multistep import (
    POLICIES, asset_fingerprint, canonical, load_data, model_fingerprint,
    sha_file, sha_value, validate_saved,
)

ALLOWED_CONFIG_CHANGES = {
    "dataset", "source", "dataset_sha256", "assets", "families", "case_ids", "shard", "shards",
}
SCHEMA = ("case_id", "policy", "result", "trace")


def plain(path, *, directory=False):
    """Reject symlinks, including directory components, before following paths."""
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlinks are not allowed: {path}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing {'directory' if directory else 'file'}: {path}")
    return path


def read_json(path):
    return json.loads(plain(path).read_text())


def readonly(path):
    return sqlite3.connect(plain(path).as_uri() + "?mode=ro", uri=True)


def cohort(config):
    source, all_cases = load_data(plain(config["dataset"]))
    if sha_file(source) != config["dataset_sha256"]:
        raise ValueError("Dataset checksum changed")
    manifest_path = source.parent / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("dataset_sha256") != config["dataset_sha256"]
            or manifest.get("annotation_status") != "procedural"):
        raise ValueError("Dataset manifest checksum/scope mismatch")
    if config.get("variants") != ["Obtainable"] or tuple(config.get("policies", [])) != POLICIES:
        raise ValueError("Only the frozen Obtainable five-policy track can be reused")
    eligible = [c for c in all_cases if c.get("variant", "Obtainable") == "Obtainable"]
    shard, shards = config["shard"], config["shards"]
    if type(shard) is not int or type(shards) is not int or not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    families = sorted({c["family_id"] for c in eligible})[shard::shards]
    selected = sorted([c for c in eligible if c["family_id"] in families],
                      key=lambda c: (c["family_id"], c["case_id"]))
    if config["families"] != families or config["case_ids"] != [c["case_id"] for c in selected]:
        raise ValueError("Manifest shard does not match its frozen dataset")
    # Check the unhashed path before the native helper resolves it.
    for case in all_cases:
        root = plain(case.get("asset_root", source.parent), directory=True)
        for item in case["evidence"]:
            path = plain(root / item["path"])
            if not path.is_relative_to(root):
                raise ValueError("Asset escapes its case root")
    if asset_fingerprint(all_cases, source.parent) != config["assets"]:
        raise ValueError("Frozen asset bytes changed")
    return source, all_cases, {c["case_id"]: c for c in selected}, manifest_path


def database_rows(db):
    if tuple(row[1] for row in db.execute("PRAGMA table_info(episodes)")) != SCHEMA:
        raise ValueError("Unexpected episodes database schema")
    rows = list(db.execute("SELECT case_id,policy,result,trace FROM episodes ORDER BY case_id,policy"))
    if len({(r[0], r[1]) for r in rows}) != len(rows):
        raise ValueError("Duplicate episode keys")
    return rows


def image_refs(trace):
    refs = set()
    for message in [*trace["messages"], *trace.get("acquisition_messages", [])]:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") == "image":
                ref = Path(part["image_ref"])
                if len(ref.parts) != 2 or ref.parts[0] != "images" or ref.suffix != ".png":
                    raise ValueError("Only relative content-addressed image refs are accepted")
                refs.add(ref)
    return refs


def check_current_source(config):
    native_script = Path(__file__).with_name("evaluate_multistep.py")
    script = (Path(__file__).with_name("evaluate_multistep_choice.py")
              if config["experiment"] == "procedural-multistep-interface-assisted-choice-v1"
              else native_script)
    if (config["source_sha256"] != source_digest()
            or config["script_sha256"] != sha_file(script)
            or ("shared_native_script_sha256" in config
                and config["shared_native_script_sha256"] != sha_file(native_script))):
        raise ValueError("Current scorer/source or evaluation runner differs from the frozen run")
    model = config["model"]
    base = plain(model["base_path"], directory=True)
    for p in [*base.glob("*.json"), *base.glob("*.safetensors")]:
        plain(p)
    adapter = model.get("adapter", {}).get("path")
    if adapter:
        saved = plain(adapter, directory=True)
        for p in [*saved.glob("*.json"), saved / "adapter_model.safetensors"]:
            plain(p)
    if model_fingerprint(base, Path(adapter) if adapter else None) != model:
        raise ValueError("Current model/processor/adapter fingerprint changed")


def reuse(source, target, *, allow_partial_source=False):
    source, target = plain(source, directory=True), plain(target, directory=True)
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and target runs must be separate directory trees")
    # Shared source lock does not mutate its bytes and excludes a live worker.
    with plain(source / "run.lock").open("rb") as source_lock, plain(target / "run.lock").open("r+b") as target_lock:
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(target_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_manifest_path, target_manifest_path = source / "manifest.json", target / "manifest.json"
        old, new = read_json(source_manifest_path), read_json(target_manifest_path)
        a, b = old["configuration"], new["configuration"]
        if (a.get("experiment") not in {"procedural-multistep-two-request-v2",
                                       "procedural-multistep-interface-assisted-choice-v1"}
                or {k: v for k, v in a.items() if k not in ALLOWED_CONFIG_CHANGES}
                != {k: v for k, v in b.items() if k not in ALLOWED_CONFIG_CHANGES}):
            raise ValueError("Model, interface, source or inference settings changed")
        if old.get("versions") != new.get("versions"):
            raise ValueError("Runtime dependency versions changed")
        check_current_source(a)
        old_data, old_cases, old_selected, old_data_manifest = cohort(a)
        new_data, new_cases, new_selected, new_data_manifest = cohort(b)
        old_index = {c["case_id"]: c for c in old_cases}
        new_index = {c["case_id"]: c for c in new_cases}
        if allow_partial_source and (a["dataset_sha256"] != b["dataset_sha256"]
                                     or set(old_index) != set(new_index)):
            raise ValueError("Partial-source reuse permits only the same complete dataset, repartitioned")
        if any(k not in new_index or canonical(c) != canonical(new_index[k])
               for k, c in old_index.items()):
            raise ValueError("The expanded dataset changed an existing case")
        if any(b["assets"].get(k) != v for k, v in a["assets"].items()):
            raise ValueError("Expanded assets are not an unchanged superset")
        summary = read_json(target / "summary.json")
        if summary.get("preflight") is not True or summary.get("model_loaded") is not False:
            raise ValueError("Target must first pass native CPU preflight without loading a model")
        if (target / "reuse_origin.json").exists():
            raise ValueError("Target already has a reuse audit; use its normal runner to resume")
        allowed = {"run.lock", "manifest.json", "episodes.sqlite", "summary.json"}
        if {p.name for p in target.iterdir()} - allowed:
            raise ValueError("Target must be a fresh preflight-only output directory")
        source_db_path, target_db_path = source / "episodes.sqlite", target / "episodes.sqlite"
        with closing(readonly(source_db_path)) as src, closing(sqlite3.connect(plain(target_db_path))) as dst:
            src.execute("BEGIN")
            rows = database_rows(src)
            for case_id, policy, raw, trace_raw in rows:
                row, trace = json.loads(raw), json.loads(trace_raw)
                if sha_value(trace) != row.get("trace_sha256"):
                    raise ValueError("A saved trace changed")
                if (case_id not in old_selected
                        or row.get("family_id") != old_selected[case_id]["family_id"]
                        or trace.get("multistep_evaluation") != row.get("multistep_evaluation")):
                    raise ValueError("Source row family or trace/result scores disagree")
                for ref in image_refs(trace):
                    plain(source / ref)
            source_expected = {(k, p) for k in old_selected for p in POLICIES}
            source_seen, _ = validate_saved(src, source_expected, old_selected, source, evaluate)
            source_complete = source_seen == source_expected
            if not source_complete and not allow_partial_source:
                raise ValueError("Source pilot is incomplete; use explicit --allow-partial-source only for same-dataset repartitioning")
            if database_rows(dst):
                raise ValueError("Target episodes table must be empty")
            selected = [r for r in rows if r[0] in new_selected]
            if not selected:
                raise ValueError("This target shard has no source-pilot rows")
            refs, row_audit = {}, []
            for case_id, policy, raw, trace_raw in selected:
                row, trace = json.loads(raw), json.loads(trace_raw)
                if row.get("family_id") != new_selected[case_id]["family_id"]:
                    raise ValueError("Stored result has a different family ID")
                if trace.get("multistep_evaluation") != row["multistep_evaluation"]:
                    raise ValueError("Stored trace/result scores disagree")
                for ref in image_refs(trace):
                    path = plain(source / ref)
                    if sha_file(path) != ref.stem:
                        raise ValueError("Source image blob changed")
                    refs[str(ref)] = {"sha256": ref.stem, "bytes": path.stat().st_size}
                row_audit.append({
                    "case_id": case_id, "policy": policy,
                    "case_sha256": sha_value(new_selected[case_id]),
                    "result_text_sha256": sha256(raw.encode()).hexdigest(),
                    "trace_text_sha256": sha256(trace_raw.encode()).hexdigest(),
                    "trace_sha256": row["trace_sha256"],
                })
            audit = {
                "schema_version": "resolveai-native-cohort-reuse-v1",
                "source_run": str(source), "target_run": str(target),
                "source_manifest_sha256": sha_file(source_manifest_path),
                "target_manifest_sha256": sha_file(target_manifest_path),
                "source_dataset_sha256": sha_file(old_data),
                "target_dataset_sha256": sha_file(new_data),
                "source_dataset_manifest_sha256": sha_file(old_data_manifest),
                "target_dataset_manifest_sha256": sha_file(new_data_manifest),
                "source_database_sha256": sha_file(source_db_path),
                "source_ordered_sql_sha256": sha_value(rows),
                "target_preflight_summary_sha256": sha_file(target / "summary.json"),
                "source_case_sha256": {k: sha_value(c) for k, c in sorted(old_index.items())},
                "source_complete": source_complete, "allow_partial_source": allow_partial_source,
                "source_rows_expected": len(source_expected), "source_rows_actual": len(rows),
                "source_complete_episodes": len(rows) if source_complete else None,
                "copied_episodes": len(selected),
                "target_rows_expected": len(new_selected) * len(POLICIES),
                "interface_experiment": a["experiment"],
                "all_source_cases_identical": True, "failures_preserved": True,
                "raw_sql_text_preserved": True, "model_loaded": False,
                "base_weight_verification": "Frozen native fingerprint: shard sizes; full base-weight SHA was not recomputed.",
                "reuse_script_sha256": sha_file(__file__), "rows": row_audit, "blobs": refs,
            }
            made_images = False
            copied = []
            audit_temp = None
            try:
                dst.execute("BEGIN IMMEDIATE")
                if database_rows(dst):
                    raise ValueError("Target is no longer empty")
                if refs:
                    (target / "images").mkdir()
                    made_images = True
                for ref, info in sorted(refs.items()):
                    dest = target / ref
                    with (source / ref).open("rb") as inp, dest.open("xb") as out:
                        copied.append(dest)
                        shutil.copyfileobj(inp, out)
                    if sha_file(dest) != info["sha256"]:
                        raise ValueError("Copied image bytes changed")
                dst.executemany("INSERT INTO episodes VALUES (?,?,?,?)", selected)
                expected = {(k, p) for k in new_selected for p in POLICIES}
                seen, _ = validate_saved(dst, expected, new_selected, target, evaluate)
                if seen != {(r[0], r[1]) for r in selected}:
                    raise ValueError("Target validation did not reproduce every imported episode")
                # Audit is prepared before commit, and installed only after successful validation.
                fd, name = tempfile.mkstemp(prefix=".reuse-origin-", dir=target)
                audit_temp = Path(name)
                with os.fdopen(fd, "w") as out:
                    out.write(json.dumps(audit, indent=2) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(audit_temp, target / "reuse_origin.json")
                audit_temp = None
                dst.commit()
            except BaseException:
                dst.rollback()
                (target / "reuse_origin.json").unlink(missing_ok=True)
                if audit_temp:
                    audit_temp.unlink(missing_ok=True)
                for path in copied:
                    path.unlink(missing_ok=True)
                if made_images:
                    (target / "images").rmdir()
                raise
        return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen source run directory; must be complete unless explicitly allowed")
    parser.add_argument("--target", type=Path, required=True, help="Fresh CPU-preflighted full-cohort shard directory")
    parser.add_argument("--allow-partial-source", action="store_true",
                        help="Allow only same-dataset repartitioning; preserve every existing row, including failures")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run CPU-only inside an existing GPU-node allocation")
    result = reuse(args.source, args.target, allow_partial_source=args.allow_partial_source)
    print(json.dumps({k: result[k] for k in ("copied_episodes", "source_complete", "source_rows_actual", "source_rows_expected", "model_loaded")}))


if __name__ == "__main__":
    main()
