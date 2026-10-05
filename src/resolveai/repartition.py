"""Repartition stopped, unsharded evaluations without regenerating completed episodes."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _image_refs(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "image_ref":
                yield child
            else:
                yield from _image_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_refs(child)


def repartition(source, data, destination, shards=2, name=None):
    """Stop all source writers first. Preserve results, matching traces and pixel blobs."""
    source, data, destination = map(Path, (source, data, destination))
    if shards < 2:
        raise ValueError("at least two shards are required")
    files = [source / name for name in ("manifest.json", "results.jsonl", "trajectories.jsonl")]
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    manifest = json.loads(files[0].read_text())
    config = manifest["configuration"]
    raw = (data / "cases.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["dataset_sha256"]:
        raise ValueError("dataset differs from source evaluation")
    if config["shards"] != 1 or config.get("limit_families"):
        raise ValueError("source must be an unsharded full-pool evaluation")
    cases = {c["case_id"]: c for c in json.loads(raw)}
    families = sorted({c["family_id"] for c in cases.values()})
    if len(families) < shards or len(families) != config["selected_families"]:
        raise ValueError("source family count differs from dataset")
    assignment = {family: index % shards for index, family in enumerate(families)}
    rows = {}
    with files[1].open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["case_id"], row["policy"]
            case = cases.get(key[0])
            if (key in rows or case is None or key[1] not in config["policies"]
                    or row["family_id"] != case["family_id"]):
                raise ValueError("duplicate or unexpected source episode")
            rows[key] = row
    traces = {}
    with files[2].open() as stream:
        for line in stream:
            record = json.loads(line)
            metadata = record["metadata"]
            key = metadata["case_id"], metadata["policy"]
            row = rows.get(key)
            if (row is not None and record["decision"] == row["decision"]
                    and metadata["termination"] == row["termination"]
                    and record["wall_latency_s"] == row["wall_latency_s"]):
                traces.setdefault(key, record)
    if set(traces) != set(rows):
        raise ValueError("a completed episode has no matching trajectory")
    destination.mkdir(parents=True, exist_ok=True)
    name = source.name if name is None else name
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a directory basename")
    targets = [destination / (name + "-shard" + str(i)) for i in range(shards)]
    if any(p.exists() for p in targets):
        raise ValueError("destination shards already exist")
    counts = []
    with tempfile.TemporaryDirectory(prefix=".repartition-", dir=destination) as name:
        stage = Path(name)
        for index, target in enumerate(targets):
            folder = stage / target.name
            (folder / "images").mkdir(parents=True)
            selected = [key for key, row in rows.items() if assignment[row["family_id"]] == index]
            seeded = deepcopy(manifest)
            seeded["configuration"].update(shards=shards, shard_index=index,
                selected_families=sum(value == index for value in assignment.values()))
            seeded["slurm_job"] = None
            seeded["seed_execution"] = {"source_directory": str(source),
                "source_manifest": manifest, "seeded_episodes": len(selected)}
            (folder / "manifest.json").write_text(json.dumps(seeded, indent=2) + "\n")
            with (folder / "results.jsonl").open("w") as result, (folder / "trajectories.jsonl").open("w") as trace:
                for key in selected:
                    result.write(json.dumps(rows[key]) + "\n")
                    trace.write(json.dumps(traces[key]) + "\n")
                    for reference in _image_refs(traces[key]):
                        pixel = (source / reference).resolve()
                        if not pixel.is_relative_to((source / "images").resolve()):
                            raise ValueError("image reference escapes source pixel directory")
                        output = folder / "images" / pixel.name
                        if not output.exists():
                            os.link(pixel, output)
            counts.append(len(selected))
        if stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]:
            raise ValueError("source changed during repartition; stop its writers")
        for target in targets:
            (stage / target.name).rename(target)
    return {"source_episodes": len(rows), "shard_episodes": counts,
            "paths": [str(p) for p in targets]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--name")
    args = parser.parse_args()
    print(json.dumps(repartition(args.source, args.data, args.destination, args.shards, args.name)))


if __name__ == "__main__":
    main()
