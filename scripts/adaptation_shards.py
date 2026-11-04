"""Evaluate disjoint family shards; publish a read-only index after all are complete."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys


def finalize(output, plan_path, count):
    plan = json.loads(plan_path.read_text())
    payload = (Path(plan["data"]) / "cases.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    cases = {c["case_id"]:c for c in json.loads(payload) if c["family_id"] in plan["family_ids"]}
    manifests, entries, seen = [], [], set()
    for shard in range(count):
        directory = output / ("shard-" + str(shard))
        raw = (directory / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        config = manifest["configuration"]
        families = [f for i,f in enumerate(plan["family_ids"]) if i % count == shard]
        if (config["shards"],config["shard"],config["families"]) != (count,shard,families):
            raise ValueError("Unexpected shard assignment")
        common = {k:v for k,v in config.items() if k not in {"families","cases","shard"}}
        if manifests:
            first = {k:v for k,v in manifests[0]["configuration"].items() if k not in {"families","cases","shard"}}
            if common != first or manifest["worker"]["gpu"] != manifests[0]["worker"]["gpu"]:
                raise ValueError("Shard policies or GPU types differ")
        expected = {(cid,arm) for cid,c in cases.items() if c["family_id"] in families for arm in config["arms"]}
        database = directory / "episodes.sqlite"
        digest = hashlib.sha256()
        actual = set()
        with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)) as db:
            for cid,arm,result,trace in db.execute("SELECT * FROM episodes ORDER BY case_id,arm"):
                row = json.loads(result)
                if cid != row["case_id"] or arm != row["arm"] or row["family_id"] != cases[cid]["family_id"]:
                    raise ValueError("Shard row identity differs")
                actual.add((cid,arm))
                digest.update(json.dumps([cid,arm,result,trace]).encode())
        if actual != expected or seen & actual:
            raise ValueError("Incomplete or overlapping evaluation shards")
        seen.update(actual)
        entries.append({"directory":directory.name,"manifest_sha256":hashlib.sha256(raw).hexdigest(),
                        "episodes":len(actual),"rows_sha256":digest.hexdigest()})
        manifests.append(manifest)
    aggregated = json.loads(json.dumps(manifests[0]))
    aggregated["configuration"].update(shard=None, families=plan["family_ids"], cases=len(cases))
    aggregated["aggregation"] = {"type":"disjoint family shards; original rows and timings retained","shards":entries}
    # No merged copy of traces/images: all scoring reads the original shard databases.
    (output / "manifest.json").write_text(json.dumps(aggregated,indent=2)+"\n")
    (output / "shard_index.json").write_text(json.dumps({"shards":entries},indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--model-spec", type=Path)
    p.add_argument("--grounding-root", type=Path)
    p.add_argument("--priority-cohort", type=Path)
    args = p.parse_args()
    output = args.run / "benchmark"
    if args.shard is None:
        finalize(output, args.plan, args.shards)
        return
    complete = json.loads((args.run / "complete.json").read_text())
    if hashlib.sha256((Path(complete["adapter"]) / "adapter_model.safetensors").read_bytes()).hexdigest() != complete["adapter_sha256"]:
        raise ValueError("RL adapter changed")
    spec = json.loads(args.model_spec.read_text())
    command = [sys.executable,"-u","-m","scripts.evaluate_study",str(args.plan),
               complete["adapter"],str(output / ("shard-"+str(args.shard))),
               "--model-id",spec["model_id"]+"+ResolveAI-GRPO","--revision",complete["adapter_sha256"],
               "--shards",str(args.shards),"--shard",str(args.shard),"--grounding-root",str(args.grounding_root)]
    if args.priority_cohort:
        command += ["--priority-cohort",str(args.priority_cohort)]
    subprocess.run(command,check=True)


if __name__ == "__main__":
    main()
