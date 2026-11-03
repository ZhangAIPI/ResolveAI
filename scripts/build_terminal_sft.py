"""Keep terminal decision targets from an audited SFT pack for an action-target ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal(row):
    calls = row["target"].get("tool_calls", [])
    return len(calls) == 1 and calls[0]["function"]["name"] == "finish"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    manifest = json.loads((args.source / "manifest.json").read_text())
    selected, stats = {}, {}
    for split, checksum in manifest["files"].items():
        path = args.source / (split + ".jsonl")
        if sha(path) != checksum:
            raise ValueError("Audited source decisions changed")
        rows = [json.loads(line) for line in path.open()]
        selected[split] = [row for row in rows if terminal(row)]
        if not selected[split]:
            raise ValueError("No terminal targets")
        stats[split] = {"source_decisions": len(rows), "decision_turns": len(selected[split]),
                       "by_task": dict(Counter(row["group"].get("task", "identity")
                                              for row in selected[split]))}
    args.output.mkdir(parents=True)
    for split, rows in selected.items():
        with (args.output / (split + ".jsonl")).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "scope": "Terminal-target-only ablation; same audited public histories, no request-action targets",
        "adaptation_name": args.name, "eval_dataset_sha256": manifest["eval_dataset_sha256"],
        "source_manifest_sha256": sha(args.source / "manifest.json"), "source_manifest": manifest,
        "counts": stats, "files": {split: sha(args.output / (split + ".jsonl")) for split in selected},
        "limits": "Same 200-update budget, different target-token composition; not equal token compute",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "name": args.name, "counts": stats}), flush=True)


if __name__ == "__main__":
    main()
