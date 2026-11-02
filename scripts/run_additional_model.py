"""Download a pinned public MLLM, verify inference, then run the frozen benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grounding-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("An allocated GPU node is required")
    spec = json.loads(args.spec.read_text())
    local = Path(spec["path"])
    print(json.dumps({"stage": "download", "model_id": spec["model_id"],
                      "revision": spec["revision"]}), flush=True)
    snapshot_download(
        spec["model_id"], revision=spec["revision"], local_dir=local,
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt"],
        max_workers=4,
    )
    if spec.get("processor_options"):
        (local / "resolveai_processor.json").write_text(json.dumps(spec["processor_options"], indent=2) + "\n")
    (local / "resolveai_revision.json").write_text(json.dumps(spec, indent=2) + "\n")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from resolveai.conversation import Conversation
    from resolveai.diagnostic import text_direct
    from resolveai.environment import Environment
    from resolveai.pilot import ModelClient
    from resolveai.rollout import run
    plan = json.loads(args.plan.read_text())
    cases = json.loads((Path(plan["data"]) / "cases.json").read_text())
    checks = []
    client = ModelClient(local, 0, "auto", assistant_prefix=True)
    try:
        for task in ["state", "identity"]:
            case = next(c for c in cases if c["task"] == task and c["variant"] == "Sufficient")
            env = Environment(case, case["asset_root"], plan["budget"])
            direct = text_direct(env, case, client)
            trace = run(Conversation(env), client, max_turns=2)
            checks.append({
                "task": task, "direct_termination": direct["termination"],
                "agent_termination": trace["metadata"]["termination"],
                "direct_generation": direct["generation"],
                "agent_generation": trace["generation"],
                "agent_turns": trace["metadata"]["turns"],
            })
            if trace["metadata"]["termination"] == "context_limit" and not trace["metadata"]["turns"]:
                raise ValueError("Initial tool prompt does not fit the frozen context budget")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "inference_preflight.json").write_text(json.dumps(
            {"worker": client.ready, "checks": checks}, indent=2) + "\n")
    finally:
        client.close()
    print(json.dumps({"stage": "benchmark", "model_id": spec["model_id"]}), flush=True)
    subprocess.run(
        [sys.executable, "-u", "-m", "scripts.evaluate_study",
         str(args.plan), str(local), str(args.output),
         "--model-id", spec["model_id"], "--revision", spec["revision"],
         "--grounding-root", str(args.grounding_root)],
        check=True,
    )

if __name__ == "__main__":
    main()
