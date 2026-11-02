"""Run the leak-checked LoRA pilot, then evaluate its adapter on the frozen plan."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--grounding-root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    dataset = json.loads((args.data / "manifest.json").read_text())
    if dataset["eval_dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("Training was not checked against this frozen benchmark")
    models = json.loads((args.plan.parent / "models.json").read_text())
    base = next(m for m in models if m["model_id"] == "Qwen/Qwen3-VL-8B-Instruct")
    subprocess.run([sys.executable, "-u", "-m", "scripts.train_acquisition_lora",
                    str(args.data), str(args.base), str(args.output),
                    "--base-revision", base["revision"]], check=True)
    complete = json.loads((args.output / "complete.json").read_text())
    adapter = Path(complete["adapter"])
    revision = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    subprocess.run([
        sys.executable, "-u", "-m", "scripts.evaluate_study", str(args.plan),
        str(adapter), str(args.output / "benchmark"),
        "--model-id", "Qwen/Qwen3-VL-8B-Instruct+ResolveAI-acquisition-LoRA",
        "--revision", revision, "--grounding-root", str(args.grounding_root),
    ], check=True)
    subprocess.run([sys.executable, "-m", "scripts.score_sft_comparison",
        "docs/results/cvpr_20261102_candidate_scores.json", str(args.plan.parent / "qwen3"),
        str(args.output / "benchmark"), str(args.output), str(args.output / "paper")], check=True)
    print(json.dumps({"stage": "benchmark_complete", "adapter_sha256": revision,
                      "measured_table": str(args.output / "paper/sft_results.tex")}), flush=True)


if __name__ == "__main__":
    main()
