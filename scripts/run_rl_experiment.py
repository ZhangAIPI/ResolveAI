"""Train online GRPO from a fixed SFT adapter, then compare base/SFT/RL on the frozen study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("model_spec",type=Path)
    parser.add_argument("sft_run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("plan",type=Path)
    parser.add_argument("--base-result",type=Path,required=True)
    parser.add_argument("--grounding-root",type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if json.loads((args.data/"manifest.json").read_text())["eval_dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("RL source checks refer to a different benchmark")
    subprocess.run([sys.executable,"-u","-m","scripts.train_online_grpo",str(args.data),
        str(args.model_spec),str(args.sft_run),str(args.grounding_root),str(args.output)],check=True)
    complete = json.loads((args.output/"complete.json").read_text())
    if complete["state"]["optimizer_updates"] < 1:
        raise ValueError("No online policy update was completed")
    spec = json.loads(args.model_spec.read_text())
    subprocess.run([sys.executable,"-u","-m","scripts.evaluate_study",str(args.plan),
        complete["adapter"],str(args.output/"benchmark"),"--model-id",spec["model_id"]+"+ResolveAI-GRPO",
        "--revision",complete["adapter_sha256"],"--grounding-root",str(args.grounding_root)],check=True)
    from scripts.score_sft_comparison import compare
    cohort = Path("docs/results/cvpr_20261102_candidate_scores.json")
    rl = compare(cohort,args.base_result,args.output/"benchmark",args.output,args.plan)
    waiting = time.monotonic()
    while not (args.sft_run/"paper/sft_results.json").exists():
        if time.monotonic()-waiting > 21600:
            raise TimeoutError("SFT evaluation summary was not ready within six hours")
        print(json.dumps({"stage":"waiting_for_sft_evaluation_summary"}),flush=True)
        time.sleep(20)
    sft = compare(cohort,args.base_result,args.sft_run/"benchmark",args.sft_run,args.plan)
    if rl["family_ids"] != sft["family_ids"] or rl["dataset_sha256"] != sft["dataset_sha256"]:
        raise ValueError("SFT and RL comparison cohorts differ")
    report = {key:rl[key] for key in ("scope","common_families","task_counts","base_model_id",
                                      "dataset_sha256","family_ids","formal_grounded_accuracy")}
    report["models"] = [rl["models"][0],
        {**sft["models"][1],"name":"SFT"},{**rl["models"][1],"name":"SFT + online GRPO"}]
    report["sft_training"] = sft["training"]
    report["rl_training"] = rl["training"]
    report["rl_checkpoint"] = rl["checkpoint"]
    report["acquisition_behavior"] = [rl["acquisition_behavior"][0],
        sft["acquisition_behavior"][1],rl["acquisition_behavior"][1]]
    destination = args.output/"paper"
    destination.mkdir(exist_ok=True)
    (destination/"training_stages.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"rl_benchmark_complete","measured_report":str(destination/"training_stages.json")}),flush=True)


if __name__ == "__main__":
    main()
