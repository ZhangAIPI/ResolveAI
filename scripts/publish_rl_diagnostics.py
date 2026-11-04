"""Publish fixed-final GRPO diagnostics separately from validation-selected main results."""
import argparse
import fcntl
import json
import hashlib
import os
from pathlib import Path
import time
from scripts.paper_result_table import NAMES
from scripts.score_sft_comparison import compare, initial_rl_checkpoint
from scripts.watch_adaptation_results import complete_cohort, publish_changes, run
from scripts.sync_api_paper import synchronize

START, END = "% rl-final-diagnostics-start", "% rl-final-diagnostics-end"


def render(reports):
    lines = [
        r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{minipage}{.94\textwidth}\centering",
        r"\caption{\textbf{Checkpoint choice after online GRPO.}",
        r"SFT and Final GRPO match the main training comparison; Selected GRPO uses held-out validation selection.",
        r"Condition, Identity and Overall are source-label accuracy (\%) on initially sufficient cases.",
        r"Time (s) and output tokens are means over all attempts.",
        r"Request (\%) is the fraction requesting a photograph in Obtainable, a different access version.",
        r"$^{*}$Validation retained the initial SFT policy. Both checkpoint choices are retained; the main table uses the fixed training endpoint.}",
        r"\label{tab:rl-final}",
        r"\begin{tabular}{@{}llrrrrrr@{}}\toprule",
        r"Model & Checkpoint & Condition & Identity & Overall & Time (s) & Output & Request (\%)\\\midrule",
    ]
    for group in reports:
        for index, (stage, report) in enumerate(group):
            if index == 0:
                label = NAMES[report["base_model_id"]]
            else:
                label = ""
            if initial_rl_checkpoint(report):
                stage += r"$^{*}$"
            scores = report["models"][1]["arms"]["interactive_agent"]
            cost = report["efficiency"][1]["arms"]["interactive_agent"]
            behavior = report["acquisition_behavior"][1]["by_variant"]["Obtainable"]
            values = [100*scores["by_task"][task]["label_agreement"] for task in ("state","identity")]
            values += [100*scores["label_agreement"],cost["mean_wall_s"],cost["mean_output_tokens"],
                       100*behavior["episodes_with_request"]/behavior["episodes"]]
            cells = [f"{v:.1f}" if i != 4 else str(round(v)) for i,v in enumerate(values)]
            lines.append(label+" & "+stage+" & "+" & ".join(cells)+r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines+[r"\end{tabular}\end{minipage}\end{table*}"])+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    parser.add_argument("--once",action="store_true")
    args = parser.parse_args()
    if os.getenv("SLURMD_NODENAME") != "macula":
        parser.error("Use a CPU-only allocation on the retained GPU host")
    repo,runtime = args.repository,args.runtime
    os.chdir(repo)
    root=runtime/"evaluation/cvpr-paper-priority"
    with (root/"final-diagnostics-publication.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference=repo/"docs/results/cvpr_20261102_candidate_scores.json"
        cohort=json.loads(reference.read_text())
        plan=runtime/"evaluation/cvpr-20261101/plan.json"
        print(json.dumps({"final_rl_diagnostics":"waiting","checkpoint_group":64}),flush=True)
        while True:
            try:
                groups,changes=[],{}
                ready=True
                for size in ("2b","8b"):
                    directory=root/("qwen3_"+size+"_grpo_final")
                    target="docs/results/cvpr_"+size+"_grpo_final_candidate_scores.json"
                    if (repo/target).exists():
                        final=json.loads((repo/target).read_text())
                    elif complete_cohort(directory/"benchmark",cohort["family_ids"]):
                        base=root/"qwen3_2b" if size=="2b" else runtime/"evaluation/cvpr-20261101/qwen3"
                        final=compare(reference,base,directory/"benchmark",directory,plan)
                    else:
                        ready=False
                        continue
                    if final["checkpoint"]["evaluated_group"] != 64:
                        raise ValueError("Diagnostic must use the predeclared final group")
                    prefix="cvpr_2b_mixed_sft_candidate_scores.json" if size=="2b" else "cvpr_mixed_sft_candidate_scores.json"
                    sft=json.loads((repo/"docs/results"/prefix).read_text())
                    selected=json.loads((repo/"docs/results"/("cvpr_"+size+"_grpo_candidate_scores.json")).read_text())
                    groups.append([("SFT",sft),("Selected GRPO",selected),("Final GRPO",final)])
                    changes[target]=json.dumps(final,indent=2)+"\n"
                if not groups:
                    if args.once:
                        return
                    time.sleep(30)
                    continue
                signature=hashlib.sha256(json.dumps([g[-1][1]["checkpoint"]["adapter_sha256"] for g in groups]).encode()).hexdigest()
                state_path=root/"final-diagnostics-published.json"
                if state_path.exists() and json.loads(state_path.read_text()).get("signature")==signature:
                    if ready or args.once:
                        return
                    time.sleep(30)
                    continue
                changes["paper/tables/rl_final_diagnostics.tex"]=render(groups)
                paragraph=(
                    "\\section{Final-Checkpoint Diagnostics}\n"
                    "The main GRPO results use the fixed endpoint at group 64. Validation selection also includes the initial policy as a candidate. "
                    "We separately evaluate the fixed final checkpoint at group 64 for both backbones to examine the policies after actual optimization. "
                    "The endpoint evaluations were specified before their scoring. We adopted endpoint reporting for the main table on November 4 after inspecting both checkpoint variants for 2B and 8B; both variants remain reported here. Remaining RL runs follow the same endpoint rule regardless of their test scores. "
                    "The data, tools, inference allowances and fixed paper cohort remain unchanged.\n"
                    "\\input{tables/rl_final_diagnostics}\n")
                for group in groups:
                    sft,selected,final=[report for _,report in group]
                    score=lambda report:100*report["models"][1]["arms"]["interactive_agent"]["label_agreement"]
                    behavior=final["acquisition_behavior"][1]["by_variant"]["Obtainable"]
                    paragraph += (NAMES[final["base_model_id"]]+f" scores {score(final):.1f}\\% at the final checkpoint, "
                        +f"compared with {score(selected):.1f}\\% after validation selection and {score(sft):.1f}\\% after SFT. "
                        +f"It requests photographs in {behavior['episodes_with_request']} Obtainable cases; "
                        +"request frequency alone does not establish evidence utility.\n")
                appendix=(repo/"paper/appendix.tex").read_text()
                block=START+"\n"+paragraph+END
                if START in appendix:
                    a=appendix.index(START);b=appendix.index(END,a)+len(END)
                    appendix=appendix[:a]+block+appendix[b:]
                else:
                    appendix=appendix.rstrip()+"\n\n"+block+"\n"
                changes["paper/appendix.tex"]=appendix
                docpath="docs/experiments/cvpr_20261104-adaptation.zh-CN.md"
                doc=(repo/docpath).read_text()
                first,last="<!-- final-rl-results-start -->","<!-- final-rl-results-end -->"
                summary=[first,"## 最终 RL 权重补充结果","",
                         "主表 SFT+RL 统一使用训练结束的第 64 组权重；验证选择的权重作为附录对照，不增加一种训练方法。该展示规则于 11 月 4 日在两种权重结果均已可见后调整；后续所有 RL 实验采用相同训练终点，不按测试高低挑权重。","",
                         "| 模型 | SFT | 验证选择的 GRPO | 最终 GRPO |",
                         "|---|---:|---:|---:|"]
                for group in groups:
                    values=[100*report["models"][1]["arms"]["interactive_agent"]["label_agreement"] for _,report in group]
                    summary.append("| "+NAMES[group[0][1]["base_model_id"]]+" | "+" | ".join(f"{v:.1f}%" for v in values)+" |")
                summary.append(last)
                block="\n".join(summary)
                if first in doc:
                    a=doc.index(first);b=doc.index(last,a)+len(last)
                    doc=doc[:a]+block+doc[b:]
                else:
                    doc=doc.rstrip()+"\n\n"+block+"\n"
                changes[docpath]=doc
                publish_changes(repo,runtime,changes,"Report fixed-final GRPO checkpoints alongside validation selection")
                if run(["git","status","--porcelain"],repo).strip():
                    raise RuntimeError("Independent edits; defer Overleaf sync")
                synchronize(repo,runtime,"6ae80679a4c7a64df04ac5b8")
                state_path.write_text(json.dumps({"signature":signature,"complete":ready},indent=2)+"\n")
                print(json.dumps({"final_rl_diagnostics":"published","backbones":len(groups),"all_complete":ready}),flush=True)
                if ready or args.once:
                    return
                time.sleep(30)
            except Exception as error:
                message=str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__
                print(json.dumps({"diagnostics_waiting":message}),flush=True)
                if args.once:
                    raise SystemExit(1)
                time.sleep(30)


if __name__=="__main__":
    main()
