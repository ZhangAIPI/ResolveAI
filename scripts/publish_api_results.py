"""Publish only completed API cohorts into Table 1 and its reproducibility notes."""
import fcntl
import json
from pathlib import Path
import re
import subprocess

from scripts.paper_result_table import render

LABELS={"terra-eval":"GPT-6.1 Terra","gpt61_sol":"GPT-6.1 Sol (low)",
        "gpt56_sol":"GPT-5.6 Sol","gpt41":"GPT-4.1","gpt41_mini":"GPT-4.1 mini",
        "gpt54_mini":"GPT-5.4 mini","luna":"GPT-6 Luna","luna_decisions":"GPT-6 Luna (Decisions)"}


def replace_block(text,start,end,body,anchor=None):
    block=start+"\n"+body+"\n"+end
    if start in text:
        return re.sub(re.escape(start)+r".*?"+re.escape(end),lambda _:block,text,flags=re.S)
    if anchor is None or anchor not in text:
        raise ValueError("Publication marker location changed")
    return text.replace(anchor,block+"\n"+anchor,1)


def publish(repository, report, lock_path):
    if not report["workers"]:
        return False
    with lock_path.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reference=json.loads((repository/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
        efficiency=json.loads((repository/"docs/results/cvpr_efficiency.json").read_text())
        table=repository/"paper/tables/benchmark_results.tex"
        table.write_text(render(reference,efficiency,report))
        workers=report["workers"]
        names=", ".join(LABELS[w["name"]] for w in workers)
        main=repository/"paper/main.tex"
        text=main.read_text()
        text=replace_block(text,"% api-models-start","% api-models-end",
                           "We additionally evaluate "+names+r" through the OpenAI API on the same cases~\cite{openai_api}.",
                           "Base checkpoints are evaluated without task demonstrations;")
        text=text.replace("We use greedy decoding, BF16 weights,","For local models, we use greedy decoding, BF16 weights,")
        text=text.replace(
            "Output tokens count generated answer/action text and end markers, excluding the input prompt and image representation.",
            "Output tokens count all generated tokens, including end markers and any API reasoning tokens; input prompts and images are excluded.")
        text=text.replace("Qwen3-VL-8B achieves the highest accuracy with tools, at","Qwen3-VL-8B reaches")
        inference=("API runs use provider-native sampling and retain the image-size, context, action and turn limits. "
                   "Their generation settings and treatment of the direct-only Decisions interface are given in the supplement.")
        text=replace_block(text,"% api-inference-start","% api-inference-end",inference,
                           "We also report mean episode time, input/output tokens")
        main.write_text(text)
        appendix=repository/"paper/appendix.tex"
        details=(r"\paragraph{API-served models.}"+"\n"+
                 "Responses runs use native function calls and best-effort schemas. Direct and interactive outputs permit "
                 "96 and 384 visible tokens. GPT-6.1 Sol uses low reasoning with a 2,048-token total cap; "
                 "the visible counts exclude its reported reasoning tokens, and a response exceeding the visible cap receives no credit. "
                 "We retain encrypted reasoning and native call identifiers between turns. "
                 "Other API models use their documented non-reasoning setting. "
                 "The Decisions API returns a choice among the same three verdicts from the same pixels; "
                 "it does not generate free-form action arguments, so we evaluate direct decisions only. "
                 "Its tool and generated-text entries in Table 1 are inapplicable. "
                 "All API times include network round trips, input counting and CPU tool execution; "
                 "they are not an equal-hardware throughput comparison with local GPUs. "
                 "Pilot response reuse retains original token counts and generation time rather than treating replay as fresh inference. "
                 "Checkpoint aliases or pinned snapshots and returned model IDs are recorded with the run."+"\n"+
                 r"\input{tables/api_settings}")
        source=appendix.read_text()
        if "% api-settings-start" not in source:
            source+="\n% api-settings-start\n"+details+"\n% api-settings-end\n"
        else:
            source=replace_block(source,"% api-settings-start","% api-settings-end",details)
        appendix.write_text(source)
        lines=[r"\begin{table*}[t]",r"\centering\small",
               r"\caption{\textbf{API inference costs on the fixed cohort.} Means include all attempts. Decisions uses the no-tools condition; other rows use the tool condition.}",
               r"\begin{tabular}{@{}llrrrrr@{}}",r"\toprule",
               r"Model & Endpoint & Input tokens (k) & Reasoning tokens & Calls & Tool calls & Finished (\%)\\",r"\midrule"]
        for worker in workers:
            arm="initial_text" if worker["endpoint"]=="decisions" else "interactive_agent"
            cost=worker["costs"][arm]
            values=[f"{cost['mean_input_tokens']/1000:.1f}",f"{cost['mean_reasoning_tokens']:.0f}",
                    f"{cost['mean_model_calls']:.2f}",f"{cost['mean_tool_calls']:.2f}",f"{cost['finished_pct']:.1f}"]
            lines.append(LABELS[worker["name"]]+" & "+worker["endpoint"]+" & "+" & ".join(values)+r"\\")
        lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
        (repository/"paper/tables/api_settings.tex").write_text("\n".join(lines)+"\n")
        bib=repository/"paper/references.bib"
        if "@misc{openai_api," not in bib.read_text():
            with bib.open("a") as out:
                out.write('\n@misc{openai_api,\n  author = {{OpenAI}},\n  title = {API Models and Decisions Documentation},\n  year = {2026},\n  howpublished = {\\url{https://developers.openai.com/api/docs/models}},\n  note = {Accessed November 3, 2026}\n}\n')
        readme=repository/"README.md"
        body=["**闭源对照：同一固定 209 族（来源标签，尚未独立审核）**","",
              "| 模型 | 无工具 | 有工具 |","| --- | ---: | ---: |"]
        for worker in workers:
            direct=worker["arms"]["initial_text"]
            initial=100*direct["label_matches"]/direct["episodes"]
            agent=worker["arms"].get("interactive_agent")
            tools=f'{100*agent["label_matches"]/agent["episodes"]:.1f}' if agent else "不适用"
            body.append(f'| {LABELS[worker["name"]]} | {initial:.1f} | {tools} |')
        body += ["","仅完整固定题组入表。Decisions 为直接分类接口；预算预留不等于实际支出。[完整记录](docs/results/openai_candidate_scores.json)。"]
        readme.write_text(replace_block(readme.read_text(),"<!-- api-results-start -->","<!-- api-results-end -->",
                                        "\n".join(body),"[**2026-11-01 四卡实验："))
        paths=["docs/results/openai_candidate_scores.json","README.md","paper/main.tex","paper/appendix.tex",
               "paper/references.bib","paper/tables/benchmark_results.tex","paper/tables/api_settings.tex"]
        difference=subprocess.run(["git","diff","--quiet","--",*paths],cwd=repository)
        untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard","--",*paths],cwd=repository,text=True)
        if difference.returncode==0 and not untracked:
            return False
        subprocess.run(["git","add","--",*paths],cwd=repository,check=True)
        subprocess.run(["git","diff","--cached","--check"],cwd=repository,check=True)
        subprocess.run(["git","commit","--only","-m","Update Table 1 with complete fixed-cohort API results","--",*paths],cwd=repository,check=True)
        subprocess.run(["git","push","origin","main"],cwd=repository,check=True)
        return True
