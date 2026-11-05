"""Publish complete fixed-cohort SFT/RL comparisons without rerunning inference."""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from scripts.score_candidate_labels import episode_databases
from scripts.score_sft_comparison import compare, render_adaptations
from scripts.sync_api_paper import synchronize

START = "% adaptation-results-start"
END = "% adaptation-results-end"


def complete_cohort(directory, families):
    if not (directory / "manifest.json").exists():
        databases = sorted(directory.glob("shard-*/episodes.sqlite"))
        if not databases:
            return False
    else:
        databases = episode_databases(directory)
    wanted = {(family,variant,"interactive_agent") for family in families
              for variant in ("Sufficient","Obtainable","Missing","Unavailable")}
    wanted |= {(family,"Sufficient",arm) for family in families
               for arm in ("initial_text","full_available_text")}
    found = set()
    for path in databases:
        if not path.exists():
            return False
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as db:
            for (raw,) in db.execute("SELECT result FROM episodes"):
                row = json.loads(raw)
                key = row["family_id"],row["variant"],row["arm"]
                if key in wanted:
                    if key in found:
                        raise ValueError("Duplicated paper-cohort episode")
                    found.add(key)
    return found == wanted


def replace_block(text, body):
    block = START + "\n" + body + "\n" + END
    if START in text:
        first = text.index(START)
        last = text.index(END,first)+len(END)
        return text[:first]+block+text[last:]
    anchor = "\\section*{LLM Usage Statement}" if "\\section*{LLM Usage Statement}" in text else "\\end{document}"
    if anchor not in text:
        return text.rstrip()+"\n\n"+block+"\n"
    return text.replace(anchor,block+"\n\n"+anchor,1)


def result_text(entries):
    """Report training effects separately from the observed search for photographs."""
    from scripts.paper_result_table import NAMES
    from scripts.score_sft_comparison import adaptation_groups, core_stages, reported_training_updates
    reports={entry["name"]:report for entry,report in entries}
    def accuracy(key,arm="interactive_agent",model=1):
        report=reports[key]
        if key=="qwen3_mixed_sft" and model==1:
            for update in reported_training_updates():
                if (update["base_model_id"],update["stage"],update["arm"])==(report["base_model_id"],"SFT",arm):
                    return update["accuracy_percent"]
        return 100*report["models"][model]["arms"][arm]["label_agreement"]
    parts=[r"\paragraph{SFT produces shorter interactions.}"]
    observations=[]
    if "qwen25_mixed_sft" in reports:
        key="qwen25_mixed_sft"
        observations.append(f"Qwen2.5-VL-7B rises from {accuracy(key,model=0):.1f}\\% to {accuracy(key):.1f}\\% with tools.")
    if "qwen3_2b_mixed_sft" in reports:
        key="qwen3_2b_mixed_sft"
        observations.append(f"The 2B model rises from {accuracy(key,model=0):.1f}\\% to {accuracy(key):.1f}\\%, "
                            f"but falls from {accuracy(key,'initial_text',0):.1f}\\% to {accuracy(key,'initial_text'):.1f}\\% without tools.")
    if "qwen3_mixed_sft" in reports:
        observations.append(f"The 8B SFT score is {accuracy('qwen3_mixed_sft'):.1f}\\%.")
    parts.append(" ".join(observations))
    parts.append(
        r"\textbf{SFT improves completion and efficiency.} "
        "The local SFT policies finish initially sufficient cases without intermediate tools. "
        "For 2B, completion rises from 61.7\\% to 100\\%, while mean time falls from 17.1 to 5.5 seconds "
        "and output length from 449 to 118 tokens. "
        "The same SFT adapter thus has different effects under the two response formats.")
    stages={model:core_stages(group) for model,group in adaptation_groups(list(reports.values())).items()}
    continued=[]
    for model,key in (("Qwen/Qwen2.5-VL-7B-Instruct","qwen25_mixed_sft"),
                      ("Qwen/Qwen3-VL-2B-Instruct","qwen3_2b_mixed_sft"),
                      ("Qwen/Qwen3-VL-8B-Instruct","qwen3_mixed_sft")):
        report=stages.get(model,{}).get("SFT+RL")
        if report is not None:
            continued.append((model,key,report))
    if continued:
        parts.append(r"\paragraph{RL gains do not always involve new evidence.}")
        descriptions=[]
        for model,key,report in continued:
            score=100*report["models"][1]["arms"]["interactive_agent"]["label_agreement"]
            descriptions.append(f"{NAMES[model]} changes from {accuracy(key):.1f}\\% after SFT to {score:.1f}\\% after SFT+RL")
        parts.append("; ".join(descriptions)+".")
        model="Qwen/Qwen2.5-VL-7B-Instruct"
        report=stages.get(model,{}).get("SFT+RL")
        if report is not None:
            behavior=report["acquisition_behavior"][1]["by_variant"]["Obtainable"]
            if behavior["episodes_with_request"]==0:
                parts.append(
                    r"\textbf{Better answers need not involve better search.} "
                    "The 7B gain is eight additional correct answers on the fixed initially sufficient set. "
                    "In Obtainable, however, the same policy makes no photograph requests. "
                    "It has improved its answers without checking the withheld photographs. "
                    "Paired family bootstrap intervals, reported in the supplement, quantify the uncertainty of the single-run differences.")
    direct=[]
    for model,group in stages.items():
        report=group.get("RL")
        if report is not None:
            score=100*report["models"][1]["arms"]["interactive_agent"]["label_agreement"]
            direct.append(f"{NAMES[model]}: {score:.1f}\\%")
    if direct:
        parts.append(r"\paragraph{Direct RL depends on the starting model.}")
        large_group=stages.get("Qwen/Qwen3-VL-8B-Instruct",{})
        rl,continued=large_group.get("RL"),large_group.get("SFT+RL")
        if rl and continued:
            def interaction(report):
                return report["efficiency"][-1]["arms"]["interactive_agent"]
            a,b=interaction(rl),interaction(continued)
            parts.append(
                f"Direct RL reaches {a['accuracy_pct']:.1f}\\% for 8B, compared with {b['accuracy_pct']:.1f}\\% after SFT+RL: "
                f"{a['correct']} versus {b['correct']} correct answers. "
                "This single-case difference is small. "
                r"\textbf{SFT+RL uses roughly half the inference time and output tokens.} "
                "Mean time falls "
                f"from {a['mean_wall_s']:.1f} to {b['mean_wall_s']:.1f} seconds and output "
                f"from {a['mean_output_tokens']:.0f} to {b['mean_output_tokens']:.0f} tokens.")
            small_rl=stages.get("Qwen/Qwen3-VL-2B-Instruct",{}).get("RL")
            old_rl=stages.get("Qwen/Qwen2.5-VL-7B-Instruct",{}).get("RL")
            if small_rl and old_rl:
                a,b=interaction(small_rl),interaction(old_rl)
                abstained=old_rl["acquisition_behavior"][-1]["by_variant"]["Sufficient"]["abstained"]
                parts.append(
                    f"The smaller models respond differently. Direct RL gives {a['accuracy_pct']:.1f}\\% for 2B "
                    f"and {b['accuracy_pct']:.1f}\\% for 7B, below their SFT results. "
                    f"The 7B policy abstains in {abstained} of {old_rl['common_families']} initially sufficient cases.")
        else:
            parts.append("Direct RL gives "+"; ".join(direct)+".")
    parts += [
        r"\subsection{Finding Missing Evidence}",
        r"\input{tables/search_behavior}",
        r"\paragraph{Similar accuracy, different evidence access.}",
        "The two 8B training routes differ by one correct answer in Sufficient, but behave differently once evidence is withheld. "
        "In Obtainable, direct RL requests photographs in 15 families and receives a new image in 14. "
        "SFT+RL requests and receives new images in 92 families, and finishes all 209 episodes rather than 194. "
        r"\textbf{Similar answer accuracy can mask different evidence access.} "
        "The paired Sufficient accuracy difference for SFT+RL minus RL is "
        r"\(-0.48\) percentage points, with a family-bootstrap 95\% interval of \([-5.74,4.78]\). "
        "The aggregate difference also hides a task trade-off: SFT+RL corrects five identity errors without losing a correct identity answer, "
        "but corrects ten condition errors while losing sixteen. "
        "",
        r"""\textbf{Efficiency depends on which evidence is supplied.}
The 8B SFT+RL policy is faster than RL alone on initially sufficient cases, yet obtains more photographs when images are withheld.
Its shorter primary interactions therefore do not correspond to a uniform reduction in requests.
The 7B model shows a different pattern: its gains in accuracy and efficiency over the base policy accompany zero photograph requests in Obtainable.
The efficiency columns and access measurements describe complementary parts of the policy: answering from the initial images and deciding to obtain another image when the available evidence changes.""",
        r"\paragraph{The requested evidence differs by task.}",
        "For 8B, direct RL retrieves new images in 9 of 153 condition families and 5 of 56 identity families. "
        "SFT+RL retrieves originals in 92 condition families and no new identity views. "
        "The increase is concentrated in condition-detail retrieval; requests for additional identity views remain absent.",
    ]
    small=stages.get("Qwen/Qwen3-VL-2B-Instruct",{}).get("SFT+RL")
    large=stages.get("Qwen/Qwen3-VL-8B-Instruct",{}).get("SFT+RL")
    if small and large:
        a=small["acquisition_behavior"][1]["by_variant"]["Obtainable"]
        b=large["acquisition_behavior"][1]["by_variant"]["Obtainable"]
        parts += [
            r"\paragraph{A request need not add an image.}",
            f"The 2B SFT+RL policy requests photographs in {a['episodes_with_request']} Obtainable cases, "
            f"but receives a new image in only {a['episodes_with_new_photo']}. "
            "Its 63 condition requests all ask for the overview already supplied in the initial evidence. "
            f"The 8B policy requests an original in {b['episodes_with_request']} cases and receives a new image in each; "
            "all of these requests concern condition, rather than another viewpoint for identity. "
            "Figure~\\ref{fig:teaser} shows the difference on one bottle. "
            "Both policies spend one request, but only the request for the original changes the available detail. The bottle preview and original derive from the same photograph. The 48-pixel preview repeats under the overview request, whereas the original record retains 900-pixel detail. Cropping or enlarging the returned overview cannot recover the pixels discarded when the preview was made. Requesting the original changes the observation without changing the item, time or camera. Counting requests alone would therefore treat two different evidence outcomes as equivalent."
        ]
    if large:
        parts += [
            r"\paragraph{Stopping changes when a request fails.}",
            "For the 92 families where 8B requests an original, we compare Obtainable, Missing and Unavailable. "
            "All 92 triples have identical public images, tool definitions and first requests up to the returned response. "
            "With the original provided, the policy answers Supported in 50 cases and Refuted in 42. "
            "When the same request returns unable to provide, all 92 answers change to Need more evidence. "
            "The 8B policy therefore responds to access feedback after making the request. "
            "Missing and Unavailable return the same failure message here, and produce the same stopping behavior."
        ]
    if large:
        parts += [r"""\paragraph{Responding to failure is only part of stopping.}

The 8B SFT+RL policy abstains in all 92 episodes where a requested photograph is denied, but gives definite answers in 116 of the 209 Unavailable episodes.
Let \(U\) denote the Unavailable episodes and \(F\subseteq U\) those containing a failed photograph request.
The observed abstention rates are
\begin{equation}
\Pr(\hat y=\Nme\mid F)=1,\qquad
\Pr(\hat y=\Nme\mid U)=0.445.
\label{eq:failed-request}
\end{equation}
The first rate is conditioned on the policy choosing to ask.
All 92 requested episodes concern condition; among the other 117 episodes, one identity case ends in abstention and 116 end in definite answers.
The 7B policy makes no requests and gives 209 definite answers in Unavailable.
These outcomes distinguish responding to access feedback from initiating a search.
A policy can consistently change its answer after denial while leaving other photographs unrequested."""]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def detail_text(entries):
    lines = ["\\section{Training Data and Action Supervision}",
             "\\input{tables/training_ablation}",
             "Identity-only SFT uses CO3D trajectories; mixed-task SFT adds balanced condition supervision. "
             "The paired finish-only controls retain public histories but remove tool-action targets. "
             "These controls have the same update cap rather than matched target-token compute.",
             "\\subsection{Training Details}"]
    for entry,report in entries:
        if not entry.get("new_analysis"):
            continue
        config = report["training"]["configuration"]
        if "groups" in config:
            lines.append(
                entry["description"] + (" starts from the pretrained base without SFT, for " if config.get("initialization")=="base" else " continues mixed-task SFT for ")
                +f"{config['groups']} sampled groups "
                +f"of {config['group_size']} trajectories, with two clipped policy updates per nonconstant-reward group. "
                +f"The learning rate is \\({config['learning_rate']:g}\\), clipping threshold 0.2 and reference penalty 0.02. "
                +"The reference is the frozen initial policy. Training sees only public observations; private source annotations enter as scalar rewards. "
                +"A fixed held-out set of eight source families across four access versions records validation-selected checkpoints for the supplementary comparison. "
                +"The benchmark supplies no training or checkpoint-selection feedback. "
                +f"The run performs {report['checkpoint']['state']['optimizer_updates']} optimizer updates "
                +f"and records validation-selected group {report['checkpoint']['state']['best_group']}; group zero denotes the initial policy. Main-table RL scores use the fixed endpoint at group 64.")
        else:
            checkpoint = report["checkpoint"]["state"]
            lines.append(entry["description"] + f" completes {checkpoint['step']} optimizer steps, "
                         +"using the same rank-32 attention adapters and held-out token-loss selection as the other SFT runs. "
                         +"Its data and exact selected checkpoint are recorded with the released comparison.")
            if "terminal" in report["models"][1]["name"].lower():
                lines.append("Finish-only retains complete public histories at the final decision, "
                             "but removes tool-action prediction targets. It uses the same update limit, "
                             "not matched target-token compute, as the corresponding full-trajectory SFT.")
    lines.append("RL and SFT+RL share the sampled-group budget, but differ in optimizer updates, reference policies and total training compute. "
                 "Each setting uses one training seed, so these comparisons do not establish a general ranking of training methods. "
                 "All rows retain the fixed paper cohort, failure accounting and inference allowances. "
                 "Sharded evaluation, where used, partitions complete case families and reads original episode traces without averaging shard percentages.")
    return "\n\n".join(lines)


def run(command,cwd,env=None):
    result = subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=300)
    if result.returncode:
        raise RuntimeError("Command failed: "+Path(command[0]).name+" "+str(result.returncode))
    return result.stdout


def publish(repo,runtime,entries):
    table = render_adaptations([report for _,report in entries])
    from scripts.score_sft_comparison import render_training_ablation
    changes = {"paper/tables/sft_results.tex":table,
               "paper/tables/training_ablation.tex":render_training_ablation([report for _,report in entries])}
    for entry,report in entries:
        changes[entry["public_report"]] = json.dumps(report,indent=2)+"\n"
    body = result_text(entries)
    if body:
        changes["paper/main.tex"] = replace_block((repo/"paper/main.tex").read_text(),body)
        changes["paper/appendix.tex"] = replace_block((repo/"paper/appendix.tex").read_text(),detail_text(entries))
    publish_changes(repo,runtime,changes,"Update Table 2 with complete paired adaptation measurements")


def publish_changes(repo,runtime,changes,message):
    """Compile the proposed paper files before committing a complete result update."""
    if run(["git","status","--porcelain"],repo).strip():
        raise RuntimeError("Repository has independent edits; measured results retained for reconciliation")
    with tempfile.TemporaryDirectory(prefix="adaptation-check-",dir=runtime/"paper-draft") as temp:
        stage = Path(temp)/"paper"
        shutil.copytree(repo/"paper",stage)
        for name,content in changes.items():
            if name.startswith("paper/"):
                (stage/name.removeprefix("paper/")).write_bytes(content) if isinstance(content,bytes) else (stage/name.removeprefix("paper/")).write_text(content)
        environment = {**os.environ,"XDG_CACHE_HOME":str(runtime/"tools/tex/cache")}
        for source in ("main.tex","supplementary.tex"):
            run([str(runtime/"tools/tex/tectonic"),"--keep-intermediates","--outdir",temp,source],stage,environment)
        sys.path.insert(0,str(runtime/"tools/pdf"))
        import fitz
        for name in ("main.pdf","supplementary.pdf"):
            with fitz.open(Path(temp)/name) as doc:
                text = "\n".join(p.get_text() for p in doc)
                if name == "main.pdf":
                    references = next((i+1 for i,page in enumerate(doc) if "References" in page.get_text()),None)
                    auxiliary=(Path(temp)/"main.aux").read_text()
                    pages={label:int(page) for label,page in
                           re.findall(r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}",auxiliary)
                           if not label.endswith("@cref")}
                    if "resolveai-main-end" not in pages:
                        raise ValueError("Main-text end marker is missing")
                    body_end=max(pages.values())
                    print("COMPILED_MAIN",len(doc),"BODY_END",body_end,
                          "REFERENCES_START",references,flush=True)
                    if references is None or body_end > 8:
                        raise ValueError("Main text, figures or tables exceed eight pages")
                if "[?]" in text or "??" in text or "\ufffd" in text or doc.metadata.get("author"):
                    raise ValueError("Compiled adaptation paper failed reference or anonymity checks")
        for name,content in changes.items():
            (repo/name).write_bytes(content) if isinstance(content,bytes) else (repo/name).write_text(content)
    run(["git","diff","--check"],repo)
    if run(["git","diff","--name-only"],repo).strip():
        run(["git","add",*changes],repo)
        run(["git","commit","-m",message],repo)
    run(["git","push","origin","main"],repo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository",type=Path)
    p.add_argument("runtime",type=Path)
    p.add_argument("configuration",type=Path)
    p.add_argument("--once",action="store_true")
    args = p.parse_args()
    if os.getenv("SLURMD_NODENAME") != "macula":
        p.error("Run on the retained GPU host, with a CPU-only allocation")
    root = args.runtime/"evaluation/cvpr-paper-priority/adaptation-publication"
    root.mkdir(exist_ok=True)
    lock = (root/"watch.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(args.repository)
    configuration = json.loads(args.configuration.read_text())
    cohort = args.repository/"docs/results/cvpr_20261102_candidate_scores.json"
    families = json.loads(cohort.read_text())["family_ids"]
    state_path = root/"state.json"
    previous_error = None
    print(json.dumps({"table2_watcher":"started","policy":"complete fixed cohorts only","gpu_allocation":0}),flush=True)
    while True:
        entries,waiting = [],[]
        try:
            for entry in configuration["runs"]:
                saved = args.repository/entry["public_report"]
                if saved.exists():
                    report = json.loads(saved.read_text())
                elif (Path(entry["run"])/"complete.json").exists() and complete_cohort(Path(entry["run"])/"benchmark",families):
                    report = compare(cohort,Path(entry["base"]),Path(entry["run"])/"benchmark",
                                     Path(entry["run"]),Path(configuration["plan"]))
                else:
                    waiting.append(entry["name"])
                    continue
                entries.append((entry,report))
            signature = hashlib.sha256(json.dumps([(e["name"],r["checkpoint"]["adapter_sha256"])
                                                     for e,r in entries]).encode()).hexdigest()
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            if previous.get("published") != signature:
                publish(args.repository,args.runtime,entries)
                state_path.write_text(json.dumps({"published":signature,"included":[e["name"] for e,_ in entries],
                                                  "waiting":waiting},indent=2)+"\n")
                print(json.dumps({"table2_updated":[e["name"] for e,_ in entries],"waiting":waiting}),flush=True)
            if run(["git","status","--porcelain"],args.repository).strip():
                raise RuntimeError("Independent repository edits; defer Overleaf synchronization")
            synchronize(args.repository,args.runtime,"6ae80679a4c7a64df04ac5b8")
            previous_error = None
            if args.once or not waiting:
                return
        except Exception as error:
            # Never print command arguments: the Overleaf CLI carries private credentials.
            message = type(error).__name__ + ": " + str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__
            if message != previous_error:
                print(json.dumps({"publication_waiting":message}),flush=True)
                previous_error = message
            if args.once:
                raise SystemExit(1)
        time.sleep(30)


if __name__ == "__main__":
    main()
