"""Publish complete fixed-cohort SFT/RL comparisons without rerunning inference."""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
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
    """Write the adaptation comparison as connected findings, with checkpoint caveats."""
    reports = {entry["name"]: report for entry, report in entries}
    def accuracy(key, arm="interactive_agent", model=1):
        return 100 * reports[key]["models"][model]["arms"][arm]["label_agreement"]
    def requests(key):
        return reports[key]["acquisition_behavior"][1]["by_variant"]["Obtainable"]["episodes_with_request"]
    mixed8, mixed2 = "qwen3_mixed_sft", "qwen3_2b_mixed_sft"
    parts = [
        r"\paragraph{Supervised adaptation.}",
        "We train attention LoRA adapters for Qwen3-VL-2B, Qwen3-VL-8B and Qwen2.5-VL-7B, "
        "freezing the base and vision weights. "
        "Mixed-task SFT balances 1,080 decisions per task from CO3D and VisA~\\cite{visa}, "
        "with 360 validation decisions per task. Training and validation images are disjoint from evaluation. "
        "All runs use rank 32 and a learning rate of \\(10^{-4}\\), with at most 200 updates and "
        "checkpoint selection by held-out assistant-token loss. Each adaptation is a single run; "
        "the supplement gives stopping steps, data construction and training details.",
    ]
    observations = []
    if "qwen25_mixed_sft" in reports:
        key = "qwen25_mixed_sft"
        observations.append(
            f"Qwen2.5-VL-7B improves from {accuracy(key,model=0):.1f}\\% to {accuracy(key):.1f}\\% "
            f"with tools, and from {accuracy(key,'initial_text',0):.1f}\\% to {accuracy(key,'initial_text'):.1f}\\% without tools.")
    if mixed2 in reports:
        observations.append(
            f"For Qwen3-VL-2B, interactive accuracy rises from {accuracy(mixed2,model=0):.1f}\\% "
            f"to {accuracy(mixed2):.1f}\\%, while direct accuracy falls from "
            f"{accuracy(mixed2,'initial_text',0):.1f}\\% to {accuracy(mixed2,'initial_text'):.1f}\\%. "
            "Its interaction completion rate rises from 61.7\\% to 100\\%.")
    if mixed8 in reports:
        observations.append(
            f"The stronger 8B base does not benefit: mixed-task SFT reaches {accuracy(mixed8):.1f}\\% "
            f"with tools, below the base's {accuracy(mixed8,model=0):.1f}\\%.")
    parts += [r"\paragraph{Large gains can reflect easier completion.}", " ".join(observations),
              "All three mixed-task adapters finish the initially sufficient cases without intermediate tools. "
              "They shorten the interaction, but higher accuracy in this setting alone does not show better acquisition of missing evidence."]
    if mixed8 in reports:
        parts.append(
            f"In Obtainable, the 8B SFT policy requests photographs in {requests(mixed8)} cases, "
            "all concerning object condition; the 2B and 7B SFT policies make no requests. "
            "Training-data and action-supervision ablations are given in "
            "the supplementary material.")
    rl_keys = [key for key in ("qwen3_2b_grpo","qwen3_8b_grpo") if key in reports]
    if rl_keys:
        parts += [r"\paragraph{Reinforcement learning.}",
                  "We apply GRPO either directly to the pretrained base (RL) or after SFT "
                  "(SFT$\\to$RL), using 64 groups of four sampled tool trajectories. "
                  "Each run freezes its own initial policy as the reference. Source-derived correctness and action costs determine rewards; "
                  "eight held-out source families, evaluated across all four access versions, select the checkpoint. "
                  "No benchmark outcome enters optimization or checkpoint selection."]
        if "qwen3_2b_grpo" in reports:
            key = "qwen3_2b_grpo"
            gain = reports[key]["models"][1]["arms"]["interactive_agent"]["label_matches"] - reports[mixed2]["models"][1]["arms"]["interactive_agent"]["label_matches"]
            parts.append(
                f"For 2B, interactive accuracy changes from {accuracy(mixed2):.1f}\\% to {accuracy(key):.1f}\\%---"
                f"{gain} additional correct answer on the fixed cohort---and the policy still requests no photographs.")
        if "qwen3_8b_grpo" in reports:
            key = "qwen3_8b_grpo"
            if reports[key]["checkpoint"]["state"]["best_group"] == 0:
                parts.append(
                    "For 8B, validation retains the initial SFT checkpoint despite 24 subsequent optimizer updates. "
                    f"Its {accuracy(key):.1f}\\% row in Table~\\ref{{tab:sft}} therefore reports the initialization, "
                    "rather than an improvement due to RL.")
            else:
                parts.append(f"The selected 8B GRPO policy scores {accuracy(key):.1f}\\% with tools.")
        additions = []
        for key in ("qwen25_grpo","qwen25_rl_only","qwen3_2b_rl_only","qwen3_8b_rl_only"):
            if key not in reports:
                continue
            report = reports[key]
            from scripts.paper_result_table import NAMES
            model = NAMES[report["base_model_id"]]
            stage = "RL" if report["training"]["configuration"].get("initialization") == "base" else "SFT$\\to$RL"
            additions.append(f"{model} reaches {accuracy(key):.1f}\\% with {stage}.")
            if report["checkpoint"]["state"]["best_group"] == 0:
                additions.append("Validation retains that run's initial policy.")
        if additions:
            parts.append(" ".join(additions))
        final_scores = []
        for size,model in (("2b","2B"),("8b","8B")):
            key = "qwen3_"+size+"_grpo_final"
            if key in reports:
                final_scores.append(f"{model}: {accuracy(key):.1f}\\%")
        if final_scores:
            parts.append(
                "A separate diagnostic evaluates the fixed final checkpoint at group 64 ("
                + "; ".join(final_scores) + "). "
                "Table~\\ref{tab:sft} reports these fixed endpoints in the Final column alongside the validation-selected checkpoints. "
                "For 8B, the validation set retains the initial policy even though the final policy scores higher on the paper cohort. "
                "The supplementary material reports checkpoint choice and acquisition behavior together.")
        parts.append("The supplementary acquisition analysis distinguishes answer accuracy from requests for missing photographs.")
    return "\n\n".join(parts)


def detail_text(entries):
    lines = ["\\section{Training Data and Action Supervision}",
             "\\input{tables/training_ablation}",
             "Identity-only SFT uses CO3D trajectories; mixed-task SFT adds balanced condition supervision. "
             "The paired finish-only controls retain public histories but remove acquisition-action targets. "
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
                +"A fixed held-out set of eight source families across four access versions selects the checkpoint. "
                +"The benchmark supplies no training or checkpoint-selection feedback. "
                +f"The run performs {report['checkpoint']['state']['optimizer_updates']} optimizer updates "
                +f"and selects group {report['checkpoint']['state']['best_group']}; group zero denotes the initial SFT adapter.")
        else:
            checkpoint = report["checkpoint"]["state"]
            lines.append(entry["description"] + f" completes {checkpoint['step']} optimizer steps, "
                         +"using the same rank-32 attention adapters and held-out token-loss selection as the other SFT runs. "
                         +"Its data and exact selected checkpoint are recorded with the released comparison.")
            if "terminal" in report["models"][1]["name"].lower():
                lines.append("Finish-only retains complete public histories at the final decision, "
                             "but removes acquisition-action prediction targets. It uses the same update limit, "
                             "not matched target-token compute, as the corresponding full-trajectory SFT.")
    lines.append("All rows retain the fixed paper cohort, failure accounting and inference allowances. "
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
                (stage/name.removeprefix("paper/")).write_text(content)
        environment = {**os.environ,"XDG_CACHE_HOME":str(runtime/"tools/tex/cache")}
        for source in ("main.tex","supplementary.tex"):
            run([str(runtime/"tools/tex/tectonic"),"--outdir",temp,source],stage,environment)
        sys.path.insert(0,str(runtime/"tools/pdf"))
        import fitz
        for name in ("main.pdf","supplementary.pdf"):
            with fitz.open(Path(temp)/name) as doc:
                text = "\n".join(p.get_text() for p in doc)
                if "[?]" in text or "??" in text or "\ufffd" in text or doc.metadata.get("author"):
                    raise ValueError("Compiled adaptation paper failed reference or anonymity checks")
        for name,content in changes.items():
            (repo/name).write_text(content)
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
