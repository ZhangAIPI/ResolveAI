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
        return False
    wanted = {(family,variant,"interactive_agent") for family in families
              for variant in ("Sufficient","Obtainable","Missing","Unavailable")}
    wanted |= {(family,"Sufficient",arm) for family in families
               for arm in ("initial_text","full_available_text")}
    found = set()
    for path in episode_databases(directory):
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
    paragraphs = []
    if any("groups" in report["training"]["configuration"] for _,report in entries):
        paragraphs.append(
            "\\paragraph{Online policy optimization.}\n"
            "GRPO continues each mixed-task SFT adapter using sampled tool interactions and source-derived rewards. "
            "The reference adapter stays frozen; validation families select the checkpoint. "
            "We use 64 groups of four trajectories per model, with training details in the supplement.")
    for entry,report in entries:
        if not entry.get("new_analysis"):
            continue
        base,adapted = report["models"]
        direct = adapted["arms"]["initial_text"]["label_agreement"]*100
        tools = adapted["arms"]["interactive_agent"]["label_agreement"]*100
        before = base["arms"]["interactive_agent"]["label_agreement"]*100
        requests = report["acquisition_behavior"][1]["by_variant"]["Obtainable"]["episodes_with_request"]
        new_photos = report["acquisition_behavior"][1]["by_variant"]["Obtainable"]["episodes_with_new_photo"]
        paragraph = (entry["description"] + f" scores {direct:.1f}\\% without tools and {tools:.1f}\\% with tools, "
                     +f"compared with {before:.1f}\\% for the base with tools. ")
        if entry.get("parent_report"):
            parent = json.loads(Path(entry["parent_report"]).read_text())
            previous = parent["models"][1]["arms"]["interactive_agent"]["label_agreement"]*100
            paragraph += f"Its SFT initialization scores {previous:.1f}\\% with tools on the same cases. "
        paragraph += (f"In Obtainable, {requests} cases trigger a photograph request and {new_photos} receive a new photograph. "
                      "These action counts alone do not establish that the requested evidence is useful.")
        paragraphs.append(paragraph)
    if not paragraphs:
        return ""
    if paragraphs[0].startswith("\\paragraph{Online"):
        return paragraphs[0]+"\n\n\\paragraph{Further adaptation results.}\n"+"\n\n".join(paragraphs[1:])
    return "\\paragraph{Further adaptation results.}\n"+"\n\n".join(paragraphs)


def detail_text(entries):
    lines = ["\\section{Additional Adaptation Runs}"]
    for entry,report in entries:
        if not entry.get("new_analysis"):
            continue
        config = report["training"]["configuration"]
        if "groups" in config:
            lines.append(
                entry["description"] + f" continues mixed-task SFT for {config['groups']} sampled groups "
                +f"of {config['group_size']} trajectories, with two clipped policy updates per nonconstant-reward group. "
                +f"The learning rate is \\({config['learning_rate']:g}\\), clipping threshold 0.2 and reference penalty 0.02. "
                +"The reference is the frozen SFT adapter. Training sees only public observations; private source annotations enter as scalar rewards. "
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
    if run(["git","status","--porcelain"],repo).strip():
        raise RuntimeError("Repository has independent edits; measured results retained for reconciliation")
    table = render_adaptations([report for _,report in entries])
    changes = {"paper/tables/sft_results.tex":table}
    for entry,report in entries:
        changes[entry["public_report"]] = json.dumps(report,indent=2)+"\n"
    body = result_text(entries)
    if body:
        changes["paper/main.tex"] = replace_block((repo/"paper/main.tex").read_text(),body)
        changes["paper/appendix.tex"] = replace_block((repo/"paper/appendix.tex").read_text(),detail_text(entries))
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
                if "[?]" in text or "\ufffd" in text or doc.metadata.get("author"):
                    raise ValueError("Compiled adaptation paper failed reference or anonymity checks")
        for name,content in changes.items():
            (repo/name).write_text(content)
    run(["git","diff","--check"],repo)
    if run(["git","diff","--name-only"],repo).strip():
        run(["git","add",*changes],repo)
        run(["git","commit","-m","Update Table 2 with complete paired adaptation measurements"],repo)
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
