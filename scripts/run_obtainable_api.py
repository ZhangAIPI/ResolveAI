"""Complete only missing Obtainable API episodes under a separate USD20 reserve.

The existing study ledger is capped at USD60. This additional ledger is capped
at USD20, so both ledgers together cannot exceed the user's USD80 authorization.
Original successful episodes and image files are copied byte-for-byte, not rerun.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
import threading

from resolveai.api_budget import ApiBudget
from scripts.evaluate_openai import select_cases

PROFILES = (("gpt61_sol", 11), ("terra-eval", 3), ("luna", 1),
            ("gpt41_mini", 2), ("gpt54_mini", 3))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve_obtainable(source, output, families):
    """Import complete saved rows and only their referenced immutable images."""
    rows = []
    database = source / "episodes.sqlite"
    if database.exists():
        with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)) as db:
            for cid, arm, raw, trace_raw in db.execute("SELECT * FROM episodes"):
                row = json.loads(raw)
                if row["variant"] == "Obtainable" and arm == "interactive_agent" and row["family_id"] in families:
                    rows.append((cid, arm, raw, trace_raw))
    output.mkdir(exist_ok=True, parents=True)
    with closing(sqlite3.connect(output / "episodes.sqlite")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS episodes (case_id TEXT, arm TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,arm))")
        for cid, arm, raw, trace_raw in rows:
            old = db.execute("SELECT result,trace FROM episodes WHERE case_id=? AND arm=?", (cid,arm)).fetchone()
            if old is not None and old != (raw, trace_raw):
                raise ValueError("A preserved episode changed")
            db.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,?,?)", (cid,arm,raw,trace_raw))
            trace = json.loads(trace_raw)
            for message in trace.get("messages", []):
                content = message.get("content", [])
                for block in content if isinstance(content,list) else []:
                    if block.get("type") != "image" or not block.get("image_ref"):
                        continue
                    relative = Path(block["image_ref"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Expected a relative saved image reference")
                    src, dest = source/relative, output/relative
                    if not src.is_file():
                        raise ValueError("Missing original trajectory image")
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if sha(dest) != sha(src):
                            raise ValueError("Conflicting trajectory image")
                    else:
                        try:
                            os.link(src,dest)
                        except OSError:
                            shutil.copy2(src,dest)
        db.commit()
    payload = sorted((cid,arm,hashlib.sha256(raw.encode()).hexdigest(),
                      hashlib.sha256(tr.encode()).hexdigest()) for cid,arm,raw,tr in rows)
    return {"source_directory":str(source),"preserved_episodes":len(rows),
            "preserved_sha256":hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            "original_manifest_sha256":sha(source/"manifest.json") if (source/"manifest.json").exists() else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository",type=Path)
    p.add_argument("runtime",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    repo, rt, out=args.repository.resolve(),args.runtime.resolve(),args.output.resolve()
    if os.environ.get("SLURMD_NODENAME") != "macula":
        p.error("Run on the retained compute node")
    old=rt/"evaluation/openai"
    with closing(sqlite3.connect((old/"budget.sqlite").resolve().as_uri()+"?mode=ro",uri=True)) as db:
        if db.execute("SELECT cap FROM budget").fetchone() != (60_000_000,):
            raise ValueError("Original ledger is not capped at USD60")
        old_upper=db.execute("SELECT coalesce(sum(coalesce(charge,bound)),0) FROM calls").fetchone()[0]/1e6
    cohort_path=repo/"docs/results/cvpr_20261102_candidate_scores.json"
    plan_path=rt/"evaluation/cvpr-20261101/plan.json"
    cohort=json.loads(cohort_path.read_text())
    plan=json.loads(plan_path.read_text())
    cases=select_cases(plan,cohort,"paper",["Obtainable"])
    if len(cases)!=209 or sum(c["task"]=="state" for c in cases)!=153:
        raise ValueError("Expected the fixed 209-family cohort")
    preview={"purpose":"Complete missing interactive Obtainable episodes only",
             "old_ledger_cap_usd":60,"old_accounted_upper_usd":old_upper,
             "additional_cap_usd":20,"combined_hard_cap_usd":80,
             "profiles":dict(PROFILES),"families":209,"tasks":{"state":153,"identity":56},
             "variants":["Obtainable"],"arms":["interactive_agent"],
             "formal_grounded_accuracy":None,"execute":args.execute}
    print(json.dumps(preview,indent=2),flush=True)
    if not args.execute:
        return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"suite.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_lock=threading.Lock()
    status={**preview,"phase":"running","models":{}}
    def update(name,**fields):
        with status_lock:
            status["models"].setdefault(name,{}).update(fields)
            tmp=out/"status.tmp"
            tmp.write_text(json.dumps(status,indent=2)+"\n")
            tmp.replace(out/"status.json")
    def evaluate(name,cap):
        output=out/name
        if name=="terra-eval":
            config=json.loads((old/name/"manifest.json").read_text())["configuration"]["api_config"]
        else:
            config=json.loads((old/"specs"/(name+".json")).read_text())
        imported=preserve_obtainable(old/name,output,set(cohort["family_ids"]))
        audit_path=output/"continuation.json"
        audit={**imported,"reason":"Fill the missing Obtainable cohort without rerunning successful episodes",
               "runner_sha256":sha(Path(__file__)),"api_config":config,
               "execution_source":str(repo),"old_ledger":str(old/"budget.sqlite"),
               "additional_parent":str(out/"budget.sqlite")}
        if audit_path.exists() and json.loads(audit_path.read_text())!=audit:
            raise ValueError("Continuation settings changed")
        audit_path.write_text(json.dumps(audit,indent=2)+"\n")
        spec=out/(name+".json")
        encoded=json.dumps(config,indent=2)+"\n"
        if spec.exists() and spec.read_text()!=encoded:
            raise ValueError("API settings changed")
        spec.write_text(encoded)
        parent=ApiBudget(out/"budget.sqlite",20)
        ledger_path=out/(name+".sqlite")
        child=parent.allocate(name,ledger_path,cap)
        if child is None:
            update(name,phase="previous_allocation_closed",**imported)
            parent.close()
            return
        child.close()
        update(name,phase="running",allocation_usd=cap,**imported)
        command=[sys.executable,"-u","-m","scripts.evaluate_openai",
                 str(plan_path),str(cohort_path),str(output),
                 "--config",str(spec),"--ledger",str(ledger_path),
                 "--budget-usd",str(cap),"--stage-budget-usd",str(cap),
                 "--stage-name","reviewer-obtainable/"+name,
                 "--stage","paper","--variants","Obtainable","--agent-only",
                 "--grounding-root",str(rt/"grounding"),
                 "--key-file",str(rt/"credentials/openai.env"),"--execute"]
        code=None
        try:
            env={**os.environ,"PYTHONPATH":str(repo/"src")+":"+str(repo),
                 "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            with (output/"run.log").open("a") as log:
                code=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            after=preserve_obtainable(old/name,output,set(cohort["family_ids"]))
            if after!=imported:
                raise ValueError("Preserved original answers changed")
        finally:
            closed=parent.close_allocation(name,ledger_path)
            parent.close()
            summary=json.loads((output/"summary.json").read_text()) if (output/"summary.json").exists() else {}
            update(name,phase="complete" if code==0 and summary.get("complete") else "stopped",
                   exit_code=code,episodes=summary.get("episodes"),
                   stop_reason=summary.get("stop_reason"),accounted_upper_usd=closed["accounted_upper_usd"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(evaluate,name,cap) for name,cap in PROFILES]
        for future in futures:
            future.result()
    status["phase"]="finished"
    update("_suite",phase="finished")


if __name__=="__main__":
    main()
