"""One audited continuation of the documented Decisions HTTP 503 interruption."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from resolveai.api_budget import ApiBudget
from resolveai.api_resume import row_digests

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    parser.add_argument("--continue-existing",action="store_true",help="Continue the same funded ledger after an externally terminated job step")
    args=parser.parse_args()
    if os.getenv("SLURMD_NODENAME")!="macula":
        parser.error("Use the retained GPU node")
    repo,runtime=args.repository,args.runtime
    root=runtime/"evaluation/openai"
    lock=(root/"decisions-access-recovery.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=root/"luna_decisions_tools-obtainable"
    output=root/"luna_decisions_tools-obtainable-resume1"
    continuing=output.exists()
    if continuing and not args.continue_existing:
        raise RuntimeError("This continuation exists; inspect it and pass --continue-existing to resume")
    if args.continue_existing and not continuing:
        raise RuntimeError("No existing continuation to resume")
    summary=json.loads((old/"summary.json").read_text())
    manifest=json.loads((old/"manifest.json").read_text())
    if summary["complete"] or "OpenAI HTTP 503" not in (old/"run.log").read_text():
        raise RuntimeError("Recovery is restricted to the documented 503 interruption")
    source=root/"code-decisions-tools"
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    if commit!=manifest["source_commit"] or subprocess.check_output(
            ["git","status","--porcelain"],cwd=source,text=True).strip():
        raise RuntimeError("The original inference source must remain unchanged")
    parent=ApiBudget(root/"budget.sqlite")
    descriptor=json.loads(parent.db.execute("SELECT response FROM calls WHERE key=?",
                                            ("allocation/luna_decisions_tools",)).fetchone()[0])
    if descriptor["state"]!="closed":
        raise RuntimeError("Original worker is not sealed")
    with sqlite3.connect(descriptor["ledger"]) as db:
        unresolved=db.execute("SELECT key,bound FROM calls WHERE charge IS NULL").fetchall()
        known=db.execute("SELECT key,response FROM calls WHERE charge IS NOT NULL AND response IS NOT NULL").fetchall()
    if len(unresolved)!=1:
        raise RuntimeError("Expected one unresolved failed request")
    name="decisions-access-resume1"
    cap=2.0
    ledger_path=root/"ledgers"/(name+".sqlite")
    child=parent.allocate(name,ledger_path,cap)
    if child is None:
        parent.close()
        raise RuntimeError("Recovery allocation is already closed")
    if child.db.execute("SELECT 1 FROM calls WHERE charge IS NULL").fetchone():
        child.close()
        parent.close()
        raise RuntimeError("Unresolved spending must be reconciled before continuing")
    if not continuing:
        with child.db:
            for key,response in known:
                child.db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?)",
                                 ("cache/"+key,key,"original-paid-cache",0,0,response))
    child.close()
    if not continuing:
        output.mkdir()
        with sqlite3.connect(old/"episodes.sqlite") as original, sqlite3.connect(output/"episodes.sqlite") as copied:
            original.backup(copied)
        if (old/"images").exists():
            shutil.copytree(old/"images",output/"images",copy_function=os.link)
    preserved=set(row_digests(output/"episodes.sqlite"))
    audit={"reason":"One explicit continuation after HTTP 503; no automatic network retry",
           "previous_manifest":str(old/"manifest.json"),
           "previous_manifest_sha256":hashlib.sha256((old/"manifest.json").read_bytes()).hexdigest(),
           "inference_source_unchanged":commit,"preserved_episodes":len(preserved),
           "original_ledger":descriptor["ledger"],"cached_known_responses":len(known),
           "unresolved_old_call_bound_usd":sum(row[1] for row in unresolved)/1e6,
           "note":"The old unresolved reservation remains counted; reused responses are charged in the old ledger."}
    if continuing:
        audit=json.loads((output/"continuation.json").read_text())
    audit.setdefault("resume_segments",[]).append({
        "slurm_job":os.getenv("SLURM_JOB_ID"),
        "preserved_episodes":len(preserved),
        "preserved_hashes_sha256":hashlib.sha256(json.dumps(sorted(preserved)).encode()).hexdigest(),
        "same_funded_ledger":str(ledger_path)})
    (output/"continuation.json").write_text(json.dumps(audit,indent=2)+"\n")
    status_path=root/"decisions-tools-status.json"
    def status(phase,**fields):
        status_path.write_text(json.dumps({"phase":phase,"primary_episodes":209,
                              "obtainable_output":str(output),"preserved_episodes":len(preserved),"slurm_job":os.getenv("SLURM_JOB_ID"),**fields},indent=2)+"\n")
    status("obtainable_recovery")
    command=[sys.executable,"-u","-m","scripts.evaluate_openai",
             str(runtime/"evaluation/cvpr-20261101/plan.json"),
             str(repo/"docs/results/cvpr_20261102_candidate_scores.json"),str(output),
             "--config",str(root/"specs/luna_decisions_tools.json"),
             "--ledger",str(ledger_path),"--budget-usd",str(cap),
             "--stage-budget-usd",str(cap),"--stage-name",name,"--stage","paper",
             "--variants","Obtainable","--agent-only",
             "--grounding-root",str(runtime/"grounding"),
             "--key-file",str(runtime/"credentials/openai.env"),"--execute"]
    environment={**os.environ,"PYTHONPATH":str(source/"src")+":"+str(source),
                 "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    try:
        with (output/"run.log").open("a") as log:
            result=subprocess.run(command,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if not preserved<=set(row_digests(output/"episodes.sqlite")):
            raise RuntimeError("A copied completed answer changed")
        summary=json.loads((output/"summary.json").read_text())
        status("complete" if result.returncode==0 and summary["complete"] else "recovery_stopped",
               obtainable_summary=summary)
    finally:
        closed=parent.close_allocation(name,ledger_path)
        parent.close()
        print(json.dumps({"closed":closed}),flush=True)

if __name__=="__main__":
    main()
