"""Resume the known GPT-4.1 mini client interruption after the main API suite."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from resolveai.api_budget import ApiBudget, BudgetStop
from resolveai.api_resume import row_digests
from scripts.score_api_results import refresh
from scripts.publish_api_results import publish

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    args=parser.parse_args()
    if os.getenv("SLURMD_NODENAME")!="macula":
        parser.error("Use the retained macula allocation")
    repo,runtime=args.repository,args.runtime
    root=runtime/"evaluation/openai"
    output=root/"gpt41_mini"
    status_path=output/"recovery_status.json"
    lock=(root/"gpt41-mini-recovery.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**fields):
        status_path.write_text(json.dumps({"phase":phase,**fields},indent=2)+"\n")
    original=root/"ledgers/gpt41_mini.sqlite"
    with sqlite3.connect(original) as db:
        unknown=db.execute("SELECT COUNT(*) FROM calls WHERE charge IS NULL AND response IS NULL").fetchone()[0]
        cached=db.execute("SELECT key,response FROM calls WHERE charge IS NOT NULL AND response IS NOT NULL").fetchall()
    if unknown!=1 or "Usage exceeded reservation" not in (output/"run.log").read_text():
        raise RuntimeError("This recovery is restricted to the one documented lost client receipt")
    # One discarded receipt has unknown usage. Reserve its entire published
    # 4.1-mini context/output ceiling (1,047,576 input + 32,768 output tokens)
    # at the audited rates, rounded upward to USD 1.11. This is a conservative
    # accounting guard, not an asserted API bill.
    parent=ApiBudget(root/"budget.sqlite")
    guard_name="gpt41_mini-lost-receipt-guard"
    guard_path=root/"ledgers"/(guard_name+".sqlite")
    status("waiting_for_receipt_reserve",preserved_answers=197)
    while True:
        try:
            guard=parent.allocate(guard_name,guard_path,1.11)
            break
        except BudgetStop:
            time.sleep(10)
    if guard is not None:
        ticket,_=guard.reserve("audit/lost-original-receipt",1.11,
                               stage="reconciliation",stage_limit_usd=1.11)
        if ticket:
            guard.settle(ticket,1.11,{"kind":"conservative_lost_receipt_guard",
                                      "not_actual_billing":True})
        guard.close()
        parent.close_allocation(guard_name,guard_path)
    status("waiting_for_main_suite",preserved_answers=197,lost_receipt_guard_usd=1.11)
    # Preserve the suite's three-worker limit and existing prepaid priorities.
    while json.loads((root/"suite_status.json").read_text())["phase"]!="finished":
        time.sleep(20)
    name="gpt41_mini-client-fix"
    child_path=root/"ledgers"/(name+".sqlite")
    cap=5.0
    try:
        child=parent.allocate(name,child_path,cap)
    except BudgetStop:
        status("insufficient_remaining_budget",preserved_answers=197,requested_allocation_usd=cap)
        parent.close()
        return
    if child is None:
        parent.close()
        return
    with child.db:
        for key,response in cached:
            # The original paid call remains charged in its sealed ledger.
            child.db.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?)",
                             ("cache/"+key,key,"original-paid-cache",0,0,response))
    child.close()
    plan=runtime/"evaluation/cvpr-20261101/plan.json"
    command=[sys.executable,"-u","-m","scripts.evaluate_openai",
             str(plan),str(repo/"docs/results/cvpr_20261102_candidate_scores.json"),str(output),
             "--config",str(root/"specs/gpt41_mini.json"),"--ledger",str(child_path),
             "--budget-usd",str(cap),"--stage-budget-usd",str(cap),
             "--stage-name","paper/"+name,"--stage","paper",
             "--grounding-root",str(runtime/"grounding"),
             "--key-file",str(runtime/"credentials/openai.env"),
             "--resume-after-client-fix","--execute"]
    before=set(row_digests(output/"episodes.sqlite"))
    status("running",preserved_answers=len(before),allocation_usd=cap,
           lost_receipt_guard_usd=1.11,known_response_cache=len(cached))
    try:
        environment={**os.environ,"PYTHONPATH":str(repo/"src")+":"+str(repo),
                     "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
        with (output/"recovery.log").open("a") as log:
            result=subprocess.run(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if not before<=set(row_digests(output/"episodes.sqlite")):
            raise RuntimeError("Previously completed answers changed")
        status("complete" if result.returncode==0 else "stopped",exit_code=result.returncode,
               preserved_answers=len(before),lost_receipt_guard_usd=1.11)
    finally:
        parent.close_allocation(name,child_path)
        parent.close()
    report=refresh(repo,root,plan)
    publish(repo,report,root/"paper-publish.lock")
    # The suite watcher may have exited before this continuation completes.
    from scripts.sync_api_paper import synchronize
    synchronize(repo,runtime,"6ae80679a4c7a64df04ac5b8")

if __name__=="__main__":
    main()
