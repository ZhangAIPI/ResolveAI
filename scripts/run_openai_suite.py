"""Parallel API cohorts using prepaid allocations inside one USD 60 parent ledger."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

from resolveai.api_budget import ApiBudget,BudgetStop
from scripts.score_api_results import refresh
from scripts.publish_api_results import publish


def write_status(path, status):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status,indent=2)+"\n")
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("cohort",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("--repository",type=Path,required=True)
    p.add_argument("--catalog",type=Path,default=Path("docs/experiments/openai-models.json"))
    p.add_argument("--key-file",type=Path,required=True)
    p.add_argument("--grounding-root",type=Path,required=True)
    p.add_argument("--parallel",type=int,default=3)
    args=p.parse_args()
    if os.getenv("SLURMD_NODENAME")!="macula" or not 1<=args.parallel<=3:
        p.error("Use the retained macula allocation and at most three API workers")
    args.root.mkdir(parents=True,exist_ok=True)
    suite_lock=(args.root/"suite.lock").open("w")
    try:
        fcntl.flock(suite_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An API suite is already running")
    catalog=json.loads(args.catalog.read_text())
    status={"phase":"running","models":[],"shared_cap_usd":60,
            "scope":"Complete fixed Table 1 cohorts first; allocations are not actual spending"}
    status_lock=threading.Lock()
    specs=args.root/"specs"
    specs.mkdir(exist_ok=True)
    ledgers=args.root/"ledgers"
    ledgers.mkdir(exist_ok=True)

    def update(state, **fields):
        with status_lock:
            state.update(fields)
            write_status(args.root/"suite_status.json",status)

    def evaluate(profile,state):
        name=profile["name"]
        cap=profile["allocation_usd"]
        child_path=ledgers/(name+".sqlite")
        parent=ApiBudget(args.root/"budget.sqlite")
        try:
            child=parent.allocate(name,child_path,cap)
        except BudgetStop:
            parent.close()
            update(state,status="insufficient_unallocated_budget")
            return
        if child is None:
            parent.close()
            update(state,status="allocation_already_closed")
            return
        child.close()
        config_path=specs/(name+".json")
        encoded=json.dumps(profile["config"],indent=2)+"\n"
        if config_path.exists() and config_path.read_text()!=encoded:
            parent.close()
            raise ValueError("Queued model configuration changed")
        config_path.write_text(encoded)
        try:
            for stage in ("pilot","paper"):
                update(state,phase=stage,status="running",allocation_usd=cap)
                output=args.root/(name+"-pilot" if stage=="pilot" else name)
                command=[sys.executable,"-u","-m","scripts.evaluate_openai",
                         str(args.plan),str(args.cohort),str(output),
                         "--config",str(config_path),"--ledger",str(child_path),
                         "--budget-usd",str(cap),"--grounding-root",str(args.grounding_root),
                         "--key-file",str(args.key_file),"--stage",stage,
                         "--stage-name",stage+"/"+name,"--stage-budget-usd",str(min(.8,cap) if stage=="pilot" else cap),
                         "--execute"]
                output.mkdir(exist_ok=True)
                environment={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
                with (output/"run.log").open("a") as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
                if result.returncode:
                    update(state,status="stopped",exit_code=result.returncode)
                    return
                summary=json.loads((output/"summary.json").read_text())
                if not summary["complete"]:
                    update(state,status="incomplete")
                    return
            update(state,status="complete")
        finally:
            # The child is sealed before unused money is returned to the parent.
            closed=parent.close_allocation(name,child_path)
            parent.close()
            update(state,accounted_upper_usd=closed["accounted_upper_usd"])

    futures=[]
    with ThreadPoolExecutor(args.parallel) as pool:
        for profile in catalog["profiles"]:
            state={"name":profile["name"],"status":"queued"}
            status["models"].append(state)
            futures.append(pool.submit(evaluate,profile,state))
        while any(not future.done() for future in futures):
            with status_lock:
                for state in status["models"]:
                    directory=args.root/state["name"]
                    db_path=directory/"episodes.sqlite"
                    if db_path.exists():
                        with sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro",uri=True) as db:
                            state["paper_attempts"]=db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                write_status(args.root/"suite_status.json",status)
            # Writes public numbers only when an entire fixed cohort becomes ready.
            report=refresh(args.repository,args.root,args.plan)
            if report["workers"]:
                publish(args.repository,report,args.root/"paper-publish.lock")
            time.sleep(10)
        for future in futures:
            future.result()
    report=refresh(args.repository,args.root,args.plan)
    if report["workers"]:
        publish(args.repository,report,args.root/"paper-publish.lock")
    status["phase"]="finished"
    write_status(args.root/"suite_status.json",status)


if __name__=="__main__":
    main()
