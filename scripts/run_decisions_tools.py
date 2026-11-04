"""Run Decisions action selection within the existing shared USD 60 budget."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from resolveai.api_budget import ApiBudget,BudgetStop
from scripts.score_api_results import refresh
from scripts.publish_api_results import publish
from scripts.sync_api_paper import synchronize

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    args=parser.parse_args()
    if os.getenv("SLURMD_NODENAME")!="macula":
        parser.error("Use the retained GPU node")
    repo,runtime=args.repository,args.runtime
    code=Path(__file__).resolve().parents[1]
    root=runtime/"evaluation/openai"
    lock=(root/"decisions-tools.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=root/"decisions-tools-status.json"
    def status(phase,**fields):
        status_path.write_text(json.dumps({"phase":phase,**fields},indent=2)+"\n")
        print(json.dumps({"phase":phase,**fields}),flush=True)
    name="luna_decisions_tools"
    cap=3.8
    ledger_path=root/"ledgers"/(name+".sqlite")
    parent=ApiBudget(root/"budget.sqlite")
    try:
        ledger=parent.allocate(name,ledger_path,cap)
    except BudgetStop:
        status("insufficient_remaining_budget")
        parent.close()
        return
    if ledger is None:
        status("allocation_already_closed")
        parent.close()
        return
    ledger.close()
    catalog=json.loads((repo/"docs/experiments/openai-models.json").read_text())
    config=next(p["config"] for p in catalog["profiles"] if p["name"]=="luna_decisions")
    config={**config,"decision_tools":"public-menu-v1"}
    spec=root/"specs"/(name+".json")
    encoded=json.dumps(config,indent=2)+"\n"
    if spec.exists() and spec.read_text()!=encoded:
        raise ValueError("Menu configuration changed")
    spec.write_text(encoded)
    plan=runtime/"evaluation/cvpr-20261101/plan.json"
    common=[sys.executable,"-u","-m","scripts.evaluate_openai",str(plan),
            str(repo/"docs/results/cvpr_20261102_candidate_scores.json")]
    options=["--config",str(spec),"--ledger",str(ledger_path),"--budget-usd",str(cap),
             "--grounding-root",str(runtime/"grounding"),
             "--key-file",str(runtime/"credentials/openai.env"),"--agent-only","--execute"]
    environment={**os.environ,"PYTHONPATH":str(code/"src")+":"+str(code),
                 "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    def evaluate(phase,directory,stage,extra=()):
        output=root/directory
        output.mkdir(exist_ok=True)
        status(phase,output=str(output),shared_cap_usd=60,allocation_usd=cap)
        command=[*common,str(output),*options,"--stage",stage,
                 "--stage-name",name+"/"+phase,"--stage-budget-usd",str(.3 if stage=="pilot" else cap),*extra]
        with (output/"run.log").open("a") as log:
            result=subprocess.run(command,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT)
        summary=json.loads((output/"summary.json").read_text()) if (output/"summary.json").exists() else {}
        return result.returncode==0 and summary.get("complete",False),summary
    try:
        # Four deterministic families, all access variants. This checks real
        # image/function-output compatibility; it does not select cases by success.
        ok,summary=evaluate("pilot",name+"-pilot","pilot",["--all-variants"])
        if not ok:
            status("pilot_stopped",summary=summary)
            return
        ok,summary=evaluate("paper",name,"paper")
        if not ok:
            status("paper_incomplete",summary=summary)
            return
        report=refresh(repo,root,plan)
        publish(repo,report,root/"paper-publish.lock")
        synchronize(repo,runtime,"6ae80679a4c7a64df04ac5b8")
        # An initially incomplete access condition tests actual photo acquisition.
        # It consumes only what remains of the same prepaid allocation.
        ok,summary=evaluate("obtainable",name+"-obtainable","paper",["--variants","Obtainable"])
        status("complete" if ok else "main_complete_extension_stopped",
               primary_episodes=209,obtainable_summary=summary)
    finally:
        descriptor=parent.close_allocation(name,ledger_path)
        parent.close()
        print(json.dumps({"allocation_closed":descriptor}),flush=True)

if __name__=="__main__":
    main()
