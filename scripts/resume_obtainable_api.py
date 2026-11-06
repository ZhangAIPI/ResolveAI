"""Finish only missing mini-model episodes using the existing USD20 parent reserve."""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from resolveai.api_budget import ApiBudget
from resolveai.api_resume import answer_snapshot, verify_snapshot
from scripts.evaluate_openai import select_cases

PROFILES = (("gpt54_mini", .25), ("gpt41_mini", 3.35))


def ledger_status(path, expected_cap):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)) as db:
        cap = db.execute("SELECT cap FROM budget").fetchone()[0]
        if cap != expected_cap * 1_000_000:
            raise ValueError("Shared ledger cap changed")
        used = db.execute("SELECT COALESCE(SUM(COALESCE(charge,bound)),0) FROM calls").fetchone()[0]
        pending = db.execute("SELECT COUNT(*) FROM calls WHERE charge IS NULL").fetchone()[0]
    if pending:
        raise ValueError("Unresolved shared reservation blocks continuation")
    return {"cap_usd": cap/1e6, "accounted_upper_usd": used/1e6,
            "available_usd": (cap-used)/1e6}


def saved_count(directory):
    with closing(sqlite3.connect((directory/"episodes.sqlite").resolve().as_uri()+"?mode=ro", uri=True)) as db:
        return db.execute("SELECT COUNT(*) FROM episodes WHERE arm='interactive_agent' AND case_id LIKE '%-Obtainable'").fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo, runtime = args.repository.resolve(), args.runtime.resolve()
    root = runtime/"evaluation/cvpr-paper-priority/reviewer-api-obtainable"
    prior_code = runtime/"evaluation/cvpr-paper-priority/reviewer-api-code-v1"
    if os.environ.get("SLURMD_NODENAME") != "macula":
        parser.error("Run on the retained macula compute node")
    old_ledger = runtime/"evaluation/openai/budget.sqlite"
    initial_old = ledger_status(old_ledger, 60)
    initial_parent = ledger_status(root/"budget.sqlite", 20)
    if initial_old["accounted_upper_usd"] + initial_parent["accounted_upper_usd"] > 80:
        raise ValueError("Combined budget exceeds authorization")
    cohort_path = repo/"docs/results/cvpr_20261102_candidate_scores.json"
    plan_path = runtime/"evaluation/cvpr-20261101/plan.json"
    cohort, plan = json.loads(cohort_path.read_text()), json.loads(plan_path.read_text())
    cases = select_cases(plan, cohort, "paper", ["Obtainable"])
    if len(cases) != 209 or sum(c["task"] == "state" for c in cases) != 153:
        raise ValueError("Frozen cohort changed")
    preview = {"purpose": "Budget-only continuation; preserve completed answers",
               "parent_cap_usd": 20, "combined_cap_usd": 80,
               "initial_combined_accounted_upper_usd": initial_old["accounted_upper_usd"] + initial_parent["accounted_upper_usd"],
               "allocations_usd": dict(PROFILES),
               "missing_episodes": {name: 209-saved_count(root/name) for name, _ in PROFILES},
               "execute": args.execute}
    print(json.dumps(preview), flush=True)
    if not args.execute:
        return
    with (root/"suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path = root/"budget-resume-20261106-status.json"
        state = {**preview, "phase": "running", "slurm_job": os.environ.get("SLURM_JOB_ID"),
                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                 "models": {}}

        def update(name, **fields):
            state["models"].setdefault(name, {}).update(fields)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2)+"\n")
            temporary.replace(state_path)

        for name, cap in PROFILES:
            output = root/name
            allocation = name+"-budget-resume-20261106"
            child_path = root/(allocation+".sqlite")
            before = answer_snapshot(output)
            before_digest = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            parent = ApiBudget(root/"budget.sqlite", 20)
            child = parent.allocate(allocation, child_path, cap)
            if child is None:
                update(name, phase="previous_allocation_closed", episodes=saved_count(output))
                parent.close()
                continue
            code, verified, error_name = None, False, None
            try:
                cache = child.import_paid_cache(root/(name+".sqlite"))
                child.close()
                update(name, phase="running", allocation_usd=cap, initial_episodes=len(before["rows"]),
                       previous_answers_sha256=before_digest, cache=cache)
                command = [sys.executable, "-u", "-m", "scripts.evaluate_openai",
                           str(plan_path), str(cohort_path), str(output),
                           "--config", str(root/(name+".json")), "--ledger", str(child_path),
                           "--budget-usd", str(cap), "--stage-budget-usd", str(cap),
                           "--stage-name", allocation, "--stage", "paper", "--variants", "Obtainable",
                           "--agent-only", "--resume-budget-from", str(prior_code),
                           "--grounding-root", str(runtime/"grounding"),
                           "--key-file", str(runtime/"credentials/openai.env"), "--execute"]
                env = {**os.environ, "PYTHONPATH": str(repo/"src")+":"+str(repo),
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                with (output/"budget-resume-20261106.log").open("a") as log:
                    code = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                verify_snapshot(output, before)
                verified = True
            except Exception as error:
                error_name = type(error).__name__
                raise
            finally:
                child.close()
                closed = parent.close_allocation(allocation, child_path)
                parent.close()
                previous = ledger_status(old_ledger, 60)
                additional = ledger_status(root/"budget.sqlite", 20)
                if previous != initial_old:
                    raise ValueError("Original study ledger changed during continuation")
                combined = previous["accounted_upper_usd"] + additional["accounted_upper_usd"]
                if combined > 80:
                    raise ValueError("Combined budget exceeded authorization")
                summary_path = output/"summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                episodes = saved_count(output)
                update(name, phase="complete" if code == 0 and episodes == 209 and verified else "stopped",
                       exit_code=code, episodes=episodes, missing_episodes=209-episodes,
                       stop_reason=error_name or summary.get("stop_reason"),
                       additional_accounted_upper_usd=closed["accounted_upper_usd"],
                       combined_accounted_upper_usd=combined,
                       original_answers_unchanged=verified)
        state["phase"] = "finished"
        update("_suite", phase="finished")

if __name__ == "__main__":
    main()
