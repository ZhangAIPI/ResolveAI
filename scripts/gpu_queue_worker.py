"""Keep an allocated GPU available while running a resumable experiment queue."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("lane")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("An allocated GPU job is required")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / (args.lane + ".lock")).open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(args.root / "queue.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, digest TEXT, status TEXT, job_id TEXT, exit_code INTEGER, updated TEXT)")
    db.commit()
    def record(task_id, digest, state, code=None):
        with db:
            db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?)",
                       (task_id, digest, state, os.environ["SLURM_JOB_ID"], code,
                        datetime.now(timezone.utc).isoformat()))
    print(json.dumps({"worker": args.lane, "job_id": os.environ["SLURM_JOB_ID"],
                      "node": os.uname().nodename, "policy": "retain allocation after tasks"}), flush=True)
    idle = False
    while True:
        try:
            queue = json.loads((args.root / "tasks.json").read_text())
            tasks = queue.get(args.lane, [])
            pending = []
            for task in tasks:
                digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
                previous = db.execute("SELECT digest,status FROM tasks WHERE id=?", (task["id"],)).fetchone()
                if previous and previous[0] != digest:
                    raise ValueError("Task changed after registration: " + task["id"])
                if not previous or previous[1] == "running":
                    pending.append((task, digest))
            if not pending:
                if not idle:
                    print(json.dumps({"worker": args.lane, "state": "awaiting queued work; allocation retained"}), flush=True)
                    idle = True
                time.sleep(10)
                continue
            idle = False
            task, digest = pending[0]
            record(task["id"], digest, "running")
            print(json.dumps({"task": task["id"], "state": "running"}), flush=True)
            try:
                env = dict(os.environ, **task.get("env", {}))
                result = subprocess.run(task["command"], cwd=task["cwd"], env=env, check=False)
                code = result.returncode
            except Exception as error:
                print(json.dumps({"task": task["id"], "error": str(error)}), flush=True)
                code = 127
            record(task["id"], digest, "completed" if code == 0 else "failed", code)
            print(json.dumps({"task": task["id"], "exit_code": code, "allocation": "retained"}), flush=True)
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps({"worker": args.lane, "queue_error": str(error), "allocation": "retained"}), flush=True)
            time.sleep(30)

if __name__ == "__main__":
    main()
