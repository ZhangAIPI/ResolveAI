"""Run dependency-ordered tasks on already allocated GPUs, claiming each task once."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


def claim(root, tasks, job_id):
    """Hold an OS lock through execution; a dead worker can safely resume its task."""
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "pool.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, digest TEXT, status TEXT, job_id TEXT, exit_code INTEGER)")
    db.commit()
    try:
        for task in tasks:
            digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
            previous = db.execute("SELECT digest,status FROM tasks WHERE id=?", (task["id"],)).fetchone()
            if previous and previous[0] != digest:
                raise ValueError("Registered task changed: " + task["id"])
            if previous and previous[1] in {"completed", "failed"}:
                continue
            if any(db.execute("SELECT status FROM tasks WHERE id=?", (dep,)).fetchone() != ("completed",)
                   for dep in task.get("requires", [])):
                continue
            lock = (root / (task["id"] + ".lock")).open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            # Recheck after acquiring the cross-process lock.
            current = db.execute("SELECT status FROM tasks WHERE id=?", (task["id"],)).fetchone()
            if current and current[0] in {"completed", "failed"}:
                lock.close()
                continue
            with db:
                db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,NULL)",
                           (task["id"], digest, "running", job_id))
            return task, lock
        return None
    finally:
        db.close()


def finish(root, task, code):
    with sqlite3.connect(root / "pool.sqlite", timeout=30) as db:
        db.execute("UPDATE tasks SET status=?,exit_code=? WHERE id=?",
                   ("completed" if code == 0 else "failed", code, task["id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Use an existing GPU allocation")
    lock = (args.root / ("worker-" + os.environ["SLURM_JOB_ID"] + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        tasks = json.loads((args.root / "tasks.json").read_text())
        selected = claim(args.root, tasks, os.environ["SLURM_JOB_ID"])
        if selected is None:
            with sqlite3.connect(args.root / "pool.sqlite") as db:
                states = dict(db.execute("SELECT id,status FROM tasks"))
            if all(states.get(t["id"]) == "completed" for t in tasks):
                return
            if any(states.get(t["id"]) == "failed" for t in tasks):
                raise RuntimeError("A pool task failed; retain logs and allocation for diagnosis")
            time.sleep(15)
            continue
        task, task_lock = selected
        print(json.dumps({"pool_task":task["id"],"job_id":os.environ["SLURM_JOB_ID"],"state":"running"}), flush=True)
        try:
            with (args.root / (task["id"] + ".log")).open("a") as log:
                result = subprocess.run(task["command"], cwd=task["cwd"],
                                        env={**os.environ, **task.get("env", {})},
                                        stdout=log, stderr=subprocess.STDOUT)
            code = result.returncode
        except OSError:
            code = 127
        finish(args.root, task, code)
        task_lock.close()
        print(json.dumps({"pool_task":task["id"],"exit_code":code}), flush=True)


if __name__ == "__main__":
    main()
