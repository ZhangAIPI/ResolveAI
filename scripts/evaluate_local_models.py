"""Run pinned local models on two GPUs; the Qwen group collects and optionally publishes reports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODELS = {
    "qwen3": ("model", "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "qwen25": ("models/qwen25", "Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "internvl35": ("models/internvl35", "OpenGVLab/InternVL3_5-8B-HF", "741a7d03020411e666c6109218ab71e08151ef86"),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group",choices=["qwen","internvl","qwen3"])
    parser.add_argument("runtime",type=Path);parser.add_argument("data",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--code",type=Path,required=True);parser.add_argument("--publish-repo",type=Path)
    parser.add_argument("--max-turns",type=int,default=6)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--shards", type=int, default=2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.gpus < 1 or args.shards < 1:
        parser.error("GPU and shard counts must be positive")
    env={**os.environ,"PYTHONPATH":str(args.code/"src"),"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"}
    if args.group == "qwen":
        configurations = [("qwen3",None),("qwen25",None)]
    else:
        model = "qwen3" if args.group == "qwen3" else "internvl35"
        configurations = [(model,shard) for shard in range(args.shards)]
    families = sorted({c["family_id"] for c in json.loads((args.data/"cases.json").read_text())})

    def folder_for(name, shard):
        return args.output / (name if shard is None else name+"-shard"+str(shard))

    def remaining(configuration):
        name, shard = configuration
        folder = folder_for(name, shard)
        count = len(families) if shard is None else len(families[shard::args.shards])
        path = folder/"results.jsonl"
        completed = sum(1 for line in path.open()) if path.exists() else 0
        return count*8-completed

    queues = [[] for _ in range(args.gpus)]
    loads = [0]*args.gpus
    for configuration in sorted(configurations, key=remaining, reverse=True):
        device = min(range(args.gpus), key=lambda index:loads[index])
        queues[device].append(configuration)
        loads[device] += max(remaining(configuration),1)
    print(json.dumps({"planned_remaining_per_gpu":loads,"queues":queues}),flush=True)

    def run_queue(device, queue):
        codes = {}
        for name, shard in queue:
            folder = folder_for(name, shard);folder.mkdir(exist_ok=True)
            path, model_id, revision = MODELS[name]
            command=[sys.executable,"-u","-m","resolveai.benchmark",str(args.data),str(args.runtime/path),str(folder),
                "--model-id",model_id,"--revision",revision,"--grounding-root",str(args.runtime/"grounding"),"--device",str(device),
                "--budget","12","--max-turns",str(args.max_turns),"--max-context-tokens","8192"]
            if shard is not None:command += ["--shards",str(args.shards),"--shard-index",str(shard)]
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if "seed_execution" in manifest:
                    manifest["slurm_job"] = os.getenv("SLURM_JOB_ID")
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            with (folder/"worker.log").open("a") as log:
                process=subprocess.Popen(command,cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT)
                print("started "+folder.name+" gpu="+str(device)+" pid="+str(process.pid),flush=True)
                codes[folder.name]=process.wait()
        return codes

    codes={}
    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        tasks=[pool.submit(run_queue,device,queue) for device,queue in enumerate(queues) if queue]
        for task in tasks:codes.update(task.result())
    status={"exit_codes":codes,"complete":all(code==0 for code in codes.values())}
    (args.output/(args.group+"-status.json")).write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status),flush=True)
    if not status["complete"]:sys.exit(1)
    if args.group=="internvl":return
    other=args.output/"internvl-status.json"
    deadline=time.monotonic()+7200
    while not other.exists():
        if time.monotonic()>deadline:raise TimeoutError("other model group has no completion status after two hours")
        time.sleep(15)
    if not json.loads(other.read_text())["complete"]:
        print("InternVL failed; partial runs retained for diagnosis",flush=True);return
    report=[sys.executable,"-m","resolveai.benchmark_report"]
    if args.group == "qwen3":
        paths = [str(args.output/("qwen3-shard"+str(index))) for index in range(args.shards)]
        subprocess.check_call(report+paths+["--merge-shards","--output",str(args.output/"qwen3")],cwd=args.code,env=env)
    merged=args.output/"internvl35"
    if not (merged/"results.jsonl").exists():
        paths = [str(path) for path in sorted(args.output.glob("internvl35-shard*")) if path.is_dir()]
        subprocess.check_call(report+paths+["--merge-shards","--output",str(merged)],cwd=args.code,env=env)
    stem=args.output/"summary"
    subprocess.check_call(report+[str(args.output/"qwen3"),str(args.output/"qwen25"),str(merged),"--output",str(stem)],cwd=args.code,env=env)
    if args.publish_repo:
        repo=args.publish_repo
        branch=subprocess.check_output(["git","-C",str(repo),"branch","--show-current"],text=True).strip()
        if branch!="main":
            print("Report ready; publication skipped because repository is not on main",flush=True);return
        paths=[]
        for suffix in (".json",".md"):
            path=Path("docs/results/co3d_candidates"+suffix);(repo/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(stem.with_suffix(suffix),repo/path);paths.append(str(path))
        subprocess.check_call(["git","-C",str(repo),"add",*paths])
        subprocess.check_call(["git","-C",str(repo),"commit","--only",*paths,"-m","Report pinned local-model CO3D candidate evaluation"])
        subprocess.check_call(["git","-C",str(repo),"push","origin","HEAD:main"],env={**env,"GIT_SSH_COMMAND":"ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"})
        (args.output/"published.json").write_text(json.dumps({"repository":"https://github.com/ZhangAIPI/ResolveAI","paths":paths}))


if __name__=="__main__":main()
