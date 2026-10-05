"""Run pinned local models on two GPUs; the Qwen group collects and optionally publishes reports."""
import argparse
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
    parser.add_argument("group",choices=["qwen","internvl"])
    parser.add_argument("runtime",type=Path);parser.add_argument("data",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--code",type=Path,required=True);parser.add_argument("--publish-repo",type=Path)
    parser.add_argument("--max-turns",type=int,default=6)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env={**os.environ,"PYTHONPATH":str(args.code/"src"),"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"}
    configurations=[("qwen3",0,None),("qwen25",1,None)] if args.group=="qwen" else [("internvl35",0,0),("internvl35",1,1)]
    running=[]
    for name,device,shard in configurations:
        folder=args.output/(name if shard is None else name+"-shard"+str(shard));folder.mkdir(exist_ok=True)
        path,model_id,revision=MODELS[name]
        command=[sys.executable,"-u","-m","resolveai.benchmark",str(args.data),str(args.runtime/path),str(folder),
            "--model-id",model_id,"--revision",revision,"--grounding-root",str(args.runtime/"grounding"),"--device",str(device),
            "--budget","12","--max-turns",str(args.max_turns),"--max-context-tokens","8192"]
        if shard is not None:command += ["--shards","2","--shard-index",str(shard)]
        log=(folder/"worker.log").open("a")
        process=subprocess.Popen(command,cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT)
        running.append((folder,process,log));print("started "+folder.name+" pid="+str(process.pid),flush=True)
    codes={}
    for folder,process,log in running:
        codes[folder.name]=process.wait();log.close()
    status={"exit_codes":codes,"complete":all(code==0 for code in codes.values())}
    (args.output/(args.group+"-status.json")).write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status),flush=True)
    if not status["complete"]:sys.exit(1)
    if args.group!="qwen":return
    other=args.output/"internvl-status.json"
    deadline=time.monotonic()+7200
    while not other.exists():
        if time.monotonic()>deadline:raise TimeoutError("other model group has no completion status after two hours")
        time.sleep(15)
    if not json.loads(other.read_text())["complete"]:
        print("InternVL failed; partial runs retained for diagnosis",flush=True);return
    report=[sys.executable,"-m","resolveai.benchmark_report"]
    merged=args.output/"internvl35"
    if not (merged/"results.jsonl").exists():
        subprocess.check_call(report+[str(args.output/"internvl35-shard0"),str(args.output/"internvl35-shard1"),
            "--merge-shards","--output",str(merged)],cwd=args.code,env=env)
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
