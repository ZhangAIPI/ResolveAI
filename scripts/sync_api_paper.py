"""Keep Overleaf and PDFs aligned with completed API cohorts while the suite runs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SUFFIXES={".pdf",".tex",".bib",".sty",".bst",".jpg",".png",".json",".md"}


def synchronize(repository,runtime,project):
    root=runtime/"evaluation/openai"
    mirror=runtime/"paper-overleaf"
    cli=json.loads((runtime/"tools/olcli/invocation.json").read_text())["command"]
    def call(arguments):
        try:
            result=subprocess.run([*cli,*arguments],capture_output=True,text=True,timeout=240)
        except (OSError,subprocess.TimeoutExpired):
            raise RuntimeError("Overleaf operation unavailable: "+arguments[0]) from None
        if result.returncode:
            raise RuntimeError("Overleaf operation failed: "+arguments[0])
    with (root/"paper-publish.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=sorted(p for p in (repository/"paper").rglob("*") if p.is_file() and p.suffix in SUFFIXES)
        digest=hashlib.sha256()
        for path in paths:
            digest.update(str(path.relative_to(repository/"paper")).encode())
            digest.update(path.read_bytes())
        signature=digest.hexdigest()
        state_path=runtime/"paper-draft/api-sync.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("source_sha256")==signature:
            return
        # Refuse to overwrite an independent Overleaf edit.
        call(["diff",project,str(mirror),"--name-only","--exit-code"])
        for path in paths:
            target=mirror/path.relative_to(repository/"paper")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        call(["push",str(mirror)])
        sys.path.insert(0,str(runtime/"tools/pdf"))
        import fitz
        output=runtime/"paper-draft"
        output.mkdir(exist_ok=True)
        exported=[]
        for source,name in (("main.tex","ResolveAI_CVPR_methods.pdf"),
                            ("supplementary.tex","ResolveAI_CVPR_supplementary.pdf")):
            pdf=output/name
            call(["pdf",project,"-o",str(pdf),"-r",source])
            with fitz.open(pdf) as document:
                text="\n".join(page.get_text() for page in document)
                if "[?]" in text or "??" in text or "\ufffd" in text or document.metadata.get("author"):
                    raise RuntimeError("Exported PDF failed text or anonymity checks")
                exported.append({"root":source,"pages":len(document)})
            shutil.copy2(pdf,Path("/u/zzh136/Documents/Codex")/name)
        call(["diff",project,str(mirror),"--name-only","--exit-code"])
        state_path.write_text(json.dumps({"source_sha256":signature,"files":len(paths),
                                         "exports":exported,"project":project},indent=2)+"\n")
        print(json.dumps({"overleaf_synced":True,"exports":exported}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("runtime",type=Path)
    parser.add_argument("--project",default="6ae80679a4c7a64df04ac5b8")
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if os.getenv("SLURMD_NODENAME")!="macula":
        parser.error("Run in the retained macula allocation")
    lock=(args.runtime/"evaluation/openai/overleaf-sync.lock").open("w")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An Overleaf API-study synchronizer is already running")
    while True:
        synchronize(args.repository,args.runtime,args.project)
        suite=args.runtime/"evaluation/openai/suite_status.json"
        if args.once or (suite.exists() and json.loads(suite.read_text())["phase"]=="finished"):
            return
        time.sleep(20)


if __name__=="__main__":
    main()
