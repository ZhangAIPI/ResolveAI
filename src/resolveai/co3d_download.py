"""Download pinned official CO3D archives, verify SHA256, then discard archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

REVISION = "eb51d7583c56ff23dc918d9deafee50f4d8178c3"
BASE = "https://raw.githubusercontent.com/facebookresearch/co3d/"+REVISION+"/"


def download(root, categories, indices, workers=3):
    root.mkdir(parents=True,exist_ok=True)
    for name in ("co3d/links.json", "co3d/co3d_sha256.json", "LICENSE"):
        with urllib.request.urlopen(BASE+name,timeout=120) as response:
            (root/Path(name).name).write_bytes(response.read())
    links=json.loads((root/"links.json").read_text())["full"]
    checks=json.loads((root/"co3d_sha256.json").read_text())["full"]
    urls=[links[category][index] for category in categories for index in indices]
    def fetch(url):
        name=url.rsplit("/",1)[1]; marker=root/(name+".verified.json")
        expected={"url":url,"sha256":checks[name]}
        if marker.exists() and json.loads(marker.read_text()) == expected:
            return
        archive=root/(name+".partial");digest=hashlib.sha256()
        with urllib.request.urlopen(url,timeout=120) as response,archive.open("wb") as output:
            while block:=response.read(4*1024*1024):
                output.write(block);digest.update(block)
        if digest.hexdigest()!=checks[name]:
            raise ValueError("official SHA256 mismatch: "+name)
        with zipfile.ZipFile(archive) as package:
            if any(not (root/member.filename).resolve().is_relative_to(root.resolve()) for member in package.infolist()):
                raise ValueError("archive path escapes dataset root")
            package.extractall(root)
        marker.write_text(json.dumps(expected));archive.unlink()
        print("verified "+name,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch,urls))
    verified=sorted(json.loads(p.read_text())["url"] for p in root.glob("*.zip.verified.json"))
    manifest={"dataset":"CO3Dv2 selected full archives","revision":REVISION,"archives":verified,
              "license":"CC-BY-NC-4.0","official_source":"https://github.com/facebookresearch/co3d"}
    (root/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    parser.add_argument("--categories",nargs="+",default=["chair","cup","bottle"])
    parser.add_argument("--archive-indices",nargs="+",type=int,default=[0,1])
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if args.workers < 1 or any(i<0 for i in args.archive_indices):parser.error("invalid worker count/archive index")
    print(json.dumps(download(args.output,args.categories,args.archive_indices,args.workers),indent=2))


if __name__=="__main__":main()
