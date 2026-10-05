"""Fetch selected CO3D test frames via official ZIP HTTP ranges, without depth caches."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import zipfile
import zlib

import requests

from .co3d_data import identifier, read_json, source_split
from .co3d_download import download, REVISION


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url=url; self.session=requests.Session(); self.position=0
        with self.session.head(url,timeout=120) as response:
            response.raise_for_status(); self.length=int(response.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if self.position<0:raise ValueError("negative range offset")
        return self.position
    def read(self, size=-1):
        size=self.length-self.position if size<0 else min(size,self.length-self.position)
        if size<=0:return b""
        if size>32*1024*1024:raise ValueError("unexpected large ZIP index read")
        data=read_range(self.session,self.url,self.position,self.position+size-1)
        self.position+=len(data);return data
    def close(self):
        self.session.close();super().close()


def read_range(session, url, start, end):
    with session.get(url,headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=120) as response:
        if response.status_code!=206:
            raise ValueError("server does not honor byte ranges")
        content_range=response.headers.get("Content-Range","")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise ValueError("unexpected HTTP range response")
        data=response.content
    if len(data)!=end-start+1:raise ValueError("truncated range")
    return data


def selected_frames(root,categories,families,views,seed=42):
    groups_by_category=[]
    for category in categories:
        partition=source_split(root,category);groups=defaultdict(list)
        for frame in read_json(root/category/"frame_annotations.jgz"):
            if partition.get(frame["sequence_name"])=="test" and frame.get("meta",{}).get("frame_type","").endswith("_known"):
                groups[frame["sequence_name"]].append(frame)
        eligible=[(category,seq,rows) for seq,rows in groups.items() if len(rows)>=2]
        eligible.sort(key=lambda item:identifier(str(seed)+category+item[1]));groups_by_category.append(eligible)
    pool=[items[i] for i in range(max(map(len,groups_by_category),default=0)) for items in groups_by_category if i<len(items)]
    if len(pool)<families:raise ValueError(f"only {len(pool)} test sequences with at least two known-context frames")
    selected=[]
    for category,sequence,rows in pool[:families]:
        rows.sort(key=lambda frame:frame["frame_number"])
        count=min(views,len(rows))
        selected.extend(rows[round(i*(len(rows)-1)/(count-1))] for i in range(count))
    return selected


def collect(root,categories,families=300,views=16,workers=12,seed=42):
    if families<1 or views<2 or workers<1:raise ValueError("invalid subset size/workers")
    # Archive zero contains official annotations, splits and a small reference sequence.
    download(root,categories,[0],min(workers,3))
    frames=selected_frames(root,categories,families,views,seed)
    wanted={frame[key]["path"] for frame in frames for key in ("image","mask")}
    wanted={name for name in wanted if not (root/name).is_file()}
    links=read_json(root/"links.json")["full"];checks=read_json(root/"co3d_sha256.json")["full"]
    urls=[url for category in categories for url in links[category][1:]]
    def index(url):
        name=url.rsplit("/",1)[1];local=root/(name+".partial")
        handle=local if local.exists() and zipfile.is_zipfile(local) else RangeFile(url)
        try:
            with zipfile.ZipFile(handle) as archive:
                rows=[{"path":info.filename,"offset":info.header_offset,"compressed":info.compress_size,
                       "size":info.file_size,"crc32":info.CRC,"method":info.compress_type,"url":url}
                      for info in archive.infolist() if info.filename in wanted]
            print("indexed "+name+" selected="+str(len(rows)),flush=True)
            return rows
        finally:
            if isinstance(handle,RangeFile):handle.close()
    entries=[]
    with ThreadPoolExecutor(max_workers=min(workers,6)) as pool:
        for rows in pool.map(index,urls):entries.extend(rows)
    by_path={row["path"]:row for row in entries}
    if wanted-by_path.keys():raise ValueError(f"{len(wanted-by_path.keys())} target files absent from official ZIP indices")
    state=threading.local()
    def extract(row):
        if not hasattr(state,"session"):state.session=requests.Session()
        local=root/(row["url"].rsplit("/",1)[1]+".partial")
        length=30+len(row["path"].encode("utf-8"))+512+row["compressed"]
        if local.exists() and zipfile.is_zipfile(local):
            with local.open("rb") as source:source.seek(row["offset"]);data=source.read(length)
        else:
            data=read_range(state.session,row["url"],row["offset"],row["offset"]+length-1)
        header=struct.unpack("<IHHHHHIIIHH",data[:30])
        if header[0]!=0x04034b50 or header[3]!=row["method"]:raise ValueError("unexpected local ZIP header")
        offset=30+header[-2]+header[-1];payload=data[offset:offset+row["compressed"]]
        if len(payload)!=row["compressed"]:raise ValueError("truncated compressed asset")
        pixels=payload if row["method"]==zipfile.ZIP_STORED else zlib.decompress(payload,-15) if row["method"]==zipfile.ZIP_DEFLATED else None
        if pixels is None or len(pixels)!=row["size"] or zlib.crc32(pixels)!=row["crc32"]:
            raise ValueError("ZIP member size/CRC32 mismatch")
        target=root/row["path"]
        if not target.resolve().is_relative_to(root.resolve()):raise ValueError("unsafe member path")
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(pixels)
        return {**row,"sha256":hashlib.sha256(pixels).hexdigest()}
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number,row in enumerate(pool.map(extract,by_path.values()),1):
            records.append(row)
            if number%100==0:print(f"fetched {number}/{len(by_path)}",flush=True)
    # Include reused official assets with their source paths and computed content hashes.
    indexed_paths={r["path"] for r in records}
    for frame in frames:
        for kind in ("image","mask"):
            name=frame[kind]["path"]
            if name not in indexed_paths:
                records.append({"path":name,"sha256":hashlib.sha256((root/name).read_bytes()).hexdigest(),"reused_local":True})
                indexed_paths.add(name)
    (root/"subset_assets.json").write_text(json.dumps(records,indent=2)+"\n")
    manifest={"dataset":"CO3Dv2 selected official test sequences","revision":REVISION,"categories":categories,
              "families":families,"candidate_views_per_family":views,"seed":seed,"asset_count":len(records),
              "source":"https://github.com/facebookresearch/co3d","archive_sha256_reference":checks,
              "verification":"metadata archives verified by official SHA256; ranged members checked by ZIP CRC32 and recorded SHA256; whole ranged archives not SHA256-verified",
              "license":"CC-BY-NC-4.0"}
    (root/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    parser.add_argument("--categories",nargs="+",default=["chair","cup","bottle","book","car","bench"])
    parser.add_argument("--families",type=int,default=300);parser.add_argument("--views",type=int,default=16)
    parser.add_argument("--workers",type=int,default=12);parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args();manifest=collect(args.output,args.categories,args.families,args.views,args.workers,args.seed)
    print(json.dumps({k:v for k,v in manifest.items() if k!="archive_sha256_reference"},indent=2))


if __name__=="__main__":main()
