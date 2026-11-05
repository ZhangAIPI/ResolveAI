"""Build a frozen 40-family blind sufficiency review package; never edits source data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from PIL import Image

SEED = 20261105
SUFFICIENCY = ("sufficient", "insufficient", "uncertain")
def digest(b): return hashlib.sha256(b).hexdigest()
def opaque(*parts): return digest(("blind40-v1|"+"|".join(map(str,parts))).encode())[:20]
def json_write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def dedupe(case, evidence):
    """Retain highest resolution for each source, without combining independent photos."""
    best={}
    for e in evidence:
        path=Path(case["asset_root"])/e["path"]
        with Image.open(path) as im: area=im.width*im.height
        key=e.get("source_id",e["id"])
        if key not in best or area>best[key][0]: best[key]=(area,e)
    return [pair[1] for pair in best.values()]
def materials(family):
    ob=family["Obtainable"]; byid={e["id"]:e for e in ob["evidence"]}
    initial=[byid[x] for x in ob["initial"]]
    full=[e for e in ob["evidence"] if e["available"]]
    high=[]
    for e in initial:
        same=[x for x in full if x.get("source_id",x["id"])==e.get("source_id",e["id"])]
        high.extend(dedupe(ob,same))
    blocked=[e for e in family["Missing"]["evidence"] if e["available"]]
    return {k:dedupe(ob,v) for k,v in {
        "initial":initial,"candidate_pool":full,
        "resolution_control":high,"blocked_pool":blocked}.items()}
def assign(units, n, size, rng):
    """Balanced packets; forbid repeated families/source groups within each rater."""
    tokens=[u for u in units for _ in range(2)]
    sizes=[size]*n if size is not None else [len(tokens)//n+(i<len(tokens)%n)for i in range(n)]
    assert len(tokens)==sum(sizes)
    rng.shuffle(tokens)
    packets=[];offset=0
    for size_i in sizes:
        packets.append(tokens[offset:offset+size_i]);offset+=size_i
    def cost(packet):
        c=0
        for i,a in enumerate(packet):
            for b in packet[:i]:
                if a["family_id"]==b["family_id"]: c+=100
                elif set(a["group_ids"])&set(b["group_ids"]): c+=10
        return c
    costs=[cost(p)for p in packets]
    for it in range(120000):
        if not sum(costs):break
        bad=[i for i,c in enumerate(costs)if c]
        i=rng.choice(bad);j=rng.randrange(n)
        if i==j:continue
        a,b=rng.randrange(len(packets[i])),rng.randrange(len(packets[j]))
        old=costs[i]+costs[j]
        packets[i][a],packets[j][b]=packets[j][b],packets[i][a]
        newi,newj=cost(packets[i]),cost(packets[j])
        if newi+newj<=old or (it<20000 and rng.random()<.0008):
            costs[i],costs[j]=newi,newj
        else: packets[i][a],packets[j][b]=packets[j][b],packets[i][a]
    if sum(costs): raise ValueError("Could not create source-disjoint short packets")
    for p in packets:rng.shuffle(p)
    return packets
def make_unit(case,kind,rows):
    uid="u-"+opaque(case["family_id"],kind)
    images=[]
    rng=random.Random(SEED+int(opaque(uid)[:10],16)); rows=list(rows);rng.shuffle(rows)
    for i,e in enumerate(rows,1):
        path=Path(case["asset_root"])/e["path"]
        with Image.open(path) as im: width,height=im.size
        label={"zh":"目标图片","en":"Target photograph"}
        if e["object"]=="normal-reference":
            label={"zh":"正常参考（另一件实物）","en":"Normal reference (another specimen)"}
        elif e["object"]in ("subject-A","subject-B"):
            slot=e["object"][-1]
            label={"zh":"对象 "+slot,"en":"Subject "+slot}
        box=e.get("target_bbox")
        if box:
            bounds=e.get("source_bbox",[0,0,*e.get("source_size",[width,height])])
            bw,bh=bounds[2]-bounds[0],bounds[3]-bounds[1]
            box=[max(0,min(1,(box[0]-bounds[0])/bw)),max(0,min(1,(box[1]-bounds[1])/bh)),
                 max(0,min(1,(box[2]-bounds[0])/bw)),max(0,min(1,(box[3]-bounds[1])/bh))]
        images.append({"id":"I"+str(i),"label":label,"width":width,"height":height,
            "target_box":box,"private":{"path":str(path),"sha256":digest(path.read_bytes()),
                "evidence_id":e["id"],"source_id":e.get("source_id",e["id"]),
                "view":e.get("camera_view",e.get("view")),"object":e["object"]}})
    from resolveai.human_study import chinese_claim
    return {"unit_id":uid,"family_id":case["family_id"],"kind":kind,
        "group_ids":case["group_ids"],"claim":{"zh":chinese_claim(case),"en":case["claim"]},
        "context":{"zh":"请只判断指定目标。正常参考是另一个实物，不是目标的之前照片。" if case["task"]=="state" else "A、B只是提交槽位，不是已经确定的物体身份；请聚焦指定前景目标。",
                   "en":case.get("task_instructions","A and B identify submission slots, not known identities. Compare the specified foreground targets.")},
        "images":images}
def build(args):
    out=Path(args.output)
    if out.exists():raise ValueError("Choose a new output directory; source and previous packages are never overwritten")
    data=Path(args.dataset); cohort=json.loads(Path(args.cohort).read_text())
    dataset_hash=digest((data/"cases.json").read_bytes())
    if cohort["dataset_sha256"]!=dataset_hash:raise ValueError("Dataset differs from frozen evaluation cohort")
    cases=json.loads((data/"cases.json").read_text())
    families=defaultdict(dict)
    for c in cases:families[c["family_id"]][c["variant"]]=c
    source=[families[f]["Obtainable"]for f in cohort["family_ids"]]
    rng=random.Random(SEED);chosen=[]
    for task in ["state","identity"]:
        for label in ["Supported","Refuted"]:
            pool=sorted([c for c in source if c["task"]==task and c["provenance_truth"]==label],key=lambda c:c["family_id"])
            chosen.extend(rng.sample(pool,10))
    main=[];extra=[];mapfamilies={}
    for c in chosen:
        mats=materials(families[c["family_id"]])
        private={"task":c["task"],"source_label":c["provenance_truth"],"category":c["category"],
            "group_ids":c["group_ids"],"provenance":c.get("provenance",{}),"units":{},
            "source_limitations":["Source labels and candidate sufficient sets are not independent visual reviews."]}
        for kind in ["initial","candidate_pool"]+(["resolution_control","blocked_pool"]if c["task"]=="identity"else[]):
            unit=make_unit(c,kind,mats[kind]);private["units"][kind]=unit["unit_id"]
            (main if kind in ["initial","candidate_pool"]else extra).append(unit)
        if c["task"]=="identity":
            high_sources={e.get("source_id",e["id"])for e in mats["resolution_control"]}
            new=[e for e in mats["candidate_pool"]if e.get("source_id",e["id"])not in high_sources]
            private["genuinely_new_recorded_frames"]=[{"evidence_id":e["id"],"source_id":e.get("source_id",e["id"]),"object":e["object"],"view":e.get("camera_view",e.get("view"))}for e in new]
            private["camera_pose_verified"]=False
            private["source_limitations"].append("Different source-frame IDs are not proof of a useful or geometrically distinct viewpoint; inspect the photos or verify camera metadata.")
        else:
            private["blocked_pool_equals_initial_sources"]=sorted(e["id"]for e in mats["blocked_pool"])==sorted(e["id"]for e in mats["initial"])
        mapfamilies[c["family_id"]]=private
    template_path=Path(__file__).with_name("review.html")
    if not template_path.exists():
        template_path=Path(__file__).resolve().parents[1]/"src/resolveai/web/blind_review.html"
    template=template_path.read_text()
    private_dir=out/"private";private_dir.mkdir(parents=True)
    package_id="blind40-"+opaque(SEED,dataset_hash)
    mapping={"package_id":package_id,"seed":SEED,"dataset_sha256":dataset_hash,
        "cohort_sha256":digest(Path(args.cohort).read_bytes()),"sampling":"40 frozen-cohort families, 10 per task/source-label stratum, random without observing model outputs",
        "families":mapfamilies,"units":{},"packets":{},"status":"prepared_no_human_annotations"}
    for phase,units,n,size,prefix in [("main",main,20,8,"R"),("qualification",extra,12,None,"Q")]:
        packets=assign(units,n,size,rng)
        for i,packet in enumerate(packets,1):
            pid=prefix+str(i).zfill(2);root=out/"public"/phase/pid
            assets=root/"assets";assets.mkdir(parents=True)
            clean=[]
            for unit in packet:
                public={k:unit[k]for k in ["unit_id","claim","context"]}
                public["images"]=[]
                mapping["units"][unit["unit_id"]]={**unit,"assigned_packets":mapping["units"].get(unit["unit_id"],{}).get("assigned_packets",[])+[pid]}
                for im in unit["images"]:
                    src=Path(im["private"]["path"])
                    name="photo-"+opaque(package_id,pid,unit["unit_id"],im["id"])+src.suffix.lower()
                    shutil.copyfile(src,assets/name)
                    public["images"].append({k:im[k]for k in ["id","label","width","height","target_box"]}|{"src":"assets/"+name})
                clean.append(public)
            mapping["packets"][pid]={"phase":phase,"unit_ids":[u["unit_id"]for u in packet]}
            payload={"package_id":package_id,"packet_id":pid,"units":clean}
            text=template.replace("__PACKET_DATA__",json.dumps(payload,ensure_ascii=False).replace("</","<\\/"))
            for banned in ["provenance_truth","minimal_evidence_sets","source_label","Obtainable","Unavailable","resolution_control","candidate_pool"]:
                if banned in text:raise ValueError("Private metadata leaked: "+banned)
            (root/"index.html").write_text(text)
    json_write(private_dir/"mapping.json",mapping)
    summary={"status":"prepared_no_human_annotations","families":40,"task_counts":dict(Counter(c["task"]for c in chosen)),
        "strata":dict(Counter(c["task"]+":"+c["provenance_truth"]for c in chosen)),
        "main":{"materials":80,"independent_votes_per_material":2,"packets":20,"units_per_packet":8,"target_minutes_per_packet":"5–10, not yet pilot timed"},
        "extra_view_qualification":{"materials":40,"independent_votes_per_material":2,"additional_packets":12,"units_per_packet":"6–7"},
        "privacy":"Each packet contains only assigned images; no source labels, expected versions, source filenames or annotations.",
        "assignment":"No rater sees the same family or underlying source group twice within a phase.",
        "completed_human_reviews":0}
    json_write(private_dir/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset");ap.add_argument("cohort");ap.add_argument("output")
    build(ap.parse_args())
