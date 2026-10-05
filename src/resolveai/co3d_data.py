"""Source-preserving CO3Dv2 multiview candidate cases; no invented damage labels."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

from PIL import Image


def identifier(value):
    return hashlib.sha256(("resolveai-co3d-v1|" + value).encode()).hexdigest()[:20]


def read_json(path):
    with gzip.open(path, "rt") if path.suffix == ".jgz" else path.open() as stream:
        return json.load(stream)


def source_split(root, category):
    """Assign whole sequences to official few-view task partitions, not frame splits."""
    membership = defaultdict(set)
    for source, target in (("train", "train"), ("dev", "val"), ("test", "test")):
        path = root/category/"set_lists"/("set_lists_fewview_"+source+".json")
        if path.exists():
            for rows in read_json(path).values():
                for sequence, _, _ in rows:
                    membership[sequence].add(target)
    if not membership:
        # The official single-sequence subset only ships many-view test context/targets.
        path = root/category/"set_lists"/"set_lists_manyview_test_0.json"
        if path.exists():
            for rows in read_json(path).values():
                for sequence, _, _ in rows:
                    membership[sequence].add("test")
    # A sequence appearing in multiple task partitions needs an explicit audit.
    return {sequence: next(iter(roles)) for sequence, roles in membership.items() if len(roles) == 1}


def foreground_box(root, frame):
    path = root/frame["mask"]["path"]
    with Image.open(path) as image:
        mask = image.convert("L"); bounds = mask.point(lambda p: 255 if p >= 128 else 0).getbbox()
        if bounds is None:
            return None
        height, width = frame["image"]["size"]
        return [math.floor(bounds[0]*width/mask.width), math.floor(bounds[1]*height/mask.height),
                math.ceil(bounds[2]*width/mask.width), math.ceil(bounds[3]*height/mask.height)]


def select_views(root, frames, count):
    """Choose spread-out frames with actual source foreground masks, skipping placeholders."""
    frames = sorted(frames,key=lambda frame:frame["frame_number"])
    selected, cached = {}, {}
    for index in (round(i*(len(frames)-1)/(count-1)) for i in range(count)):
        for candidate in sorted(range(len(frames)),key=lambda j:abs(j-index)):
            if candidate in selected:
                continue
            if candidate not in cached:
                cached[candidate] = foreground_box(root,frames[candidate])
            if cached[candidate] is not None:
                selected[candidate] = (frames[candidate],cached[candidate])
                break
        else:
            break
    return [selected[i] for i in sorted(selected)] if len(selected)>=2 else None


def link_asset(source, target):
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError("asset collision")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def prepare(source, output, categories, families=300, views=8, split="test", seed=42):
    """Generate positive identity candidates from official same-sequence provenance.

    Mask regions and minimum evidence are provisional. A sequence is an identity
    proxy, not an independent human sufficiency annotation. Human review is
    required before these cases become semantic benchmark/training supervision.
    """
    if families < 1 or views < 2:
        raise ValueError("require positive family count and at least two views")
    if (output/"cases.json").exists():
        raise ValueError("use a fresh output directory to preserve frozen case manifests")
    assets = output/"assets"; assets.mkdir(parents=True,exist_ok=True)
    selected = []
    for category in categories:
        groups = defaultdict(list)
        partition = source_split(source,category)
        for frame in read_json(source/category/"frame_annotations.jgz"):
            sequence = frame["sequence_name"]
            if partition.get(sequence) != split:
                continue
            paths = [source/frame["image"]["path"], source/frame["mask"]["path"]]
            if all(p.is_file() for p in paths):
                groups[sequence].append(frame)
        eligible = [(sequence, rows) for sequence, rows in groups.items() if len(rows) >= 2]
        eligible.sort(key=lambda item: identifier(str(seed)+category+item[0]))
        selected.append([(category, sequence, rows) for sequence,rows in eligible])
    # Round-robin category balance, without claiming equal counts when sources are absent.
    pool = [items[i] for i in range(max(map(len,selected),default=0)) for items in selected if i < len(items)]
    if len(pool) < families:
        raise ValueError(f"only {len(pool)} eligible {split} sequences; requested {families}")
    validated = []
    for category,sequence,frames in pool:
        chosen = select_views(source,frames,min(views,len(frames)))
        if chosen is not None:
            validated.append((category,sequence,chosen))
        if len(validated) == families:
            break
    if len(validated) < families:
        raise ValueError(f"only {len(validated)} sequences have enough nonempty foreground masks")
    cases, provenance, review = [], [], []
    view_counts=[]
    for category, sequence, chosen in validated:
        family = "co3d-"+identifier(category+"/"+sequence)
        evidence, boxes = [], {}
        view_counts.append(len(chosen))
        rng = random.Random(identifier(str(seed)+family))
        for i,(frame,box) in enumerate(chosen):
            photo = "photo-"+identifier(frame["image"]["path"])
            target = assets/(photo+".jpg")
            link_asset(source/frame["image"]["path"],target)
            boxes[photo] = box
            evidence.append({"id":photo,"source_id":photo,"path":"assets/"+target.name,
                "party":rng.choice(["A","B"]),"object":"item","time":"capture", "view":f"view-{i+1:02d}","available":True})
            provenance.append({"family_id":family,"image_id":photo,"source_path":frame["image"]["path"],
                "sequence":sequence,"category":category,"split":split,"frame_number":frame["frame_number"],
                "frame_timestamp_seconds":frame["frame_timestamp"],"source_mask":frame["mask"]["path"],
                "viewpoint":frame.get("viewpoint"),"source_meta":frame.get("meta"),"source_frame_annotation":frame,
                "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
        first, second = evidence[0], evidence[-1]
        with Image.open(output/second["path"]) as image:
            preview = image.convert("RGB"); source_size=list(preview.size); preview.thumbnail((48,48))
            preview_path=assets/(second["id"]+"-preview.jpg");preview.save(preview_path,quality=85)
        preview_row = {**second,"id":second["id"]+"-preview","path":"assets/"+preview_path.name,
                       "source_size":source_size,"source_bbox":[0,0,*source_size],"view":"preview", "camera_view":second["view"]}
        endpoint = lambda row: {"image_id":row["id"],"bbox":boxes[row["id"]],"time":"capture"}
        identity = {"relation":"same_object","left":endpoint(first),"right":endpoint(second),"min_iou":.5}
        annotation = {"protocol":"evidence-chain-v1","status":"source-derived-unreviewed","verdict":"Supported",
            "subclaims":[{"id":"identity","kind":"identity","truth":"Supported", "minimal_evidence_sets":{"Supported":[[identity]]}}]}
        for variant in ("Sufficient","Obtainable","Missing","Unavailable"):
            rows = deepcopy(evidence)+[deepcopy(preview_row)]
            if variant == "Missing": rows = [r for r in rows if r["id"] != second["id"]]
            if variant == "Unavailable":
                for row in rows:
                    if row["id"] == second["id"]: row["available"] = False
            cases.append({"case_id":family+"-"+variant,"family_id":family,"variant":variant,"split":split,
                "category":category,"claim":f"The objects in {first['id']} and {second['id']} are the same physical item.",
                "claim_parts":[{"id":"identity","text":"The two pictured objects are the same physical item."}],
                "scene":{"source_sequence":sequence,"category":category,"source_type":"captured_multiview"},
                "initial":[first["id"],second["id"] if variant=="Sufficient" else preview_row["id"]],
                "evidence":rows,"request_options":{"objects":["item"],"times":["capture"],
                    "views":[r["view"] for r in evidence]}, "annotation":deepcopy(annotation)})
        review.append({"family_id":family,"sequence":sequence,"claim_parts":cases[-1]["claim_parts"],
            "candidate_annotation":annotation,"review_status":"pending","annotators":[],
            "review_questions":["Is object identity visually determinable?", "Which regions identify the object?",
                "Are the proposed evidence sets actually minimal and sufficient?", "Can other released views bridge the identity?",
                "Is the preview already sufficient?", "Is the request outcome semantically missing/unavailable?"]})
    (output/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (output/"annotation_review.json").write_text(json.dumps(review,indent=2)+"\n")
    manifest = {"dataset":"CO3Dv2 multiview identity candidates","source_manifest":read_json(source/"source_manifest.json"),
        "families":families,"cases":len(cases),"original_images":sum(view_counts),"max_views_per_family":views,
        "views_per_family":{"min":min(view_counts),"max":max(view_counts),"mean":sum(view_counts)/len(view_counts)},
        "categories":{c:sum(row[0]==c for row in validated) for c in categories},"split":split,"seed":seed,
        "scope":"positive-identity integration candidates; no hard-negative or damage-change benchmark claim",
        "annotation_status":"source-derived-unreviewed","grounding_protocol":"evidence-chain-v1",
        "uncertainty":"source identity and masks are proxies; sufficiency/availability require independent review",
        "grouping":"official sequence; variants are dependent; sequences crossing official task partitions excluded",
        "frame_selection":"uniform frame targets with nearest nonempty official foreground masks; zero-mask placeholders excluded; source frame roles retained",
        "time":"capture denotes one video; frame timestamps are not before/after damage labels",
        "license":"CC-BY-NC-4.0; retain official attribution and source license"}
    (output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--categories",nargs="+",default=["chair","cup","bottle"])
    parser.add_argument("--families",type=int,default=300);parser.add_argument("--views",type=int,default=8)
    parser.add_argument("--split",choices=["train","val","test"],default="test");parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.output,args.categories,args.families,args.views,args.split,args.seed),indent=2))


if __name__ == "__main__": main()
