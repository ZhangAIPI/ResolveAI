"""Blind visual-review UI and strict two-human-review admission for benchmark v0.4."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

from PIL import Image

from .grounding import full_pool_verdict, validate_annotation
from .public_data import VARIANTS
from .tools import VERDICTS


def validate_vote(vote):
    if not isinstance(vote, dict):
        raise ValueError("Review must be an object")
    if (
        vote.get("protocol") != "visual-review-v1"
        or vote.get("reviewer_type") != "human"
        or not isinstance(vote.get("reviewer_id"), str)
        or not vote["reviewer_id"].strip()
        or not vote.get("family_id")
        or not vote.get("reason", "").strip()
    ):
        raise ValueError("A named human review with a reason is required")
    if (
        not isinstance(vote.get("dataset_sha256"), str)
        or len(vote["dataset_sha256"]) != 64
    ):
        raise ValueError("Review must name the frozen dataset hash")
    for key in ["claim_clear", "alternatives_checked"]:
        if type(vote.get(key)) is not bool:
            raise ValueError("Review attestations must be boolean")
    for key in ["initial_verdicts", "pool_verdicts"]:
        if set(vote.get(key, {})) != set(VARIANTS) or any(
            v not in VERDICTS for v in vote[key].values()
        ):
            raise ValueError("Review all four initial and obtainable-pool conditions")
    if not isinstance(vote.get("annotation"), dict):
        raise ValueError("Review requires a structured evidence annotation")
    validate_annotation(vote["annotation"])
    if vote["annotation"].get("protocol") != "evidence-chain-v1":
        raise ValueError("Review requires explicit region/time or relation evidence")


def validate_truth_vote(vote):
    """A single full-pool judgment supplies truth, not availability labels."""
    if (
        not isinstance(vote, dict)
        or vote.get("protocol")
        not in {"visual-truth-review-v2", "visual-truth-review-v3"}
        or vote.get("scope") != "full_pool_only"
        or vote.get("reviewer_type") != "human"
        or not isinstance(vote.get("reviewer_id"), str)
        or not vote["reviewer_id"].strip()
        or not vote.get("family_id")
        or not vote.get("reason", "").strip()
    ):
        raise ValueError("A full-pool human truth review is required")
    if (
        not isinstance(vote.get("dataset_sha256"), str)
        or len(vote["dataset_sha256"]) != 64
    ):
        raise ValueError("Review must name the frozen dataset hash")
    if type(vote.get("claim_clear")) is not bool:
        raise ValueError("Claim clarity must be explicit")
    if "initial_verdicts" in vote or "pool_verdicts" in vote:
        raise ValueError("Single-answer reviews cannot claim availability judgments")
    if vote["protocol"] == "visual-truth-review-v3":
        if vote.get("verdict") not in VERDICTS:
            raise ValueError("Review requires a valid judgment")
        if "annotation" in vote:
            raise ValueError(
                "Optional-region reviews cannot claim sufficient annotations"
            )
        if not isinstance(vote.get("regions"), list) or not isinstance(
            vote.get("links"), list
        ):
            raise ValueError("Optional regions and links must be lists")
        for endpoint in vote["regions"]:
            box = endpoint.get("bbox", [])
            if (
                not endpoint.get("image_id")
                or not endpoint.get("time")
                or len(box) != 4
                or any(type(x) is not int for x in box)
                or not 0 <= box[0] < box[2]
                or not 0 <= box[1] < box[3]
            ):
                raise ValueError("Invalid optional region")
        for link in vote["links"]:
            if (
                link.get("relation") not in {"same_object", "different_object"}
                or any(link.get(k) not in vote["regions"] for k in ("left", "right"))
                or link["left"]["image_id"] == link["right"]["image_id"]
            ):
                raise ValueError("Invalid optional identity link")
        return
    validate_annotation(vote.get("annotation", {}))
    if vote["annotation"].get("protocol") != "evidence-chain-v1":
        raise ValueError("Review requires explicit evidence regions or relations")


def full_truth_consensus(votes, cases, dataset_hash):
    """Report full-material consensus without admitting unreviewed release variants."""
    families = {c["family_id"]: c for c in cases if c["variant"] == "Obtainable"}
    groups = defaultdict(list)
    for vote in votes:
        validate_truth_vote(vote)
        if vote["dataset_sha256"] != dataset_hash or vote["family_id"] not in families:
            raise ValueError("Truth review does not belong to this dataset")
        groups[vote["family_id"]].append(vote)
    accepted, pending = [], []
    for family_id, group in sorted(groups.items()):
        if len(group) != 2 or len({v["reviewer_id"].strip() for v in group}) != 2:
            pending.append(
                {"family_id": family_id, "reason": "two independent reviewers required"}
            )
            continue
        if not all(v["claim_clear"] for v in group):
            pending.append(
                {"family_id": family_id, "reason": "claim or target unclear"}
            )
            continue
        verdicts = {
            v["verdict"]
            if v["protocol"] == "visual-truth-review-v3"
            else v["annotation"]["verdict"]
            for v in group
        }
        if len(verdicts) != 1:
            pending.append(
                {"family_id": family_id, "reason": "human verdict disagreement"}
            )
            continue
        case = families[family_id]
        available = {e["id"]: e for e in case["evidence"] if e["available"]}
        for vote in group:
            if vote["protocol"] == "visual-truth-review-v3":
                endpoints = vote["regions"]
            else:
                requirements = [
                    r
                    for subclaim in vote["annotation"]["subclaims"]
                    for sets in subclaim["minimal_evidence_sets"].values()
                    for evidence_set in sets
                    for r in evidence_set
                ]
                endpoints = [
                    e
                    for r in requirements
                    for e in ([r["left"], r["right"]] if "relation" in r else [r])
                ]
            for endpoint in endpoints:
                if endpoint["image_id"] not in available:
                    raise ValueError("Review cites unavailable evidence")
                row = available[endpoint["image_id"]]
                with Image.open(Path(case["asset_root"]) / row["path"]) as picture:
                    size = row.get("source_size", list(picture.size))
                box = endpoint["bbox"]
                bounds = row.get("source_bbox", [0, 0, *size])
                if not (
                    bounds[0] <= box[0] < box[2] <= bounds[2]
                    and bounds[1] <= box[1] < box[3] <= bounds[3]
                    and endpoint["time"] == row["time"]
                ):
                    raise ValueError("Review region or time outside visible evidence")
        accepted.append(
            {
                "family_id": family_id,
                "verdict": next(iter(verdicts)),
                "reviewers": [v["reviewer_id"] for v in group],
                "full_pool_ids": sorted(available),
            }
        )
    return {
        "scope": "full_pool_label_only",
        "consensus": accepted,
        "pending": pending,
        "availability_review_complete": False,
        "formal_evaluation_ready": False,
    }


def merge_votes(votes, family):
    """Agreement is necessary; source truth alone never promotes a case."""
    if len(votes) != 2 or len({v["reviewer_id"].strip() for v in votes}) != 2:
        raise ValueError("Exactly two distinct human reviewers are required")
    for vote in votes:
        validate_vote(vote)
        if (
            vote["family_id"] != family[0]["family_id"]
            or not vote["claim_clear"]
            or not vote["alternatives_checked"]
        ):
            raise ValueError(
                "Unclear targets or unchecked alternatives cannot be admitted"
            )
    for key in ["initial_verdicts", "pool_verdicts"]:
        if votes[0][key] != votes[1][key]:
            raise ValueError(
                "Review disagreement requires adjudication, not a majority guess"
            )
    annotations = [v["annotation"] for v in votes]
    truth = lambda a: (
        a["verdict"],
        sorted((s["id"], s["kind"], s["truth"]) for s in a["subclaims"]),
    )
    if truth(annotations[0]) != truth(annotations[1]):
        raise ValueError("Subclaim truth disagreement requires adjudication")
    merged = deepcopy(annotations[0])
    for subclaim in merged["subclaims"]:
        other = next(
            s for s in annotations[1]["subclaims"] if s["id"] == subclaim["id"]
        )
        for verdict, groups in other["minimal_evidence_sets"].items():
            existing = subclaim["minimal_evidence_sets"].setdefault(verdict, [])
            for group in groups:
                if group not in existing:
                    existing.append(deepcopy(group))
    merged["status"] = "reviewed"
    validate_annotation(merged)
    full = next(c for c in family if c["variant"] == "Obtainable")
    evidence = {e["id"]: e for e in full["evidence"]}
    for s in merged["subclaims"]:
        for groups in s["minimal_evidence_sets"].values():
            for group in groups:
                for requirement in group:
                    endpoints = (
                        [requirement["left"], requirement["right"]]
                        if "relation" in requirement
                        else [requirement]
                    )
                    for endpoint in endpoints:
                        if endpoint["image_id"] not in evidence:
                            raise ValueError(
                                "Review cites an image outside the evidence library"
                            )
                        row = evidence[endpoint["image_id"]]
                        with Image.open(
                            Path(full["asset_root"]) / row["path"]
                        ) as picture:
                            size = row.get("source_size", list(picture.size))
                        x0, y0, x1, y1 = endpoint["bbox"]
                        if (
                            not 0 <= x0 < x1 <= size[0]
                            or not 0 <= y0 < y1 <= size[1]
                            or endpoint["time"] != row["time"]
                        ):
                            raise ValueError(
                                "Review source coordinates or timestamp invalid"
                            )
    for case in family:
        available = {e["id"] for e in case["evidence"] if e["available"]}
        initial = set(case["initial"])
        variant = case["variant"]
        if (
            full_pool_verdict(merged, available) != votes[0]["pool_verdicts"][variant]
            or full_pool_verdict(merged, initial)
            != votes[0]["initial_verdicts"][variant]
        ):
            raise ValueError(
                "Annotated alternatives do not explain the reviewed initial/pool verdicts"
            )
    return merged


def availability_class(initial, pool, oracle, evidence):
    """Proposal classes follow reviewed judgeability, not a missing-original flag."""
    if initial != "Need more evidence":
        return "Sufficient"
    if pool != "Need more evidence":
        return "Obtainable"
    if oracle == "Need more evidence":
        return "Underdetermined"
    return "Unavailable" if any(not row["available"] for row in evidence) else "Missing"


def admit(data, votes_path, output):
    if output.exists():
        raise ValueError("Use a fresh admitted dataset directory")
    cases = json.loads((data / "cases.json").read_text())
    dataset_hash = hashlib.sha256((data / "cases.json").read_bytes()).hexdigest()
    families, votes = defaultdict(list), defaultdict(list)
    for case in cases:
        families[case["family_id"]].append(case)
    for line in votes_path.read_text().splitlines():
        if line.strip():
            vote = json.loads(line)
            validate_vote(vote)
            if vote["dataset_sha256"] != dataset_hash:
                raise ValueError("Review belongs to a different frozen dataset")
            votes[vote["family_id"]].append(vote)
    if set(votes) - set(families):
        raise ValueError("Votes reference unknown families")
    admitted, pending = [], []
    for family_id, family in families.items():
        try:
            annotation = merge_votes(votes[family_id], family)
        except (ValueError, KeyError, TypeError) as error:
            pending.append({"family_id": family_id, "reason": str(error)})
            continue
        for case in family:
            case = deepcopy(case)
            case["annotation"] = deepcopy(annotation)
            release = case["variant"]
            case["review"] = {
                "protocol": "visual-review-v1",
                "reviewers": [v["reviewer_id"] for v in votes[family_id]],
                "initial_verdict": votes[family_id][0]["initial_verdicts"][release],
                "pool_verdict": votes[family_id][0]["pool_verdicts"][release],
            }
            case["release_configuration"] = release
            case["variant"] = availability_class(
                case["review"]["initial_verdict"],
                case["review"]["pool_verdict"],
                annotation["verdict"],
                case["evidence"],
            )
            admitted.append(case)
    output.mkdir(parents=True)
    (output / "cases.json").write_text(json.dumps(admitted, indent=2) + "\n")
    (output / "pending.json").write_text(json.dumps(pending, indent=2) + "\n")
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    manifest.update(
        stage="reviewed" if admitted else "no-admitted-cases",
        formal_evaluation_ready=bool(admitted),
        families=len(admitted) // 4,
        cases=len(admitted),
        pending_families=len(pending),
        parent_dataset=str(data),
        review_records_sha256=hashlib.sha256(votes_path.read_bytes()).hexdigest(),
    )
    manifest["availability_classes"] = dict(Counter(c["variant"] for c in admitted))
    manifest["release_configurations"] = dict(
        Counter(c["release_configuration"] for c in admitted)
    )
    manifest["scope"] = (
        "Two-human-reviewed development subset; source splits and shared-family limitations retained."
    )
    manifest["cases_sha256"] = hashlib.sha256(
        (output / "cases.json").read_bytes()
    ).hexdigest()
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def reviewed_outcome(environment, verdict):
    """Evaluator-only diagnosis: reasonable abstention is distinct from search success."""
    case = environment._case
    if case.get("review", {}).get("protocol") != "visual-review-v1":
        return {}
    annotation = case["annotation"]
    current = full_pool_verdict(annotation, environment._released)
    initial = case["review"]["initial_verdict"]
    obtainable = case["review"]["pool_verdict"]
    return {
        "reviewed_initial_verdict": initial,
        "reviewed_pool_verdict": obtainable,
        "current_evidence_verdict": current,
        "decision_matches_current_evidence": verdict == current,
        "acquisition_opportunity": initial == "Need more evidence"
        and obtainable != "Need more evidence",
        "unused_obtainable_evidence": current == "Need more evidence"
        and obtainable != "Need more evidence",
    }


def require_admitted(manifest, cases):
    """Formal v0.4 scoring cannot silently fall back to candidate proxy labels."""
    if manifest.get("protocol") != "benchmark-v0.4":
        return
    if not manifest.get("formal_evaluation_ready") or not cases:
        raise ValueError(
            "v0.4 curation pool is unreviewed; admit independent reviews before formal evaluation"
        )
    for case in cases:
        review = case.get("review", {})
        if (
            case["annotation"].get("status") != "reviewed"
            or review.get("protocol") != "visual-review-v1"
            or len(set(review.get("reviewers", []))) != 2
        ):
            raise ValueError(
                "Every v0.4 case requires two independent reviewed annotations"
            )
        for ids, key in [
            (case["initial"], "initial_verdict"),
            ([e["id"] for e in case["evidence"] if e["available"]], "pool_verdict"),
        ]:
            if full_pool_verdict(case["annotation"], set(ids)) != review.get(key):
                raise ValueError("Reviewed labels no longer match the evidence library")


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ResolveAI 题目审核</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#172332}
button,select,input{font:inherit;padding:7px;margin:4px}button{cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.card{background:#f4f6f8;padding:12px}.card img{max-width:100%;max-height:440px;object-fit:contain;cursor:crosshair}
textarea{width:100%;height:150px}label{display:inline-block;margin:6px}small{color:#536173}
</style>
<h1>看图片，确认这道题能不能判</h1>
<p>页面不显示来源答案。先检查初始材料，再检查可获取材料；最后查看完整资料库。正常参考不是同一实物的“之前照片”。</p>
<button onclick="load(Math.max(0,index-1))">上一题</button><input id="number" type="number" min="1" value="1">
<button onclick="load(Number(document.querySelector('#number').value)-1)">跳转</button><button onclick="load(index+1)">下一题</button>
<span id="counter"></span><h2 id="claim"></h2><p id="context"></p>
<select id="variant" onchange="render()"></select>
<select id="access" onchange="render()"><option value="initial">初始材料</option><option value="pool">该版本全部可获取材料</option><option value="oracle">完整资料库（仅审核）</option></select>
<p>勾选构成充分证据的图片。框默认指向公开目标；点击图片两角可改框。放大缩略图不会恢复细节。可以记录多组替代证据。</p>
<div id="pictures" class="grid"></div>
<h2>独立判断</h2>
<label>审核者 ID <input id="reviewer"></label>
<label><input type="checkbox" id="clear">题干目标明确</label>
<label><input type="checkbox" id="alternatives">已检查其他视角和缩略图可能构成的替代证据</label>
<p>完整资料库的判断：<select id="full"></select>。无法从像素可靠判断同一实物时选“需要更多证据”。</p>
<table id="judgments"><tr><th>版本</th><th>初始材料判断</th><th>全部可获取材料判断</th></tr></table>
<button onclick="addSet()">把所选区域记为一个充分集合</button><button onclick="sets=[];showSets()">清空集合</button>
<textarea id="annotation"></textarea>
<p>判断理由（描述可见线索或无法判断的原因）：<input id="reason" style="width:80%"></p>
<button onclick="save()">提交独立审核</button><p id="status"></p>
<script>
let index=0,data,sets=[],corners={},boxes={};
const labels={'Supported':'支持','Refuted':'反驳','Need more evidence':'需要更多证据'};
const variants=['Sufficient','Obtainable','Missing','Unavailable'];
function choices(id){let s=document.querySelector(id);s.innerHTML='<option value="">请选择</option>'+Object.entries(labels).map(([v,t])=>'<option value="'+v+'">'+t+'</option>').join('')}
document.querySelector('#variant').innerHTML=variants.map(v=>'<option>'+v+'</option>').join('');
choices('#full');
document.querySelector('#judgments').innerHTML+=variants.map(v=>'<tr><td>'+v+'</td><td><select id="initial-'+v+'"></select></td><td><select id="pool-'+v+'"></select></td></tr>').join('');
variants.forEach(v=>{choices('#initial-'+v);choices('#pool-'+v)});
async function load(i){
 let r=await fetch('/case/'+i);if(!r.ok){document.querySelector('#status').textContent='没有该题';return}
 data=await r.json();index=i;sets=[];boxes={};corners={};
 document.querySelector('#number').value=i+1;
 document.querySelector('#counter').textContent=(i+1)+' / '+data.total;
 document.querySelector('#claim').textContent=data.claim;
 document.querySelector('#context').textContent=data.context;
 document.querySelector('#clear').checked=false;document.querySelector('#alternatives').checked=false;
 document.querySelector('#reason').value='';document.querySelector('#status').textContent='';
 document.querySelector('#full').value='';
 variants.forEach(v=>{document.querySelector('#initial-'+v).value='';document.querySelector('#pool-'+v).value=''});
 render();showSets();
}
function render(){
 const v=document.querySelector('#variant').value,a=document.querySelector('#access').value;
 const ids=a==='oracle'?data.images.map(p=>p.id):data.versions[v][a];
 const container=document.querySelector('#pictures');container.innerHTML='';
 data.images.filter(p=>ids.includes(p.id)).forEach(p=>{
  boxes[p.id] ||= p.target_bbox||p.source_bbox;
  const c=document.createElement('div');c.className='card';
  const title=document.createElement('label');const cb=document.createElement('input');cb.type='checkbox';cb.dataset.photo=p.id;
  title.append(cb,document.createTextNode(p.object+' / '+p.view+' / '+p.id));c.append(title);
  const wrap=document.createElement('div');wrap.style.position='relative';wrap.style.width='fit-content';wrap.style.maxWidth='100%';
  const img=document.createElement('img');img.src='/image/'+index+'/'+encodeURIComponent(p.id);img.style.display='block';
  const overlay=document.createElement('canvas');overlay.style.cssText='position:absolute;inset:0;width:100%;height:100%;pointer-events:none';
  function draw(){
   overlay.width=img.naturalWidth;overlay.height=img.naturalHeight;
   const ctx=overlay.getContext('2d'),box=boxes[p.id],b=p.source_bbox;
   ctx.strokeStyle='#18aa6a';ctx.lineWidth=Math.max(2,overlay.width/180);
   ctx.strokeRect((box[0]-b[0])/(b[2]-b[0])*overlay.width,(box[1]-b[1])/(b[3]-b[1])*overlay.height,
   (box[2]-box[0])/(b[2]-b[0])*overlay.width,(box[3]-box[1])/(b[3]-b[1])*overlay.height);
  }
  img.onload=draw;wrap.append(img,overlay);
  const note=document.createElement('p');note.textContent='原图坐标框：'+JSON.stringify(boxes[p.id]);
  img.onclick=e=>{
   const b=img.getBoundingClientRect(),x=Math.round((e.clientX-b.left)/b.width*(p.source_bbox[2]-p.source_bbox[0])+p.source_bbox[0]),
   y=Math.round((e.clientY-b.top)/b.height*(p.source_bbox[3]-p.source_bbox[1])+p.source_bbox[1]);
   if(!corners[p.id]){corners[p.id]=[x,y];note.textContent='已选第一角，再点对角';return}
   const [x0,y0]=corners[p.id];delete corners[p.id];
   boxes[p.id]=[Math.min(x0,x),Math.min(y0,y),Math.max(x0,x),Math.max(y0,y)];
   note.textContent='原图坐标框：'+JSON.stringify(boxes[p.id]);draw();
  };c.append(wrap,note);container.append(c);
 });
}
function annotation(){
 const verdict=document.querySelector('#full').value,truth=verdict==='Need more evidence'?'uncertain':verdict;
 return {protocol:'evidence-chain-v1',verdict,subclaims:[{id:data.task==='state'?'condition':'identity',
 kind:data.task==='state'?'condition':'identity',truth,minimal_evidence_sets:truth==='uncertain'?{}:{[verdict]:sets}}]};
}
function showSets(){document.querySelector('#annotation').value=JSON.stringify(annotation(),null,2)}
document.querySelector('#full').onchange=()=>{sets=[];showSets()};
function addSet(){
 const chosen=[...document.querySelectorAll('input[data-photo]:checked')].map(c=>{
  const p=data.images.find(p=>p.id===c.dataset.photo);return {image_id:p.id,bbox:boxes[p.id],time:p.time};
 });
 const verdict=document.querySelector('#full').value;
 if(!verdict||verdict==='Need more evidence'){alert('先给出完整资料库的确定判断；不确定时无需充分集合');return}
 if(data.task==='identity'){
  if(chosen.length!==2){alert('身份关系选择两张不同图片');return}
  sets.push([{relation:verdict==='Supported'?'same_object':'different_object',left:chosen[0],right:chosen[1],min_iou:0.5}]);
 }else{if(!chosen.length){alert('请选择证据图片');return}sets.push(chosen)}
 showSets();
}
async function save(){
 const vote={protocol:'visual-review-v1',reviewer_type:'human',
 reviewer_id:document.querySelector('#reviewer').value,family_id:data.family_id,dataset_sha256:data.dataset_sha256,
 claim_clear:document.querySelector('#clear').checked,alternatives_checked:document.querySelector('#alternatives').checked,
 reason:document.querySelector('#reason').value,initial_verdicts:{},pool_verdicts:{},
 annotation:JSON.parse(document.querySelector('#annotation').value)};
 variants.forEach(v=>{vote.initial_verdicts[v]=document.querySelector('#initial-'+v).value;vote.pool_verdicts[v]=document.querySelector('#pool-'+v).value});
 const r=await fetch('/votes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(vote)});
 document.querySelector('#status').textContent=await r.text();
}
load(0);
</script></html>"""


def serve(data, votes_path, port):
    families = defaultdict(list)
    for case in json.loads((data / "cases.json").read_text()):
        families[case["family_id"]].append(case)
    entries = list(families.values())
    dataset_hash = hashlib.sha256((data / "cases.json").read_bytes()).hexdigest()
    votes_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, content, mime="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/":
                return self.respond(PAGE.encode(), "text/html; charset=utf-8")
            try:
                parts = self.path.split("/")
                index = int(parts[2])
                if index < 0:
                    raise ValueError("Negative family index")
                family = entries[index]
                case = next(c for c in family if c["variant"] == "Obtainable")
                rows = {r["id"]: r for r in case["evidence"]}
                if parts[1] == "image" and len(parts) == 4:
                    row = rows[unquote(parts[3])]
                    path = (Path(case["asset_root"]) / row["path"]).resolve()
                    if not path.is_relative_to(Path(case["asset_root"]).resolve()):
                        raise ValueError("Invalid asset")
                    return self.respond(
                        path.read_bytes(),
                        "image/png" if path.suffix == ".png" else "image/jpeg",
                    )
                if parts[1] != "case" or len(parts) != 3:
                    raise ValueError("Unknown route")
                images = []
                for row in rows.values():
                    with Image.open(Path(case["asset_root"]) / row["path"]) as picture:
                        size = row.get("source_size", list(picture.size))
                    images.append(
                        {k: row[k] for k in ["id", "object", "time", "view"]}
                        | {
                            "source_bbox": row.get("source_bbox", [0, 0, *size]),
                            "target_bbox": row.get("target_bbox"),
                        }
                    )
                payload = {
                    "dataset_sha256": dataset_hash,
                    "family_id": case["family_id"],
                    "claim": case["claim"],
                    "task": case["task"],
                    "context": case.get("task_instructions", "")
                    + " "
                    + " ".join(p["text"] for p in case.get("claim_parts", [])),
                    "total": len(entries),
                    "images": images,
                    "versions": {
                        c["variant"]: {
                            "initial": c["initial"],
                            "pool": [e["id"] for e in c["evidence"] if e["available"]],
                        }
                        for c in family
                    },
                }
                self.respond(json.dumps(payload).encode())
            except (ValueError, KeyError, IndexError, OSError):
                self.respond(b"Not found", "text/plain", 404)

        def do_POST(self):
            try:
                if (
                    self.path != "/votes"
                    or self.headers.get_content_type() != "application/json"
                ):
                    raise ValueError("Expected a JSON review")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100000:
                    raise ValueError("Invalid review size")
                vote = json.loads(self.rfile.read(length))
                validate_vote(vote)
                if vote["dataset_sha256"] != dataset_hash:
                    raise ValueError("Review dataset hash mismatch")
                if vote["family_id"] not in families:
                    raise ValueError("Unknown family")
                with votes_lock:
                    previous = (
                        [json.loads(s) for s in votes_path.read_text().splitlines()]
                        if votes_path.exists()
                        else []
                    )
                    if any(
                        v["family_id"] == vote["family_id"]
                        and v["reviewer_id"] == vote["reviewer_id"]
                        for v in previous
                    ):
                        raise ValueError(
                            "This reviewer already submitted this family; do not overwrite an independent review"
                        )
                    votes_path.parent.mkdir(parents=True, exist_ok=True)
                    with votes_path.open("a") as stream:
                        stream.write(json.dumps(vote, ensure_ascii=False) + "\n")
                self.respond(
                    "已保存。需要第二位独立审核者；一致并通过证据检查后才能入库。".encode(),
                    "text/plain; charset=utf-8",
                )
            except (ValueError, KeyError, TypeError):
                self.respond(
                    "审核不完整或格式无效；请检查四个版本、理由和证据集合。".encode(),
                    "text/plain; charset=utf-8",
                    400,
                )

    print(
        f"Blind review: http://127.0.0.1:{port}; source answers are hidden", flush=True
    )
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("serve")
    ui.add_argument("data", type=Path)
    ui.add_argument("votes", type=Path)
    ui.add_argument("--port", type=int, default=8765)
    promote = sub.add_parser("admit")
    promote.add_argument("data", type=Path)
    promote.add_argument("votes", type=Path)
    promote.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.data, args.votes, args.port)
    else:
        print(json.dumps(admit(args.data, args.votes, args.output), indent=2))


if __name__ == "__main__":
    main()
