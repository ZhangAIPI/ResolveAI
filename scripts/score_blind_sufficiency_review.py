"""Validate human exports; report agreement and pending adjudication, never fabricate ratings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
LABELS={"Supported","Refuted","Need more evidence"}
SUFF={"sufficient","insufficient","uncertain"}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def validate_answer(answer,unit):
    if answer.get("sufficiency")not in SUFF or answer.get("answer")not in LABELS:
        raise ValueError("Missing or invalid evidence judgment")
    if type(answer.get("claim_clear"))is not bool:
        raise ValueError("Claim clarity must be explicit")
    ids={i["id"]for i in unit["images"]}
    cites=answer.get("citations",[])
    if not isinstance(cites,list)or len(set(cites))!=len(cites)or not set(cites)<=ids:
        raise ValueError("Unknown or duplicated image citation")
    definite=answer["answer"]!="Need more evidence"
    if definite!=(answer["sufficiency"]=="sufficient"):
        raise ValueError("Definite answer and sufficient-evidence judgment conflict")
    if definite and not cites:raise ValueError("Definite judgment needs image citations")
    for r in answer.get("regions",[]):
        box=r.get("bbox")
        if r.get("image_id")not in ids or not isinstance(box,list)or len(box)!=4:
            raise ValueError("Unknown region reference")
        if not all(type(x)in(int,float)for x in box)or not 0<=box[0]<box[2]<=1 or not 0<=box[1]<box[3]<=1:
            raise ValueError("Region outside displayed pixels")
    return answer
def agreement(pairs,key):
    if not pairs:return {"n":0,"agreement":None,"fleiss_kappa":None}
    a=[x[0][key]for x in pairs];b=[x[1][key]for x in pairs];n=len(a)
    pa=sum(x==y for x,y in zip(a,b))/n
    ca,cb=Counter(a),Counter(b)
    pe=sum(((ca[k]+cb[k])/(2*n))**2 for k in set(a)|set(b))
    return {"n":n,"agreement":pa,"fleiss_kappa":None if pe==1 else(pa-pe)/(1-pe)}
def score(mapping,paths,adjudications=()):
    groups=defaultdict(list);reviewers={};seen_packets=set()
    for path in paths:
        payload=json.loads(Path(path).read_text())
        if payload.get("package_id")!=mapping["package_id"]or payload.get("format")!="resolveai-blind-sufficiency-v1":
            raise ValueError("Export belongs to another review package")
        if payload.get("reviewer_type")!="human"or payload.get("independent_review")is not True:
            raise ValueError("Human independent-review attestation required")
        pid=payload.get("packet_id");rid=payload.get("reviewer_id","").strip()
        if pid not in mapping["packets"]or not rid:raise ValueError("Unknown packet or reviewer")
        if pid in seen_packets:raise ValueError("Duplicate packet export")
        seen_packets.add(pid)
        if rid in reviewers and reviewers[rid]!=pid:raise ValueError("One person cannot supply independent votes under multiple packets")
        reviewers[rid]=pid;expected=set(mapping["packets"][pid]["unit_ids"]);got=set()
        for a in payload["answers"]:
            uid=a["unit_id"]
            if uid not in expected or uid in got:raise ValueError("Unassigned or duplicate answer")
            got.add(uid);validate_answer(a,mapping["units"][uid])
            groups[uid].append({**a,"reviewer_id":rid,"packet_id":pid})
        if got!=expected:raise ValueError("Packet export is incomplete")
    pairs=[];consensus={};pending=[]
    for uid,u in mapping["units"].items():
        votes=groups[uid]
        if len(votes)==2:
            if len({v["reviewer_id"]for v in votes})!=2:raise ValueError("Two different humans are required")
            pairs.append(votes)
            same=all(votes[0][k]==votes[1][k]for k in ["sufficiency","answer","claim_clear"])
            if same and votes[0]["claim_clear"]and votes[0]["sufficiency"]!="uncertain":
                consensus[uid]={k:votes[0][k]for k in ["claim_clear","sufficiency","answer"]}
                consensus[uid]["citations_by_reviewer"]=[v["citations"]for v in votes]
                consensus[uid]["regions_by_reviewer"]=[v.get("regions",[])for v in votes]
                consensus[uid]["reviewers"]=[v["reviewer_id"]for v in votes]
                consensus[uid]["adjudicated"]=False
            else:
                pending.append({"unit_id":uid,"reason":"disagreement, unclear claim, or unresolved reviewer uncertainty",
                    "votes":votes,"source_labels_hidden":True,
                    "adjudication_template":{"unit_id":uid,"reviewer_id":"","reviewer_type":"human",
                        "claim_clear":None,"sufficiency":None,"answer":None,
                        "citations":[],"regions":[],"comment":""}})
        elif len(votes)>2:raise ValueError("Unexpected extra votes")
    queue_ids={x["unit_id"]for x in pending}
    for record in adjudications:
        uid=record["unit_id"];rid=record.get("reviewer_id","").strip()
        if uid not in queue_ids or record.get("reviewer_type")!="human"or not rid:
            raise ValueError("Adjudication needs a pending item and named human")
        if rid in {v["reviewer_id"]for v in groups[uid]}:raise ValueError("Adjudicator must be independent of both raters")
        validate_answer(record,mapping["units"][uid])
        if not record["claim_clear"]or record["sufficiency"]=="uncertain":continue
        consensus[uid]={k:record[k]for k in ["claim_clear","sufficiency","answer"]}
        consensus[uid].update(citations_by_reviewer=[record["citations"]],
            regions_by_reviewer=[record.get("regions",[])],reviewers=[rid],adjudicated=True)
    family_status=[]
    for fid,f in mapping["families"].items():
        units=f["units"];initial=consensus.get(units["initial"]);pool=consensus.get(units["candidate_pool"])
        item={"family_id":fid,"task":f["task"],"source_label":f["source_label"],
            "main_review_complete":bool(initial and pool),"initial_to_pool_gap":None,
            "different_frame_helpful_candidate":None,"genuine_multiview_needed_validated":False,"blocked_pool_remains_unresolved":None}
        if initial and pool:
            item["initial_to_pool_gap"]=initial["sufficiency"]=="insufficient"and pool["sufficiency"]=="sufficient"
            item["initial_material_sufficient"]=initial["sufficiency"]=="sufficient"
            item["pool_answer"]=pool["answer"];item["pool_agrees_with_source"]=pool["answer"]==f["source_label"]
        if f["task"]=="identity":
            high=consensus.get(units["resolution_control"]);blocked=consensus.get(units["blocked_pool"])
            if high and pool:
                cited=[im["private"]for vote in pool["citations_by_reviewer"]for alias in vote
                    for im in mapping["units"][units["candidate_pool"]]["images"]if im["id"]==alias]
                initial_sources={im["private"]["source_id"]for im in mapping["units"][units["resolution_control"]]["images"]}
                cited_new=any(x["source_id"]not in initial_sources for x in cited)
                item["different_frame_helpful_candidate"]=high["sufficiency"]=="insufficient"and pool["sufficiency"]=="sufficient"and cited_new
                item["requires_geometric_and_citation_adjudication"]=item["different_frame_helpful_candidate"]
            if blocked:item["blocked_pool_remains_unresolved"]=blocked["sufficiency"]=="insufficient"
        family_status.append(item)
    report={"status":"no_human_annotations"if not sum(map(len,groups.values())) else"partial_blind_human_review",
        "received_packets":len(seen_packets),"human_votes":sum(map(len,groups.values())),
        "paired_materials":len(pairs),"consensus_materials":len(consensus),
        "adjudication_pending":sum(x["unit_id"]not in consensus for x in pending),
        "agreement":{"sufficiency":agreement(pairs,"sufficiency"),"answer":agreement(pairs,"answer")},
        "families_with_complete_initial_and_pool_review":sum(f["main_review_complete"]for f in family_status),
        "formal_grounded_accuracy_ready":False,
        "scope":"Independent material-set sufficiency judgments; image citations are retained as candidate evidence sets. This does not establish minimality, citation sufficiency, or policy use of pixels.",
        "family_results":family_status}
    return report,[x for x in pending if x["unit_id"]not in consensus]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("mapping");ap.add_argument("responses");ap.add_argument("output")
    ap.add_argument("--adjudications");a=ap.parse_args()
    mapping=json.loads(Path(a.mapping).read_text());responses=sorted(Path(a.responses).glob("*.json"))
    adj=json.loads(Path(a.adjudications).read_text())if a.adjudications else[]
    report,pending=score(mapping,responses,adj);out=Path(a.output)
    if out.exists():raise ValueError("Use a new output directory to preserve previous audits")
    out.mkdir(parents=True);dump(out/"summary.json",report);dump(out/"adjudication_queue.json",pending)
    print(json.dumps({k:v for k,v in report.items()if k!="family_results"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
