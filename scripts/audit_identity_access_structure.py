"""Audit recorded-frame access in frozen identity cases; this does not assess visual sufficiency."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

def required_images(case):
    ids=set()
    for subclaim in case["annotation"]["subclaims"]:
        for evidence_sets in subclaim["minimal_evidence_sets"].values():
            for evidence_set in evidence_sets:
                for requirement in evidence_set:
                    if "relation"in requirement:
                        ids.update(requirement[k]["image_id"]for k in ["left","right"])
                    else:ids.add(requirement["image_id"])
    return ids

def audit(dataset,cohort,provenance=None):
    source_bytes=Path(dataset).read_bytes()
    selection_bytes=Path(cohort).read_bytes()
    selected=json.loads(selection_bytes)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    if selected["dataset_sha256"]!=sha(source_bytes):
        raise ValueError("Dataset does not match frozen cohort")
    cases=[c for c in json.loads(source_bytes)if c["family_id"]in selected["family_ids"]and c["variant"]=="Obtainable"and c["task"]=="identity"]
    metadata_bytes=Path(provenance).read_bytes()if provenance else None
    meta={r["image_id"]:r for r in json.loads(metadata_bytes)}if metadata_bytes else{}
    families=[]
    for case in cases:
        evidence={e["id"]:e for e in case["evidence"]}
        initial=set(case["initial"])
        initial_sources={evidence[i].get("source_id",i)for i in initial}
        hidden=required_images(case)-initial
        new=[e for e in case["evidence"]if e["available"]and e.get("source_id",e["id"])not in initial_sources]
        angles=[]
        for image in new:
            for old_id in initial:
                old=evidence[old_id]
                if image["object"]!=old["object"]:continue
                a=meta.get(image.get("source_id",image["id"]))
                b=meta.get(old.get("source_id",old_id))
                if not a or not b:continue
                trace=sum(a["viewpoint"]["R"][i][j]*b["viewpoint"]["R"][i][j]for i in range(3)for j in range(3))
                angle=math.degrees(math.acos(max(-1,min(1,(trace-1)/2))))
                angles.append(angle)
        families.append({"family_id":case["family_id"],"construction":case["provenance"]["construction"],
            "hidden_source_minset_ids":sorted(hidden),
            "hidden_sources_all_already_present":all(evidence[i].get("source_id",i)in initial_sources for i in hidden),
            "new_source_frames_in_source_minsets":sum(evidence[i].get("source_id",i)not in initial_sources for i in hidden),
            "other_available_new_frames":len(new),"relative_camera_orientation_degrees":angles})
    summary={"scope":"Structural audit of source-derived, unreviewed evidence sets; not human sufficiency labels",
        "dataset_sha256":sha(source_bytes),"cohort_sha256":sha(selection_bytes),
        "source_metadata_sha256":sha(metadata_bytes)if metadata_bytes else None,
        "identity_families":len(families),
        "source_constructions":dict(Counter(x["construction"]for x in families)),
        "hidden_required_sources_already_initial":sum(x["hidden_sources_all_already_present"]for x in families),
        "families_with_new_source_frames_in_source_minsets":sum(x["new_source_frames_in_source_minsets"]>0 for x in families),
        "families_with_other_new_recorded_frames_available":sum(x["other_available_new_frames"]>0 for x in families),
        "families_with_available_camera_orientation_change_over_5deg":sum(any(a>5 for a in x["relative_camera_orientation_degrees"])for x in families)if provenance else None,
        "caveat":"Different camera orientation can include roll. Available new frames do not establish that a claim needs them."}
    return summary,families
def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset",help="Frozen cases.json");p.add_argument("cohort");p.add_argument("output")
    p.add_argument("--provenance");p.add_argument("--details",action="store_true")
    a=p.parse_args();summary,rows=audit(a.dataset,a.cohort,a.provenance)
    out=Path(a.output)
    if out.exists():raise ValueError("Choose a new output; existing audits are preserved")
    out.write_text(json.dumps({"summary":summary,"families":rows}if a.details else summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
