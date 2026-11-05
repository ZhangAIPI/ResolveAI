"""Aggregate recorded Obtainable episodes without model inference or database writes."""
from pathlib import Path
from collections import Counter
from contextlib import closing
from functools import lru_cache
import argparse
import hashlib
import json
import sqlite3
import sys

ARMS=("initial_text","full_available_text","interactive_agent")
VARIANT="Obtainable"
TASKS=("overall","state","identity")
REVIEWED_METRICS=("grounded_accuracy_pct","evidence_found_rate_pct","correct_evidence_use_rate_pct","unsupported_decision_rate_pct")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"))

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def semantic_signature(row,trace):
    return canonical({"result":row,"messages":trace.get("messages"),"tools":trace.get("tools"),"decision":trace.get("decision")})

def admit(index,key,item):
    if key in index:
        previous=index[key]
        if semantic_signature(previous["result"],previous["trace"])!=semantic_signature(item["result"],item["trace"]):
            raise ValueError("Conflicting recorded attempts for "+str(key))
        previous["provenance"].extend(item["provenance"])
        return False
    index[key]=item
    return True

def parts(message):
    content=message.get("content",[])
    return content if isinstance(content,list) else []

@lru_cache(maxsize=40000)
def pixel_hash(path):
    """Hash displayed RGB pixels and shape, independent of image file encoding."""
    from PIL import Image
    with Image.open(path) as image:
        rgb=image.convert("RGB")
        return hashlib.sha256(canonical({"size":rgb.size,"mode":"RGB"}).encode()+b"\0"+rgb.tobytes()).hexdigest()

def image_fingerprint(part,directory):
    ref=part.get("image_ref")
    if not ref:raise ValueError("Recorded image lacks image_ref")
    rel=Path(ref)
    if rel.is_absolute() or ".." in rel.parts:raise ValueError("Unexpected image_ref path")
    return pixel_hash(str(directory/rel))

def request_behavior(trace,directory,fingerprint=image_fingerprint):
    has_request=any(message.get("name")=="request_photo" or any(
        call.get("function",{}).get("name")=="request_photo" for call in message.get("tool_calls",[]))
        for message in trace.get("messages",[]))
    if not has_request:return []
    seen=set()
    functions={}
    events=[]
    for message in trace.get("messages",[]):
        if message.get("role")=="assistant":
            for call in message.get("tool_calls",[]):
                functions[call.get("id")]=call.get("function",{})
        displayed={fingerprint(p,directory) for p in parts(message) if p.get("type")=="image"}
        function=functions.get(message.get("tool_call_id"),{})
        if message.get("role")=="tool" and (message.get("name")=="request_photo" or function.get("name")=="request_photo"):
            status=None
            for part in parts(message):
                if part.get("type")!="text":continue
                try:value=json.loads(part["text"])
                except (ValueError,TypeError):continue
                if isinstance(value,dict) and "status" in value:status=value["status"]
            arguments=function.get("arguments",{})
            if isinstance(arguments,str):
                try:arguments=json.loads(arguments)
                except ValueError:arguments={}
            query=arguments.get("query",{}) if isinstance(arguments,dict) else {}
            events.append({"status":status,"new_pixel_images":len(displayed-seen),
                           "repeated_only":bool(displayed) and not bool(displayed-seen),
                           "query_view":query.get("view"),"returned_images":len(displayed)})
        seen.update(displayed)
    return events

def efficiency(items,api_helper=None):
    values=[]
    for item in items:
        row,trace=item["result"],item["trace"]
        if item.get("api"):
            if api_helper is None:raise ValueError("API usage helper required")
            value=api_helper(row,trace)
        else:
            generations=[g for g in trace.get("generation",[]) if "output_tokens" in g]
            wall=trace.get("wall_latency_s",row.get("latency_s"))
            if row.get("latency_s") is not None and abs(wall-row["latency_s"])>1e-6:
                raise ValueError("Stored latency mismatch")
            value={"wall_s":wall,
              "input_tokens":sum(g.get("input_tokens",0) for g in generations),
              "output_tokens":sum(g.get("output_tokens",0) for g in generations),
              "model_calls":len(generations),
              "rejected_calls":len(trace.get("generation",[]))-len(generations)}
        values.append(value)
    # Decisions exposes action scores rather than comparable generated-text usage.
    applicable=not any(item.get("decisions") for item in items)
    return {"episodes":len(items),
      "mean_episode_wall_s":sum(v["wall_s"] for v in values)/len(values),
      "mean_input_tokens":sum(v["input_tokens"] for v in values)/len(values) if applicable else None,
      "mean_output_tokens":sum(v["output_tokens"] for v in values)/len(values) if applicable else None,
      "mean_model_calls":sum(v["model_calls"] for v in values)/len(values),
      "mean_rejected_calls":sum(v["rejected_calls"] for v in values)/len(values),
      "token_counts_applicable":applicable,
      "accounting":"All expected episodes, including failures and abstentions. API replay timing follows score_api_results.api_usage; API latency is not an equal-hardware GPU comparison."}

def paired_analysis(indexed,cases,families):
    wanted={(family,arm) for family in families for arm in ("initial_text","interactive_agent")}
    if not wanted<=set(indexed):return None
    output={"scope":"Descriptive paired source-label outcomes. Requested/no_request strata are post-treatment, selected by the interactive policy; they do not identify a causal effect of a photo request.",
      "variant":VARIANT,"by_task":{}}
    for task in TASKS:
        strata={}
        selected=[f for f in sorted(families) if task=="overall" or cases[f]["task"]==task]
        for label in ("all","requested","no_request"):
            counts=Counter()
            provenance=[]
            for family in selected:
                before=indexed[(family,"initial_text")]
                after=indexed[(family,"interactive_agent")]
                row0,row1=before["result"],after["result"]
                requested=row1.get("requests",0)>0
                if label=="requested" and not requested:continue
                if label=="no_request" and requested:continue
                a=row0.get("verdict")==row0["source_label"]
                b=row1.get("verdict")==row1["source_label"]
                counts["episodes"]+=1
                counts["initial_correct"]+=int(a)
                counts["agent_correct"]+=int(b)
                counts[("correct" if a else "wrong")+"_"+("correct" if b else "wrong")]+=1
                if a and not b:
                    answer=row1.get("verdict")
                    reason=("no_valid_final" if answer not in ("Supported","Refuted","Need more evidence") else
                            "abstained" if answer=="Need more evidence" else "wrong_definite")
                    counts["lost_correct_"+reason]+=1
                provenance.append({"family_id":family,
                     "initial_traces":before["provenance"],"interactive_traces":after["provenance"]})
            for key in ("episodes","initial_correct","agent_correct","correct_correct","correct_wrong","wrong_correct","wrong_wrong",
                        "lost_correct_no_valid_final","lost_correct_abstained","lost_correct_wrong_definite"):
                counts.setdefault(key,0)
            n=counts["episodes"]
            strata[label]={"episodes":n,"counts":dict(counts),
               "net_correct_change":counts["agent_correct"]-counts["initial_correct"],
               "accuracy_difference_pp":100*(counts["agent_correct"]-counts["initial_correct"])/n if n else None,
               "ordered_family_trace_digest":digest(canonical(provenance))}
            assert counts["correct_wrong"]==sum(counts[k] for k in ("lost_correct_no_valid_final","lost_correct_abstained","lost_correct_wrong_definite"))
        assert strata["all"]["episodes"]==strata["requested"]["episodes"]+strata["no_request"]["episodes"]
        output["by_task"][task]=strata
    return output

def summarize(items,expected,api_helper=None):
    if len(items)!=expected:return None
    counts=Counter()
    views=Counter()
    statuses=Counter()
    for item in items:
        row,trace=item["result"],item["trace"]
        requests=request_behavior(trace,item["directory"])
        counts["source_label_correct"]+=int(row.get("verdict")==row["source_label"])
        counts["finished"]+=int(row.get("termination")=="finished" and row.get("verdict") in ("Supported","Refuted","Need more evidence"))
        counts["abstained"]+=int(row.get("verdict")=="Need more evidence")
        counts["definite"]+=int(row.get("verdict") in ("Supported","Refuted"))
        counts["episodes_with_request"]+=int(row.get("requests",0)>0)
        counts["request_calls"]+=row.get("requests",0)
        counts["request_responses"]+=len(requests)
        counts["episodes_with_new_pixels"]+=int(any(r["new_pixel_images"]>0 for r in requests))
        counts["new_pixel_images"]+=sum(r["new_pixel_images"] for r in requests)
        counts["episodes_with_repeated_image_request"]+=int(any(r["repeated_only"] for r in requests))
        counts["repeated_image_request_calls"]+=sum(int(r["repeated_only"]) for r in requests)
        for r in requests:
            views[str(r["query_view"])]+=1
            statuses[str(r["status"])]+=1
    assert counts["episodes_with_new_pixels"]<=counts["episodes_with_request"]<=expected
    assert counts["episodes_with_repeated_image_request"]<=counts["episodes_with_request"]
    assert counts["finished"]==counts["abstained"]+counts["definite"]
    answer_rows={x["result"]["family_id"]:x["result"] for x in items}
    provenance=[{"family_id":x["result"]["family_id"],"traces":x["provenance"]} for x in sorted(items,key=lambda x:x["result"]["family_id"])]
    return {"episodes":expected,"counts":dict(counts),
      "source_label_accuracy_pct":100*counts["source_label_correct"]/expected,
      "finished_pct":100*counts["finished"]/expected,
      "abstained_pct":100*counts["abstained"]/expected,
      "definite_pct":100*counts["definite"]/expected,
      "request_pct":100*counts["episodes_with_request"]/expected,
      "new_pixel_image_pct":100*counts["episodes_with_new_pixels"]/expected,
      "repeated_image_request_pct":100*counts["episodes_with_repeated_image_request"]/expected,
      "request_views":dict(views),"request_status":dict(statuses),
      "reviewed_metrics":{key:None for key in REVIEWED_METRICS},
      "efficiency":efficiency(items,api_helper),
      "trace_provenance":{"ordered_family_trace_digest":digest(canonical(provenance)),
        "source_databases":sorted({p["database"] for x in items for p in x["provenance"]}),
        "trace_count":sum(len(x["provenance"]) for x in items),
        "unique_episode_count":expected}}

def profiles(repository,runtime):
    old=runtime/"evaluation/cvpr-20261101"
    priority=runtime/"evaluation/cvpr-paper-priority"
    api=runtime/"evaluation/openai"
    reference=json.loads((repository/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    displays={"qwen35":"Qwen3.5-9B","qwen3":"Qwen3-VL-8B","qwen25":"Qwen2.5-VL-7B","internvl35":"InternVL3.5-8B",
      "qwen3_2b":"Qwen3-VL-2B","qwen3_4b":"Qwen3-VL-4B","llava_onevision":"LLaVA-OneVision-7B",
      "smolvlm2":"SmolVLM2-2.2B","internvl14":"InternVL3.5-14B","pixtral":"Pixtral-12B"}
    output=[]
    for w in reference["workers"]:
        name=w["name"]
        if name not in displays:continue
        directory=(old if name in {"qwen35","qwen3","qwen25","internvl35"} else priority)/name
        output.append({"name":name,"label":displays[name],"group":"Open weights","model_id":w["model_id"],"directories":[directory]})
    api_workers={w["name"]:w for w in json.loads((repository/"docs/results/openai_candidate_scores.json").read_text())["workers"]}
    for name,label in [("gpt61_sol","GPT-6.1 Sol"),("terra-eval","GPT-6.1 Terra"),("gpt54_mini","GPT-5.4 mini"),
                       ("gpt41_mini","GPT-4.1 mini"),("luna","GPT-6 Luna"),("luna_decisions_tools","GPT-6 Luna (Decisions)")]:
        directories=[api/name]+sorted(p for p in api.glob(name+"-obtainable*") if p.is_dir())
        resumed=priority/"reviewer-api-obtainable"/name
        if resumed.exists():directories.append(resumed)
        output.append({"name":name,"label":label,"group":"API","model_id":api_workers[name]["model_id"],
          "additional_interface":name=="luna_decisions_tools","directories":directories})
    assert len(output)==16 and len({x["model_id"] for x in output})==15
    return output

def aggregate(repository,runtime,output):
    sys.path.insert(0,str(repository))
    from scripts.score_api_results import api_usage
    reference=json.loads((repository/"docs/results/cvpr_20261102_candidate_scores.json").read_text())
    families=set(reference["family_ids"]);assert len(families)==209
    cases_path=runtime/"benchmark-v041/cases.json"
    payload=cases_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==reference["dataset_sha256"]
    cases={c["family_id"]:c for c in json.loads(payload) if c["variant"]=="Sufficient" and c["family_id"] in families}
    assert Counter(c["task"] for c in cases.values())=={"state":153,"identity":56}
    expected_by_task={"overall":209,"state":153,"identity":56}
    result={"schema_version":"resolveai-obtainable-audit-v2","scope":"Unreviewed source-label agreement and observed behavior; no model inference or database mutation.",
      "variant":VARIANT,"dataset_sha256":reference["dataset_sha256"],"family_ids":sorted(families),
      "task_counts":expected_by_task,"distinct_models":15,"interfaces":16,
      "definitions":{
         "source_label_accuracy":"Final answer equals the source-derived Supported/Refuted label. All expected attempts remain in the denominator; abstentions and invalid/no final answers score zero.",
         "new_pixel_image":"A request returns an RGB image whose shape-and-pixel SHA256 is absent from all preceding observations. It does not certify new independent or sufficient evidence.",
         "repeated_image_request":"At least one request returns images already observed, with no previously unseen image in that response. An episode can also contain a different request for a new image.",
         "finished":"A valid final answer, including an explicit Need more evidence answer.",
         "complete_key_gate":"An arm is scored only when all 209 frozen family keys are present exactly once after verified-equivalent replay deduplication.",
         "reviewed_metrics":"Grounded accuracy, evidence found, correct evidence use and unsupported-decision rate are unavailable without independent sufficient-evidence sets."},
      "models":[]}
    coverage=[]
    placeholders=",".join("?" for _ in families)
    case_ids=[f+"-"+VARIANT for f in sorted(families)]
    for spec in profiles(repository,runtime):
        indexed={}
        manifests=[]
        counts_by_variant=Counter()
        duplicates=0
        for directory in spec["directories"]:
            if not (directory/"episodes.sqlite").exists() or not (directory/"manifest.json").exists():continue
            manifest_bytes=(directory/"manifest.json").read_bytes()
            manifest=json.loads(manifest_bytes);config=manifest["configuration"]
            assert config["dataset_sha256"]==reference["dataset_sha256"],directory
            assert config["model_id"]==spec["model_id"],directory
            manifest_families=set(config.get("family_ids",config.get("families",[])))
            assert families<=manifest_families,directory
            manifests.append({"path":str((directory/"manifest.json").relative_to(runtime)),"sha256":hashlib.sha256(manifest_bytes).hexdigest(),
              "source_commit":manifest.get("source_commit"),"model_revision":config.get("revision"),
              "configuration_sha256":digest(canonical(config)),
              "public_evaluation_settings":{key:config.get(key) for key in
                ("budget","max_turns","max_assistant_turns","max_context_tokens","image_max_pixels","decoding","tool_adapter","assistant_prefix")},
              "admitted_obtainable_rows":0,"equivalent_replays":0,
              "configured_variants":config.get("variants"),"configured_arms":config.get("arms")})
            database=directory/"episodes.sqlite"
            with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True)) as db:
                raw_count=db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                for raw, in db.execute("SELECT result FROM episodes"):
                    row=json.loads(raw)
                    if row["family_id"] in families:counts_by_variant[(row["variant"],row["arm"])]+=1
                for raw,raw_trace in db.execute("SELECT result,trace FROM episodes WHERE case_id IN ("+placeholders+")",case_ids):
                    row=json.loads(raw);trace=json.loads(raw_trace)
                    family=row["family_id"]
                    assert family in families and row["variant"]==VARIANT
                    assert row["case_id"]==family+"-"+VARIANT
                    assert row["task"]==cases[family]["task"]
                    assert row["source_label"]==cases[family]["provenance_truth"]
                    assert row["arm"] in ARMS
                    key=(family,row["arm"])
                    item={"result":row,"trace":trace,"directory":directory,"api":spec["group"]=="API",
                      "decisions":bool(spec.get("additional_interface")),"provenance":[
                        {"database":str(database.relative_to(runtime)),"trace_sha256":hashlib.sha256(raw_trace.encode()).hexdigest()}]}
                    manifests[-1]["admitted_obtainable_rows"]+=1
                    if not admit(indexed,key,item):
                        duplicates+=1
                        manifests[-1]["equivalent_replays"]+=1
            manifests[-1]["raw_episodes"]=raw_count
        seen_by_arm={arm:sum((f,arm) in indexed for f in families) for arm in ARMS}
        task_seen={arm:{t:sum((f,arm) in indexed for f in families if cases[f]["task"]==t) for t in ("state","identity")} for arm in ARMS}
        entry={k:v for k,v in spec.items() if k!="directories"}
        entry.update({"coverage_by_arm":seen_by_arm,"task_coverage_by_arm":task_seen,
          "complete_three_arms":all(n==209 for n in seen_by_arm.values()),"complete_interactive":seen_by_arm["interactive_agent"]==209,
          "equivalent_replay_duplicates":duplicates,"manifests":manifests,"arms":{}})
        for arm in ARMS:
            complete=seen_by_arm[arm]==209
            entry["arms"][arm]={"complete":complete,"present":seen_by_arm[arm],"expected":209,
              "by_task":{task:summarize([x for (f,a),x in indexed.items() if a==arm and (task=="overall" or cases[f]["task"]==task)],expected,api_usage)
                 if complete else None for task,expected in expected_by_task.items()}}
        entry["paired_analysis"]=paired_analysis(indexed,cases,families)
        result["models"].append(entry)
        coverage.append({k:entry[k] for k in ["name","label","model_id","group","coverage_by_arm","task_coverage_by_arm",
             "complete_three_arms","complete_interactive","equivalent_replay_duplicates","manifests"]})
        coverage[-1]["recorded_variant_arm_counts_including_replays"]={v+":"+a:n for (v,a),n in sorted(counts_by_variant.items())}
        print(spec["name"],seen_by_arm,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/"obtainable_results.json").write_text(json.dumps(result,indent=2)+"\n")
    (output/"obtainable_coverage.json").write_text(json.dumps({"dataset_sha256":reference["dataset_sha256"],"family_ids":sorted(families),
      "task_counts":expected_by_task,"models":coverage},indent=2)+"\n")
    render(result,output)
    return result

def render(result,output):
    header=["Model","Initial acc.","Full-access acc.","Agent condition","Agent identity","Agent overall","Request","New image","Repeated image","Finish"]
    markdown=["# Obtainable: source-label agreement and recorded behavior","",
      "Fixed 209 families: 153 condition and 56 identity. Accuracy, requests and completion use every expected family; missing or abstaining final answers remain in the denominator.",
      "Each arm requires all 209 episode keys before any score is shown. -- means unavailable, never zero. New pixels do not certify sufficiency.",
      "Grounded accuracy, evidence-found rate, correct-evidence-use rate and unsupported-decision rate are null until independent evidence sets exist.","",
      "| "+" | ".join(header)+" |","| "+" | ".join(["---"]+["---:"]*(len(header)-1))+" |"]
    tex=[r"\begin{table*}[t]\centering\small",r"\setlength{\tabcolsep}{3.2pt}",
      r"\caption{\textbf{Obtainable evidence: source-label agreement and observed search.} Percentages on the frozen 209 families (153 condition, 56 identity); scores require every episode key in an arm. New image counts requests returning previously unseen pixels, not reviewed sufficient evidence. -- denotes unmeasured results.}",
      r"\label{tab:obtainable-all}",r"\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule",
      r" & Initial & Full access & \multicolumn{3}{c}{Interactive accuracy} & \multicolumn{4}{c}{Interactive behavior}\\",
      r"\cmidrule(lr){4-6}\cmidrule(l){7-10}",
      r"Model & Overall & Overall & Condition & Identity & Overall & Request & New image & Repeated & Finish\\\midrule"]
    def value(group,key):
        return "--" if group is None else f"{group[key]:.1f}"
    previous=None
    for m in result["models"]:
        if previous!=m["group"]:
            tex.append(r"\multicolumn{10}{l}{\emph{"+m["group"]+r"}}\\")
            previous=m["group"]
        arms=m["arms"];ini=arms["initial_text"]["by_task"]["overall"];full=arms["full_available_text"]["by_task"]["overall"]
        a=arms["interactive_agent"]["by_task"]
        cells=[m["label"],value(ini,"source_label_accuracy_pct"),value(full,"source_label_accuracy_pct"),
          value(a["state"],"source_label_accuracy_pct"),value(a["identity"],"source_label_accuracy_pct"),value(a["overall"],"source_label_accuracy_pct")]
        cells += [value(a["overall"],k) for k in ["request_pct","new_pixel_image_pct","repeated_image_request_pct","finished_pct"]]
        markdown.append("| "+" | ".join(cells)+" |")
        tex.append(" & ".join(cells)+r"\\")
    tex.extend([r"\bottomrule\end{tabular}",r"\end{table*}"])
    markdown.extend(["","Coverage by arm (initial / full / interactive):",""]+
       [f"- {m['label']}: "+"/".join(str(m["coverage_by_arm"][a]) for a in ARMS)+" of 209." for m in result["models"]])
    markdown.append("\nThe Decisions row is an additional interface for GPT-6 Luna, not a sixteenth distinct model.")
    (output/"obtainable_table.md").write_text("\n".join(markdown)+"\n")
    (output/"obtainable_table.tex").write_text("\n".join(tex)+"\n")

def self_test():
    row={"family_id":"x","task":"state","verdict":None,"source_label":"Supported","termination":"max_turns","requests":0}
    trace={"messages":[],"generation":[],"wall_latency_s":2.0}
    def item(r=row,t=trace):
        return {"result":dict(r),"trace":t,"directory":Path("."),"provenance":[{"database":"fake.sqlite","trace_sha256":"abc"}]}
    assert summarize([item()],2) is None
    summary=summarize([item()],1)
    assert summary["source_label_accuracy_pct"]==0 and summary["finished_pct"]==0
    assert all(x is None for x in summary["reviewed_metrics"].values())
    index={}
    assert admit(index,("x","interactive_agent"),item())
    assert not admit(index,("x","interactive_agent"),item())
    try:admit(index,("x","interactive_agent"),item({**row,"verdict":"Supported"}))
    except ValueError:pass
    else:raise AssertionError("Conflicting duplicates admitted")
    trace={"messages":[{"role":"user","content":[{"type":"image","image_ref":"old"}]},
      {"role":"assistant","tool_calls":[{"id":"1","function":{"name":"request_photo","arguments":{"query":{"view":"overview"}}}}]},
      {"role":"tool","tool_call_id":"1","content":[{"type":"text","text":'{"status":"provided"}'},{"type":"image","image_ref":"old"}]},
      {"role":"assistant","tool_calls":[{"id":"2","function":{"name":"request_photo","arguments":{"query":{"view":"original"}}}}]},
      {"role":"tool","tool_call_id":"2","content":[{"type":"text","text":'{"status":"provided"}'},{"type":"image","image_ref":"new"}]}]}
    events=request_behavior(trace,Path("."),lambda part,d:part["image_ref"])
    assert events[0]["repeated_only"] and events[0]["new_pixel_images"]==0
    assert not events[1]["repeated_only"] and events[1]["new_pixel_images"]==1
    import tempfile
    from PIL import Image, PngImagePlugin
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)
        image=Image.new("RGB",(3,2),(120,25,35))
        image.save(path/"a.png")
        metadata=PngImagePlugin.PngInfo()
        metadata.add_text("provenance","same RGB pixels, different encoded PNG")
        image.save(path/"b.png",pnginfo=metadata)
        Image.new("RGB",(3,2),(121,25,35)).save(path/"c.png")
        assert (path/"a.png").read_bytes()!=(path/"b.png").read_bytes()
        assert pixel_hash(str(path/"a.png"))==pixel_hash(str(path/"b.png"))
        assert pixel_hash(str(path/"a.png"))!=pixel_hash(str(path/"c.png"))
    good_row={**row,"family_id":"y","verdict":"Supported","termination":"finished"}
    good_trace={"messages":[],"generation":[{"input_tokens":20,"output_tokens":10}],"wall_latency_s":4.0}
    e=efficiency([item(),item(good_row,good_trace)])
    assert e["episodes"]==2 and e["mean_episode_wall_s"]==3.0
    assert e["mean_input_tokens"]==10 and e["mean_output_tokens"]==5 and e["mean_model_calls"]==0.5
    assert efficiency([{**item(good_row,good_trace),"decisions":True}])["mean_output_tokens"] is None
    # Paired failures, explicit abstentions, and wrong definite answers remain distinct.
    index={}
    cases={f:{"task":"state"} for f in ("a","b","c","d")}
    outcomes=[None,"Need more evidence","Refuted","Supported"]
    for f,answer in zip(cases,outcomes):
        initial={**row,"family_id":f,"verdict":"Supported","termination":"finished"}
        after={**initial,"verdict":answer,"requests":int(f=="a")}
        for arm,r in [("initial_text",initial),("interactive_agent",after)]:
            index[(f,arm)]={**item(r),"provenance":[{"database":"fake","trace_sha256":f+arm}]}
    pair=paired_analysis(index,cases,set(cases))["by_task"]["overall"]["all"]["counts"]
    assert pair["correct_wrong"]==3 and pair["lost_correct_no_valid_final"]==1
    assert pair["lost_correct_abstained"]==1 and pair["lost_correct_wrong_definite"]==1
    print("PASS meaningful invariants: complete gating, failed denominator, replay conflicts, normalized image identity, failure-inclusive efficiency, Decisions token N/A, and paired loss partition")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path,nargs="?")
    parser.add_argument("runtime",type=Path,nargs="?")
    parser.add_argument("output",type=Path,nargs="?")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not all([args.repository,args.runtime,args.output]):parser.error("repository runtime output required")
        aggregate(args.repository,args.runtime,args.output)
