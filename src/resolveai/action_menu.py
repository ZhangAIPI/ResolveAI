"""Finite action choices constructed exclusively from the public conversation."""
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product
import json
import math
import random

from .tools import validate_action

VERSION = "public-menu-v1"


def public_state(public):
    views, state = {}, {}
    for message in public["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "text":
                continue
            try:
                value = json.loads(block["text"])
            except (ValueError, KeyError):
                continue
            if not isinstance(value, dict):
                continue
            for key in ("claim", "budget", "request_options", "tool_costs", "view_limits"):
                if key in value:
                    state[key] = value[key]
            if all(k in value for k in ("image_id", "view_id", "display_size", "source_bbox", "time")):
                views[value["view_id"]] = value
    return views, state


def regions(view):
    width, height = view["display_size"]
    boxes = [[0, 0, max(1,width//2), max(1,height//2)],
             [width//2, 0, width, max(1,height//2)],
             [0, height//2, max(1,width//2), height],
             [width//2, height//2, width, height],
             [width//4, height//4, max(width//4+1,3*width//4),
              max(height//4+1,3*height//4)]]
    if "target_bbox" in view:
        source, target = view["source_bbox"], view["target_bbox"]
        sx, sy = width/(source[2]-source[0]), height/(source[3]-source[1])
        box = [max(0,math.floor((target[0]-source[0])*sx)),
               max(0,math.floor((target[1]-source[1])*sy)),
               min(width,math.ceil((target[2]-source[0])*sx)),
               min(height,math.ceil((target[3]-source[1])*sy))]
        boxes.insert(0,box)
    return [list(box) for box in dict.fromkeys(tuple(b) for b in boxes)
            if 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height
            and list(box)!=[0,0,width,height]]


def action_candidates(public, limit=64):
    """Offer concrete calls, without consulting an Environment, label or asset path."""
    if not 16<=limit<=255:
        raise ValueError("Action-menu size must be between 16 and 255")
    views, state = public_state(public)
    available = {t["function"]["name"] for t in public["tools"]}
    costs = state.get("tool_costs", {})
    budget = state.get("budget", 0)
    buckets, seen = defaultdict(list), set()

    def add(name, **arguments):
        if name not in available or (name!="finish" and costs.get(name,999)>budget):
            return
        call = {"name":name,"arguments":arguments}
        validate_action({"type":name,**arguments})
        key = json.dumps(call,sort_keys=True)
        if key not in seen:
            buckets[name].append(call)
            seen.add(key)

    # Finish choices explicitly name their cited regions. The model chooses the
    # verdict and citation preset; no claim label or sufficiency rule is consulted.
    add("finish",verdict="Need more evidence",citations=[])
    originals = {}
    for view in views.values():
        originals.setdefault(view["image_id"], view)
    citations = [{"image_id":v["image_id"],"bbox":v["source_bbox"],"time":v["time"]}
                 for v in originals.values()]
    cite_sets = [[c] for c in citations]
    if len(citations)>1:
        cite_sets.append(citations)
    for verdict in ("Supported","Refuted"):
        for subset in cite_sets:
            add("finish",verdict=verdict,citations=subset)
        if len(citations)>1:
            for relation in ("same_object","different_object","earlier_than","same_time"):
                add("finish",verdict=verdict,citations=citations,
                    links=[{"relation":relation,"left":citations[0],"right":citations[1]}])

    options = state.get("request_options", {})
    for obj,time,view in product(options.get("objects",[]),options.get("times",[]),options.get("views",[])):
        add("request_photo",query={"object":obj,"time":time,"view":view})
    for view_id, view in views.items():
        for name in ("inspect","ocr","assess_quality","read_metadata"):
            add(name,image_id=view_id)
        for box in regions(view):
            add("crop",image_id=view_id,bbox=box)
        for factor in (.5,2):
            size = [max(1,round(x*factor)) for x in view["display_size"]]
            if math.prod(size)<=state.get("view_limits",{}).get("max_pixels",0):
                add("zoom",image_id=view_id,factor=factor)
        text = str(view.get("object") or state.get("claim",""))[:500].strip()
        if text:
            add("ground_text_to_image",image_id=view_id,text=text)
        add("ground_image_to_text",image_id=view_id,
            candidates=["An intact object.","A damaged object.","An obscured object."])
    for left,right in combinations(views,2):
        add("compare",image_ids=[left,right])
        add("ground_image_to_image",query_image_id=left,target_image_id=right,top_k=3)
        add("ground_image_to_image",query_image_id=right,target_image_id=left,top_k=3)

    # Round-robin tools to retain each action family under the finite menu cap.
    # Within a tool, interleave views so a new observation remains actionable.
    for name in list(buckets):
        if name=="finish":
            continue
        by_view=defaultdict(list)
        for call in buckets[name]:
            args=call["arguments"]
            key=args.get("image_id",args.get("target_image_id", "shared"))
            by_view[key].append(call)
        buckets[name]=[]
        while any(by_view.values()):
            for calls in by_view.values():
                if calls:
                    buckets[name].append(calls.pop(0))
    result=buckets.pop("finish",[])
    if len(result)>limit:
        raise ValueError("Menu cap cannot hold the explicit finish choices")
    while len(result)<limit and any(buckets.values()):
        for calls in buckets.values():
            if calls and len(result)<limit:
                result.append(calls.pop(0))
    if len(result)<2:
        raise ValueError("At least two public action choices are required")
    random.Random(17+len(public["messages"])).shuffle(result)
    return result


def action_question(candidates):
    return {"name":"action","type":"choice",
            "instructions":("Select one next action for the visual claim. Each option is a complete executable "
                            "tool call. Use its stated parameters and costs. Examine tool results before choosing "
                            "again. Choose finish with the appropriate verdict and cited observed regions when "
                            "ready. A normal reference may show another physical item. Similarity is not identity."),
            "choices":[{"value":f"a{i:03d}",
                        "description":json.dumps(call,separators=(",",":"))}
                       for i,call in enumerate(candidates)]}


def selected_action(answer, candidates):
    if answer.get("type")!="choice" or answer.get("name")!="action":
        return None
    value=answer.get("choice")
    mapping={f"a{i:03d}":call for i,call in enumerate(candidates)}
    return deepcopy(mapping.get(value)) if isinstance(value,str) else None
