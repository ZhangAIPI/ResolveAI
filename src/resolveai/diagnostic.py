"""Paired capability diagnostics: perception, evidence acquisition and tool interface.

This is a development experiment, not a replacement for reviewed evidence
sufficiency annotations. Finite menus intentionally control argument generation.
"""
import argparse
import base64
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import string
import time

from PIL import Image

from .benchmark import locked_manifest, source_digest
from .conversation import Conversation, TrajectoryStore, content_for
from .environment import Environment
from .pilot import ModelClient
from .rollout import run

VERDICTS = ("Supported", "Refuted", "Need more evidence")
BASIC_TOOLS = {"finish", "inspect", "crop", "zoom", "compare", "request_photo"}
DIRECT_SYSTEM = (
    "Judge the stated visual claim from the supplied pixels. Supported means "
    "the claim is supported; Refuted means the claim is contradicted; "
    "Need more evidence means you cannot reliably judge. A matching-looking "
    "object is not necessarily the same physical item. Do not assume that "
    "items share identity because of their image IDs or background. "
    "Choose one displayed option. You may always choose Need more evidence."
)
MENU_SYSTEM = (
    "Verify the claim using released pixels. Choose one displayed action per "
    "turn. Each entry is a concrete tool call with its actual arguments. "
    "Requests may fail and do not change the object. Crop/zoom operate only on "
    "existing pixels. Select useful evidence rather than repeating actions. "
    "Finish when you can judge or useful material cannot be obtained. Similarity "
    "is not identity truth. Need more evidence is always available. "
    "The finite menu controls argument formatting, not which action you choose."
)


def select_cases(mvtec, co3d, per_label=2, per_category=10):
    """Freeze source-balanced development families before any model evaluation."""
    selected = []
    for source, quota, group in [
        (Path(mvtec), per_label, lambda c: (c["provenance"]["category"], c["annotation"]["visual_verdict"])),
        (Path(co3d), per_category, lambda c: (c["category"],)),
    ]:
        buckets = defaultdict(list)
        for case in json.loads((source / "cases.json").read_text()):
            if case["variant"] == "Obtainable":
                buckets[group(case)].append(case)
        for key, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda c: hashlib.sha256(("diagnostic-v1|" + c["family_id"]).encode()).hexdigest())
            if len(bucket) < quota:
                raise ValueError(f"insufficient candidates in {key}")
            for case in bucket[:quota]:
                case = deepcopy(case)
                case["diagnostic_source"] = "mvtec" if source == Path(mvtec) else "co3d"
                case["asset_root"] = str(source.resolve())
                case["diagnostic_truth"] = case["annotation"].get("visual_verdict", case["annotation"]["verdict"])
                selected.append(case)
    add_identity_controls(selected)
    if len({c["family_id"] for c in selected}) != len(selected):
        raise ValueError("development family IDs must be unique")
    return selected


def identity_edge(case):
    part = case["annotation"]["subclaims"][0]
    return part["minimal_evidence_sets"][part["truth"]][0][0]


def add_identity_controls(cases):
    """Add easy visual negatives without exposing category differences in the claim.

    These control pairs use different source categories, not alleged same-item
    damage transitions. Shared source sequences remain development-only; this
    experiment does not treat generated pairs as independent test families.
    """
    groups = defaultdict(list)
    for case in cases:
        if case["diagnostic_source"] == "co3d":
            groups[case["category"]].append(case)
    categories = sorted(groups)
    originals = deepcopy(groups)
    for n, category in enumerate(categories):
        bucket = groups[category]
        for index, case in enumerate(bucket):
            edge = deepcopy(identity_edge(case))
            case["diagnostic_source_sequences"] = [case["scene"]["source_sequence"]]
            if index >= len(bucket)//2:
                partner = originals[categories[(n+1)%len(categories)]][index]
                foreign_edge = identity_edge(partner)
                old_id, foreign_id = edge["right"]["image_id"], foreign_edge["right"]["image_id"]
                old_original = next(e for e in case["evidence"] if e["id"] == old_id)
                foreign_original = deepcopy(next(e for e in partner["evidence"] if e["id"] == foreign_id))
                foreign_preview = deepcopy(next(e for e in partner["evidence"] if e["id"] == foreign_id+"-preview"))
                foreign_original["view"] = old_original["view"]
                foreign_preview["camera_view"] = old_original["view"]
                case["evidence"] = [e for e in case["evidence"] if e["id"] not in {old_id,old_id+"-preview"}]
                case["evidence"].extend([foreign_original, foreign_preview])
                case["initial"] = [edge["left"]["image_id"], foreign_id+"-preview"]
                edge["right"] = deepcopy(foreign_edge["right"])
                edge["relation"] = "different_object"
                case["diagnostic_truth"] = "Refuted"
                case["annotation"] = {"protocol":"evidence-chain-v1", "status":"source-derived-unreviewed",
                    "verdict":"Refuted", "subclaims":[{"id":"identity", "kind":"identity", "truth":"Refuted",
                        "minimal_evidence_sets":{"Refuted":[[edge]]}}]}
                case["diagnostic_source_sequences"].append(partner["scene"]["source_sequence"])
                case["family_id"] += "-cross-category-control"
                case["case_id"] = case["family_id"]+"-Obtainable"
                case["diagnostic_identity_control"] = "easy-cross-category-negative"
            else:
                case["diagnostic_identity_control"] = "source-sequence-positive"
            case["claim"] = ("The objects in the two specified foreground regions are the same physical item. "
                             "Ignore other objects and the background.")
            case["claim_parts"] = [{"id":"identity", "text":case["claim"]}]
            case["diagnostic_public_targets"] = [
                {"initial_image_id": e["image_id"] if e["image_id"] in case["initial"] else e["image_id"]+"-preview",
                 "source_region":e["bbox"], "coordinate_space":"source_pixels"}
                for e in (edge["left"],edge["right"])]


def public_observation(env, case):
    observation = env.observation()
    if "diagnostic_public_targets" in case:
        observation["target_regions"] = case["diagnostic_public_targets"]
    return observation


def menu_actions(env):
    """Concrete choices use public metadata only; no annotation or hidden pixels."""
    obs = env.observation()
    images = obs["images"]
    citations = [{"image_id": r["image_id"], "bbox": r["source_bbox"], "time": r["time"]}
                 for r in images]
    actions = [{"name": "finish", "arguments": {"verdict": v, "citations": citations, "links": []}}
               for v in VERDICTS]
    options = obs.get("request_options", {})
    for obj, stamp, view in itertools.product(options.get("objects", []), options.get("times", []),
                                             options.get("views", [])):
        actions.append({"name": "request_photo", "arguments": {
            "query": {"object": obj, "time": stamp, "view": view}}})
    for row in images[-3:]:
        ref = row["view_id"]
        width, height = row["display_size"]
        actions.append({"name": "inspect", "arguments": {"image_id": ref}})
        if width >= 4 and height >= 4:
            actions.append({"name": "crop", "arguments": {"image_id": ref,
                "bbox": [width//8, height//8, width-width//8, height-height//8]}})
        if width * height * 4 <= env.MAX_VIEW_PIXELS:
            actions.append({"name": "zoom", "arguments": {"image_id": ref, "factor": 2}})
    if len(images) >= 2:
        actions.append({"name": "compare", "arguments": {
            "image_ids": [images[0]["view_id"], images[1]["view_id"]]}})
    actions = [a for a in actions if a["name"] == "finish" or getattr(env.costs, a["name"]) <= env.budget]
    if len(actions) > 26:
        raise ValueError("menu exceeds single-letter choice capacity")
    return dict(zip(string.ascii_uppercase, actions))


def choose(client, messages, choices):
    return client.ask_conversation({"mode": "choice", "messages": messages,
                                   "choices": list(choices)}, max_context_tokens=8192)


def direct(env, case, client):
    started = time.perf_counter()
    observation = public_observation(env, case)
    options = dict(zip("ABC", VERDICTS))
    messages = [{"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": content_for({**observation, "options": options})}]
    response = choose(client, messages, options)
    verdict = options[response["choice"]] if not response.get("halt") else None
    messages.append({"role": "assistant", "content": response.get("raw", "")})
    return {"messages": messages, "decision": {"verdict": verdict} if verdict else None,
            "termination": response.get("halt", "finished"), "errors": 0,
            "requests": 0, "tool_calls": 0, "generation": [response],
            "wall_latency_s": time.perf_counter() - started, "request_events": []}


def menu_rollout(env, case, client, max_turns):
    started = time.perf_counter()
    session = Conversation(env)
    session.messages[0]["content"] = MENU_SYSTEM + f" Finish within {max_turns} turns."
    session.tools = [t for t in session.tools if t["function"]["name"] in BASIC_TOOLS]
    session.messages[1]["content"] = content_for(public_observation(env, case))
    generations, events = [], []
    for _ in range(max_turns):
        options = menu_actions(env)
        request_messages = deepcopy(session.messages)
        # Portable transport preserves every actual call, response and image,
        # without introducing a schema/prefix for the finite-choice response.
        from .vlm_worker import portable_history
        request_messages = portable_history(request_messages)
        request_messages.append({"role": "user", "content": json.dumps({
            "remaining_budget": env.budget, "options": options,
            "instruction": "Choose exactly one option letter. Do not answer in prose."})})
        response = choose(client, request_messages, options)
        generations.append(response)
        if response.get("halt"):
            session.termination = response["halt"]
            break
        call = options[response["choice"]]
        result = session.call(call)
        if call["name"] == "request_photo":
            events.append({"query": call["arguments"]["query"], "status": result.get("status"),
                           "image_ids": [r["image_id"] for r in result.get("images", [])],
                           "error": result.get("error")})
        if env.finished:
            break
    session.termination = session.termination or "max_turns"
    record = session.record()
    # Menu finish deliberately omits free localization/chain prediction. Its
    # score below is verdict/access, never the environment's chain metric.
    record.pop("evaluation", None)
    record.update(generation=generations, requests=env.requests, tool_calls=env.calls,
                  errors=session.errors, termination=session.termination,
                  wall_latency_s=time.perf_counter()-started, request_events=events)
    return record


def original_id(case):
    if case["diagnostic_source"] == "mvtec":
        return "original"
    return identity_edge(case)["right"]["image_id"]


def request_query(case):
    return next({"object": e["object"], "time": e["time"], "view": e["view"]}
                for e in case["evidence"] if e["id"] == original_id(case))


def rich_rollout(env, case, client, max_turns):
    session = Conversation(env)
    session.tools = [t for t in session.tools if t["function"]["name"] in BASIC_TOOLS]
    session.messages[1]["content"] = content_for(public_observation(env, case))
    record = run(session, client, max_turns)
    events = []
    for message in session.messages:
        if message["role"] != "tool" or message["name"] != "request_photo":
            continue
        result = json.loads(message["content"][0]["text"])
        events.append({"status": result.get("status"), "error": result.get("error"),
                       "image_ids": [r["image_id"] for r in result.get("images", [])]})
    record.pop("evaluation", None)
    record.update(requests=env.requests, tool_calls=env.calls, errors=session.errors,
                  termination=session.termination, request_events=events)
    return record


def outcome(case, arm, record, released):
    verdict = record["decision"]["verdict"] if record.get("decision") else None
    return {"family_id": case["family_id"], "source": case["diagnostic_source"],
        "category": case.get("category", case.get("provenance", {}).get("category")),
        "label": case["diagnostic_truth"], "arm": arm, "verdict": verdict,
        "label_correct": verdict == case["diagnostic_truth"],
        "finished": record["termination"] == "finished", "termination": record["termination"],
        "errors": record["errors"], "requests": record["requests"],
        "tool_calls": record["tool_calls"], "received_target_original": original_id(case) in released,
        "request_events": record["request_events"], "wall_latency_s": record["wall_latency_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        parser.error("invalid shards")
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"][args.shard_index::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "source_sha256": source_digest(), "model_id": args.model_id, "revision": args.revision,
        "shards": args.shards, "shard_index": args.shard_index, "max_turns": 6, "budget": 12,
        "arms": ["initial", "scripted_request", "menu_agent", "rich_agent",
                 "menu_unavailable", "rich_unavailable"],
        "choice_decoding": "greedy-next-token-over-displayed-single-letter-options",
        "rich_decoding": "portable-prefix-v1; greedy; max_new_tokens=384"}
    manifest = locked_manifest(args.output/"manifest.json", config)
    result_path = args.output/"results.jsonl"
    completed = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            key = row["family_id"], row["arm"]
            if key in completed:
                raise ValueError("duplicate diagnostic result")
            completed[key] = row
    expected = {(c["family_id"], a) for c in cases for a in config["arms"]}
    if not completed.keys() <= expected:
        raise ValueError("unexpected diagnostic result")
    if completed.keys() == expected:
        print(json.dumps({"complete": True, "episodes": len(completed)}), flush=True)
        return
    client = ModelClient(args.model, args.device, "portable")
    manifest["worker"] = client.ready
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    store = TrajectoryStore(args.output/"trajectories.jsonl")
    try:
        with result_path.open("a") as output:
            for case in cases:
                for arm in config["arms"]:
                    if (case["family_id"], arm) in completed:
                        continue
                    episode = deepcopy(case)
                    if arm.endswith("unavailable"):
                        for evidence in episode["evidence"]:
                            if evidence["id"] == original_id(case):
                                evidence["available"] = False
                    env = Environment(episode, case["asset_root"], 12)
                    if arm == "scripted_request":
                        result = env.step({"type": "request_photo", "query": request_query(case)})
                        if result["status"] != "provided" or original_id(case) not in env._released:
                            raise RuntimeError("scripted retrieval failed; benchmark not ready")
                        record = direct(env, case, client)
                        record["requests"] = env.requests
                        record["tool_calls"] = env.calls
                        record["request_events"] = [{"status": result["status"],
                            "image_ids": [r["image_id"] for r in result["images"]]}]
                    elif arm == "initial":
                        record = direct(env, case, client)
                    elif arm.startswith("menu"):
                        record = menu_rollout(env, case, client, 6)
                    else:
                        record = rich_rollout(env, case, client, 6)
                    record.setdefault("tools", [])
                    record.update(metadata={"family_id": case["family_id"], "arm": arm,
                        "annotation_status": "source-derived-unreviewed-development-only"},
                        evaluation=None, schema_version="resolveai-diagnostic-v1")
                    store.write(record)
                    row = outcome(case, arm, record, env._released)
                    output.write(json.dumps(row)+"\n")
                    output.flush()
                    completed[(case["family_id"], arm)] = row
                    print(json.dumps({"completed": len(completed), "expected": len(expected), **row}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
