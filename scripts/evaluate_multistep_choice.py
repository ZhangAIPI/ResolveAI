"""Finite-choice diagnostic on the procedural two-request suite.

This interface supplies RGB-derived candidate boxes and automatic identity links.
It measures assisted evidence-chain selection, not native tool use or predicted
region grounding. Its results must remain separate from the native benchmark.
"""
import argparse
import base64
from copy import deepcopy
import fcntl
import io
import json
import os
from pathlib import Path
import sqlite3
import string
import sys
import time

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_multistep as native
from resolveai.conversation import content_for
from resolveai.environment import Environment
from resolveai.pilot import ModelClient

INTERFACE = "interface-assisted-finite-choice-diagnostic"
EXPERIMENT = "procedural-multistep-interface-assisted-choice-v1"
CANDIDATE_GENERATION = "public-rgb-saturation-connected-components-v1"
CHOICE_SYSTEM = (
    "Verify the visual claim using only the released photographs and public "
    "request feedback. Select exactly one displayed option. Candidate rectangles "
    "are supplied by an image-processing helper; they do not establish identity "
    "or condition. Need more evidence is appropriate when the visible photographs "
    "cannot justify a definite conclusion. Reply with the option letter."
)
ACQUISITION_PROMPT = (
    "Choose the photograph to request next. There are two request slots in total, "
    "including repeats. The five options name public inspection views. Use the "
    "visible evidence to decide which next photograph is useful."
)
TERMINAL_PROMPT = (
    "No requests remain. Choose one complete answer and evidence chain below. "
    "Each definite option states its conclusion and the supplied source-pixel "
    "regions and identity links it will cite. Select the chain only if the "
    "photographs justify both the identity and condition. These boxes and links "
    "are interface assistance, not your own region predictions."
)


def rgb_boxes(image):
    """Detect brown connected components from one public image, never case data."""
    import cv2
    rgb = np.asarray(Image.open(io.BytesIO(image["image_png"])).convert("RGB"))
    value = rgb.max(axis=2).astype(np.float32)
    minimum = rgb.min(axis=2).astype(np.float32)
    saturation = (value - minimum) / np.maximum(value, 1)
    red, green, blue = (rgb[:, :, k].astype(np.float32) for k in range(3))
    mask = ((saturation >= .22) & (red > 1.04 * green)
            & (green > 1.04 * blue) & (value >= 20)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = rgb.shape[:2]
    source = image["source_bbox"]
    scale_x, scale_y = (source[2] - source[0]) / width, (source[3] - source[1]) / height
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if area < .015 * width * height or area > .95 * width * height:
            continue
        boxes.append([
            int(np.floor(source[0] + x * scale_x)),
            int(np.floor(source[1] + y * scale_y)),
            int(np.ceil(source[0] + (x + w) * scale_x)),
            int(np.ceil(source[1] + (y + h) * scale_y)),
        ])
    return sorted(boxes, key=lambda box: (box[0], box[1]))


def endpoint(image, bbox):
    return {"image_id": image["image_id"], "bbox": list(bbox), "time": image["time"]}


def terminal_candidates(observation):
    """Construct every candidate from the current public RGB and metadata only."""
    images = {image["image_id"]: image for image in observation["images"]}
    candidates = [{
        "verdict": "Need more evidence", "front_slot": None,
        "citations": [], "links": [],
    }]
    reference = images.get("reference")
    reference_boxes = rgb_boxes(reference) if reference else []
    reference_region = (endpoint(reference, reference_boxes[0])
                        if len(reference_boxes) == 1 else None)
    front = images.get("front_survey")
    front_boxes = rgb_boxes(front) if front else []
    usable_front = len(front_boxes) == 3
    slots = ("left", "center", "right")
    for slot in slots:
        rear = images.get("rear_" + slot)
        if rear is None:
            continue
        boxes = rgb_boxes(rear)
        if len(boxes) != 1:
            continue
        rear_region = endpoint(rear, boxes[0])
        citations = [rear_region]
        links = []
        if reference_region is not None:
            citations.insert(0, reference_region)
        if reference_region is not None and usable_front:
            front_region = endpoint(front, front_boxes[slots.index(slot)])
            citations.insert(1, front_region)
            links = [
                {"relation": "same_object", "left": deepcopy(reference_region),
                 "right": deepcopy(front_region)},
                {"relation": "same_object", "left": deepcopy(front_region),
                 "right": deepcopy(rear_region)},
            ]
        for verdict in ("Supported", "Refuted"):
            candidates.append({
                "verdict": verdict, "front_slot": slot if links else None,
                "citations": deepcopy(citations), "links": deepcopy(links),
            })
    if len(candidates) == 1:
        # No rear is available. Keep unsupported definite choices possible rather
        # than force restoration-only policies to abstain through the interface.
        citations = [] if reference_region is None else [reference_region]
        for verdict in ("Supported", "Refuted"):
            candidates.append({
                "verdict": verdict, "front_slot": None,
                "citations": deepcopy(citations), "links": [],
            })
    return {letter: item for letter, item in zip(string.ascii_uppercase, candidates)}


def public_payload(observation, feedback, options, phase):
    public = deepcopy(observation)
    public["request_feedback"] = deepcopy(feedback)
    public["choice_phase"] = phase
    public["options"] = deepcopy(options)
    instruction = ACQUISITION_PROMPT if phase == "acquisition" else TERMINAL_PROMPT
    return {
        "mode": "choice", "assistant_prefix": False,
        "messages": [
            {"role": "system", "content": CHOICE_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": instruction}, *content_for(public)]},
        ],
        "choices": list(options), "max_new_tokens": 1,
    }


def ask_choice(client, payload, phase, args, generations, audit_messages):
    audit_messages.extend(deepcopy(payload["messages"]))
    try:
        answer = client.ask_conversation(payload, args.max_context_tokens)
    except (RuntimeError, OSError, EOFError, json.JSONDecodeError) as error:
        generations.append({"phase": phase, "transport_error": type(error).__name__})
        raise
    generations.append({"phase": phase, "input_sha256": native.sha_value(payload),
                        **deepcopy(answer)})
    audit_messages.append({"role": "assistant", "content": answer.get("raw", "")})
    choice = answer.get("choice")
    if answer.get("halt") or choice not in payload["choices"]:
        return None
    return choice


def run_episode(case, data_directory, client, policy, args, observation_token, evaluate):
    started = time.perf_counter()
    env = Environment(case, case.get("asset_root", data_directory), args.budget,
                      ocr_backend=native.NoOCR())
    fingerprint = env._world.fingerprint
    initial = env.observation()
    queries = native.public_queries(initial)
    request_options = {
        letter: {"request_photo": deepcopy(queries[view])}
        for letter, view in zip(string.ascii_uppercase, native.REQUEST_VIEWS)
    }
    initial_sha = native.sha_value(public_payload(initial, [], request_options, "acquisition"))
    tokens = [observation_token(image, view=image["image_id"]) for image in initial["images"]]
    seen_rgb = {token["rgb_sha256"] for token in tokens}
    new_image_requests, new_pixel_images = 0, 0
    generations, acquisition_messages, terminal_messages = [], [], []
    executed, feedback = [], []
    selection_error = runtime_error = None
    terminal_sha = None
    selected = None
    choice_valid = False
    candidates = {}
    schedule = (None if policy == "adaptive_acquisition"
                else native.scheduled_views(case, policy, args.seed))
    try:
        for index in range(2):
            if policy == "adaptive_acquisition":
                payload = public_payload(env.observation(), feedback, request_options, "acquisition")
                letter = ask_choice(client, payload, f"acquisition-{index + 1}", args,
                                    generations, acquisition_messages)
                if letter is None:
                    selection_error = "invalid_acquisition_choice"
                    break
                query = request_options[letter]["request_photo"]
            else:
                query = queries[schedule[index]]
            call = {"name": "request_photo", "arguments": {"query": deepcopy(query)}}
            result = env.step({"type": "request_photo", "query": deepcopy(query)})
            returned = [observation_token(image, view=image["image_id"])
                        for image in result.get("images", [])]
            new_rgb = {token["rgb_sha256"] for token in returned} - seen_rgb
            new_image_requests += bool(new_rgb)
            new_pixel_images += len(new_rgb)
            seen_rgb.update(token["rgb_sha256"] for token in returned)
            tokens.extend(returned)
            response = {k: deepcopy(v) for k, v in result.items() if k != "images"}
            ids = [image["image_id"] for image in result.get("images", [])]
            executed.append({"call": call, "status": result.get("status"),
                             "error": result.get("error"), "feedback": response, "image_ids": ids})
            feedback.append({"query": deepcopy(query), "response": response,
                             "returned_image_ids": ids})
            if result.get("error"):
                selection_error = result["error"]
                break
        candidates = terminal_candidates(env.observation())
        payload = public_payload(env.observation(), feedback, candidates, "terminal")
        terminal_sha = native.sha_value(payload)
        letter = ask_choice(client, payload, "terminal", args, generations, terminal_messages)
        choice_valid = letter is not None
        if choice_valid:
            selected = candidates[letter]
            env.step({
                "type": "finish", "verdict": selected["verdict"],
                "citations": deepcopy(selected["citations"]), "links": deepcopy(selected["links"]),
            })
    except (RuntimeError, OSError, EOFError, json.JSONDecodeError) as error:
        runtime_error = type(error).__name__
    if env._world.fingerprint != fingerprint:
        raise ValueError("An episode mutated its hidden world")
    score = evaluate(case, env.decision if env.finished else None, tokens)
    elapsed = time.perf_counter() - started
    termination = ("model_runtime_error" if runtime_error else
                   "finished" if env.finished else "invalid_terminal_choice")
    assessable = bool(selected and selected["front_slot"] is not None)
    # Private target is used only for final diagnostic evaluation, not candidates.
    slot_correct = bool(assessable and
        "rear_" + selected["front_slot"] == case["multistep_certificate"]["target_rear_view"])
    turns = len(executed) + bool(selection_error) + any(g["phase"] == "terminal" for g in generations)
    trace = {
        "schema_version": "resolveai-multistep-choice-v1",
        "interface_scope": INTERFACE, "native_grounding_claim": False,
        "messages": terminal_messages, "acquisition_messages": acquisition_messages,
        "tools": [], "metadata": {
            "case_id": case["case_id"], "family_id": case["family_id"],
            "world_fingerprint": fingerprint, "annotation_status": "procedural",
            "termination": termination, "turns": turns,
            "errors": int(bool(selection_error)) + int(not choice_valid),
            "policy": policy, "privileged_policy": policy == "oracle",
        },
        "decision": deepcopy(env.decision), "observation_tokens": tokens,
        "evaluation": env.evaluate(env.decision) if env.finished else None,
        "native_evaluation": env.evaluate(env.decision) if env.finished else None,
        "multistep_evaluation": deepcopy(score), "generation": generations,
        "wall_latency_s": elapsed, "choice_candidates": candidates,
        "candidate_generation": CANDIDATE_GENERATION, "automatic_identity_links": True,
        "acquisition": {
            "executed_requests": executed, "selection_error": selection_error,
            "selection_complete": len(executed) == 2 and selection_error is None,
            "scheduled_prefix": policy != "adaptive_acquisition",
            "public_views": list(native.REQUEST_VIEWS),
        },
    }
    row = {
        "case_id": case["case_id"], "family_id": case["family_id"], "policy": policy,
        "interface_scope": INTERFACE, "privileged_policy": policy == "oracle",
        "termination": termination, "finished": bool(env.finished),
        "selection_complete": trace["acquisition"]["selection_complete"],
        "selection_error": selection_error, "runtime_error": runtime_error,
        "verdict": env.decision["verdict"] if env.finished else None,
        "multistep_evaluation": score, "requests": env.requests, "tool_calls": env.calls,
        "tool_cost": env.tool_cost, "request_cost": env.request_cost,
        "new_image_requests": new_image_requests, "new_pixel_images": new_pixel_images,
        "choice_valid": choice_valid, "selected_front_slot": selected["front_slot"] if selected else None,
        "front_slot_assessable": assessable, "selected_chain_target_correct": slot_correct,
        "token_accounting_complete": all("transport_error" not in g for g in generations),
        "assistant_turns": turns, "model_calls": len(generations),
        "input_tokens": sum(g.get("input_tokens", 0) for g in generations),
        "output_tokens": sum(g.get("output_tokens", 0) for g in generations),
        "wall_latency_s": elapsed, "initial_public_sha256": initial_sha,
        "terminal_public_sha256": terminal_sha,
    }
    return row, trace


def summarize(db, cases, policies):
    rows = [json.loads(row[0]) for row in db.execute("SELECT result FROM episodes")]
    complete = len(rows) == len(cases) * len(policies)
    groups = {}
    for policy in policies:
        entries = [row for row in rows if row["policy"] == policy]
        counts = {
            "attempted": len(entries),
            "source_label_correct": sum(bool(r["multistep_evaluation"]["source_label_correct"]) for r in entries),
            "evidence_sufficient": sum(bool(r["multistep_evaluation"]["procedural_evidence_sufficient"]) for r in entries),
            "valid_choice": sum(r["choice_valid"] for r in entries),
            "finished": sum(r["finished"] for r in entries),
            "selection_complete": sum(r["selection_complete"] for r in entries),
            "front_slot_assessable": sum(r["front_slot_assessable"] for r in entries),
            "selected_chain_target_correct": sum(r["selected_chain_target_correct"] for r in entries),
        }
        group = {"privileged_reference": policy == "oracle", "counts": counts}
        if complete:
            n = len(entries)
            group["rates_pct"] = {key: 100 * value / n for key, value in counts.items() if key != "attempted"}
            group["mean"] = {key: sum(r[key] for r in entries) / n for key in (
                "requests", "new_image_requests", "new_pixel_images", "tool_calls",
                "tool_cost", "request_cost", "model_calls", "input_tokens", "output_tokens", "wall_latency_s")}
            if not all(r["token_accounting_complete"] for r in entries):
                group["mean"]["input_tokens"] = group["mean"]["output_tokens"] = None
        groups[policy] = group
    return {
        "complete": complete, "episodes": len(rows),
        "expected_episodes": len(cases) * len(policies), "policies": groups,
        "interface_scope": INTERFACE, "metric_scope": "procedural-interface-assisted",
        "native_grounding_claim": False, "independent_human_review": False,
        "scope": "Finite-choice diagnostic; CV boxes and automatic links are interface assistance.",
        "grounding_scores": "Scorer chain/strict values remain raw assisted-selection diagnostics, not native grounding toplines.",
        "denominator": "All frozen episodes, including failed choices and transport failures.",
        "partial_rates_withheld": not complete,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--policies", nargs="+", choices=native.POLICIES, default=list(native.POLICIES))
    parser.add_argument("--variants", nargs="+", choices=("Obtainable",), default=["Obtainable"])
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run inside an existing GPU-node allocation")
    if (args.budget < 6 or args.max_context_tokens < 1 or args.shards < 1
            or not 0 <= args.shard < args.shards or (args.limit is not None and args.limit < 1)
            or len(set(args.policies)) != len(args.policies)):
        parser.error("Invalid limits or duplicate policies")
    from resolveai.multistep import observation_token, evaluate
    source, all_cases = native.load_data(args.data)
    eligible = [case for case in all_cases if case.get("variant", "Obtainable") in args.variants]
    families = sorted({case["family_id"] for case in eligible})[args.shard::args.shards]
    selected = sorted((c for c in eligible if c["family_id"] in families),
                      key=lambda c: (c["family_id"], c["case_id"]))
    if not selected:
        parser.error("This shard has no cases")
    indexed = {c["case_id"]: c for c in selected}
    expected = {(c["case_id"], p) for c in selected for p in args.policies}
    import cv2
    config = {
        "experiment": EXPERIMENT, "interface_scope": INTERFACE,
        "native_grounding_claim": False, "automatic_identity_links": True,
        "candidate_generation": CANDIDATE_GENERATION, "opencv_version": cv2.__version__,
        "cv_rule": {"saturation_min": .22, "red_green_ratio_min": 1.04,
                    "green_blue_ratio_min": 1.04, "value_min": 20,
                    "component_area_fraction": [.015, .95], "connectivity": 8},
        "source": str(source), "dataset": str(source), "dataset_sha256": native.sha_file(source),
        "assets": native.asset_fingerprint(all_cases, source.parent),
        "model": native.model_fingerprint(args.model, args.adapter),
        "model_id": args.model_id, "revision": args.revision,
        "source_sha256": native.source_digest(), "script_sha256": native.sha_file(__file__),
        "shared_native_script_sha256": native.sha_file(native.__file__),
        "families": families, "case_ids": [c["case_id"] for c in selected],
        "policies": args.policies, "variants": args.variants, "seed": args.seed,
        "shards": args.shards, "shard": args.shard, "budget": args.budget,
        "request_slots": 2, "request_views": list(native.REQUEST_VIEWS),
        "random_seed_group": "paired_history_id", "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": 1, "do_sample": False, "decoder": "restricted-next-token-letter-argmax",
        "system_instruction": CHOICE_SYSTEM, "acquisition_instruction": ACQUISITION_PROMPT,
        "terminal_instruction": TERMINAL_PROMPT,
        "terminal_calls": 1, "terminal_input": "fresh-public-observation-and-public-request-feedback",
        "scope": "Interface-assisted finite-choice diagnostic; separate from native results.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = native.locked_manifest(args.output / "manifest.json", config)
        with sqlite3.connect(args.output / "episodes.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
            db.commit()
            completed, initial_hashes = native.validate_saved(db, expected, indexed, args.output, evaluate)
            pending = [f for f in families if any((c["case_id"], p) not in completed
                       for c in selected if c["family_id"] == f for p in args.policies)]
            if args.limit is not None:
                pending = pending[:args.limit]
            for case in selected:
                if case["family_id"] not in pending:
                    continue
                env = Environment(case, case.get("asset_root", source.parent), args.budget,
                                  ocr_backend=native.NoOCR())
                observation = env.observation()
                native.public_queries(observation)
                initial_score = evaluate(case, None, [
                    observation_token(i, view=i["image_id"]) for i in observation["images"]])
                if (not initial_score.get("certificate_valid")
                        or initial_score.get("minimum_request_count") != 2
                        or initial_score.get("procedural_evidence_sufficient")):
                    raise ValueError("Case is not a certified initially unresolved two-request task")
                terminal_candidates(observation)
                for policy in args.policies:
                    if policy != "adaptive_acquisition":
                        native.scheduled_views(case, policy, args.seed)
            if not args.execute or not pending:
                result = summarize(db, selected, args.policies)
                result.update(preflight=True, model_loaded=False)
                (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"preflight": True, "model_loaded": False, "pending_families": len(pending)}))
                return
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            client = ModelClient(args.adapter or args.model, 0, "auto", assistant_prefix=False)
            manifest["worker"] = client.ready
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            store = native.TrajectoryStore(args.output / "pending-trace.jsonl")
            try:
                for case in selected:
                    if case["family_id"] not in pending:
                        continue
                    for policy in args.policies:
                        if (case["case_id"], policy) in completed:
                            continue
                        row, trace = run_episode(case, source.parent, client, policy, args,
                                                 observation_token, evaluate)
                        old = initial_hashes.setdefault(case["case_id"], row["initial_public_sha256"])
                        if old != row["initial_public_sha256"]:
                            raise ValueError("Initial public input differs across policies")
                        native.save_record(db, store, row, trace)
                        completed.add((case["case_id"], policy))
                        print(json.dumps({"completed": len(completed), "expected": len(expected),
                                          "case_id": case["case_id"], "policy": policy,
                                          "choice_valid": row["choice_valid"]}), flush=True)
                        if row["runtime_error"]:
                            raise RuntimeError("Model worker failed; attempted episode retained")
            finally:
                client.close()
                (args.output / "summary.json").write_text(
                    json.dumps(summarize(db, selected, args.policies), indent=2) + "\n")


if __name__ == "__main__":
    main()
