"""Renderer-specific public-RGB sanity policy, separate from MLLM comparisons.

The CV readout uses only released RGB and public image metadata. Private cases
are passed only to the Environment, baseline scheduler (oracle is privileged),
and final scorer. Tune on the 16-case pilot; held-out execution requires its
completed, matching code-fingerprint manifest. This is not a learned policy or
evidence of transfer to natural images.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_multistep as native
from scripts.evaluate_multistep_choice import rgb_boxes, endpoint
from resolveai.environment import Environment
from resolveai.multistep import evaluate, observation_token

SCOPE = "renderer-specific-public-RGB-program"
SLOTS = ("left", "center", "right")
PARAMETERS = {
    "texture_inner_fraction": .1, "texture_size": 128,
    "texture_gaussian_sigma": 3, "NCC_minimum_margin": .002,
    "scratch_brightness_above_median": 12, "scratch_component_pixels": 4,
    "scratch_bbox_padding": 2, "panel_color_distance": 5,
}


def pixels(image):
    return np.asarray(Image.open(io.BytesIO(image["image_png"])).convert("RGB"))


def local_box(image, box):
    height, width = pixels(image).shape[:2]
    source = image["source_bbox"]
    return [int(round((box[k] - source[k % 2]) *
                     (width if k % 2 == 0 else height) /
                     (source[2 + k % 2] - source[k % 2]))) for k in range(4)]


def source_box(image, box):
    height, width = pixels(image).shape[:2]
    source = image["source_bbox"]
    return [int(round(source[k % 2] + box[k] *
                     (source[2 + k % 2] - source[k % 2]) /
                     (width if k % 2 == 0 else height))) for k in range(4)]


def texture(image, box):
    x0, y0, x1, y1 = local_box(image, box)
    crop = pixels(image)[y0:y1, x0:x1]
    dy, dx = [max(1, round(v * PARAMETERS["texture_inner_fraction"]))
              for v in crop.shape[:2]]
    crop = crop[dy:-dy, dx:-dx]
    if min(crop.shape[:2], default=0) < 2:
        return None
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY),
                      (PARAMETERS["texture_size"],) * 2, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray.astype(float), (0, 0),
                            PARAMETERS["texture_gaussian_sigma"])
    deviation = gray.std()
    return None if deviation < 1e-6 else (gray - gray.mean()) / deviation


def match_front(observation):
    images = {im["image_id"]: im for im in observation["images"]}
    if not {"reference", "front_survey"} <= images.keys():
        return None
    reference, front = images["reference"], images["front_survey"]
    reference_boxes, boxes = rgb_boxes(reference), rgb_boxes(front)
    if len(reference_boxes) != 1 or len(boxes) != 3:
        return None
    patterns = [texture(reference, reference_boxes[0]),
                *[texture(front, box) for box in boxes]]
    if any(pattern is None for pattern in patterns):
        return None
    scores = [float((patterns[0] * pattern).mean()) for pattern in patterns[1:]]
    order = np.argsort(scores)
    margin = scores[int(order[-1])] - scores[int(order[-2])]
    if margin < PARAMETERS["NCC_minimum_margin"]:
        return None
    slot = int(order[-1])
    return {"slot": SLOTS[slot], "scores": scores, "margin": margin,
            "reference": endpoint(reference, reference_boxes[0]),
            "front": endpoint(front, boxes[slot])}


def diagnose(image):
    boxes = rgb_boxes(image)
    if len(boxes) != 1:
        return None
    x0, y0, x1, y1 = local_box(image, boxes[0])
    crop = pixels(image)[y0:y1, x0:x1]
    brightness = crop.max(axis=2).astype(float)
    bright = (brightness > np.median(brightness) +
              PARAMETERS["scratch_brightness_above_median"]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    components = [i for i in range(1, count)
                  if stats[i, 4] >= PARAMETERS["scratch_component_pixels"]]
    if components:
        ys, xs = np.where(np.isin(labels, components))
        pad = PARAMETERS["scratch_bbox_padding"]
        box = [x0 + max(0, int(xs.min()) - pad), y0 + max(0, int(ys.min()) - pad),
               x0 + min(crop.shape[1], int(xs.max()) + pad + 1),
               y0 + min(crop.shape[0], int(ys.max()) + pad + 1)]
        verdict = "Supported"
    else:
        color = np.median(crop.reshape(-1, 3), axis=0)
        flat = (np.max(np.abs(crop.astype(float) - color), axis=2) <
                PARAMETERS["panel_color_distance"]).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(flat)
        if count < 2:
            return None
        x, y, width, height, area = [int(v) for v in stats[1 + np.argmax(stats[1:, 4])]]
        if area < .5 * crop.shape[0] * crop.shape[1]:
            return None
        box = [x0 + x, y0 + y, x0 + x + width, y0 + y + height]
        verdict = "Refuted"
    return {"verdict": verdict, "object": endpoint(image, boxes[0]),
            "diagnostic": endpoint(image, source_box(image, box))}


def readout(observation):
    empty = {"verdict": "Need more evidence", "citations": [], "links": []}
    match = match_front(observation)
    images = {im["image_id"]: im for im in observation["images"]}
    rear = images.get("rear_" + match["slot"]) if match else None
    condition = diagnose(rear) if rear else None
    if condition is None:
        return empty, {"identity": match, "condition": None}
    reference, front, rear = match["reference"], match["front"], condition["object"]
    decision = {"verdict": condition["verdict"],
                "citations": [reference, front, rear, condition["diagnostic"]],
                "links": [{"relation": "same_object", "left": deepcopy(reference),
                           "right": deepcopy(front)},
                          {"relation": "same_object", "left": deepcopy(front),
                           "right": deepcopy(rear)}]}
    return decision, {"identity": match, "condition": condition}


def episode(case, policy, seed):
    started = time.perf_counter()
    env = Environment(case, case["asset_root"], 6, ocr_backend=native.NoOCR())
    initial = env.observation()
    queries = native.public_queries(initial)
    tokens = [observation_token(im, view=im["image_id"]) for im in initial["images"]]
    first_hash = native.sha_value(tokens)
    actions = []
    schedule = None if policy == "adaptive_acquisition" else native.scheduled_views(case, policy, seed)
    for index in range(2):
        if schedule is not None:
            view = schedule[index]
        elif index == 0:
            view = "front_survey"
        else:
            match = match_front(env.observation())
            view = "rear_" + match["slot"] if match else "original"
        result = env.step({"type": "request_photo", "query": deepcopy(queries[view])})
        tokens.extend(observation_token(im, view=im["image_id"])
                      for im in result.get("images", []))
        actions.append({"view": view, "status": result.get("status"),
                        "error": result.get("error"),
                        "image_ids": [im["image_id"] for im in result.get("images", [])]})
    decision, predictions = readout(env.observation())
    result = env.step({"type": "finish", **deepcopy(decision)})
    if result.get("error") or not env.finished:
        raise ValueError("Program emitted an invalid terminal decision")
    score = evaluate(case, env.decision, tokens)
    trace = {"decision": env.decision, "public_RGB_predictions": predictions,
             "requests": actions, "observation_tokens": tokens,
             "multistep_evaluation": score}
    row = {"case_id": case["case_id"], "family_id": case["family_id"],
           "policy": policy, "scope": SCOPE, "privileged_reference": policy == "oracle",
           "requests": env.requests, "tool_calls": env.calls, "finished": env.finished,
           "verdict": decision["verdict"], "initial_public_sha256": first_hash,
           "multistep_evaluation": score, "trace_sha256": native.sha_value(trace),
           "wall_latency_s": time.perf_counter() - started}
    return row, trace


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(),
             root / "scripts/evaluate_multistep_choice.py",
             root / "scripts/evaluate_multistep.py",
             root / "src/resolveai/multistep.py"]
    return {str(p.relative_to(root)): native.sha_file(p) for p in paths}


def summarize(rows, expected):
    complete = len(rows) == expected
    groups = {}
    for policy in native.POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        counts = {"attempted": len(selected)}
        for metric in ("source_label_correct", "procedural_evidence_sufficient",
                       "chain_grounded_correct", "localized_grounded_correct",
                       "simulator_grounded_correct", "unsupported_decision"):
            counts[metric] = sum(bool(row["multistep_evaluation"][metric]) for row in selected)
        counts["abstained"] = sum(row["verdict"] == "Need more evidence" for row in selected)
        groups[policy] = {"privileged_reference": policy == "oracle", "counts": counts}
        if complete:
            groups[policy]["rates_pct"] = {key: 100 * value / len(selected)
                                          for key, value in counts.items() if key != "attempted"}
    return {"complete": complete, "episodes": len(rows), "expected_episodes": expected,
            "scope": SCOPE, "learned_policy": False, "natural_image_generalization_claim": False,
            "same_CV_readout_all_policies": True, "request_slots": 2, "policies": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("pilot", "heldout"), required=True)
    parser.add_argument("--frozen-pilot", type=Path)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run on an allocated compute node")
    code = fingerprints()
    frozen = None
    if args.subset == "heldout":
        if not args.frozen_pilot:
            parser.error("Held-out data requires a completed pilot code freeze")
        frozen = json.loads(args.frozen_pilot.read_text())
        if not frozen.get("complete") or frozen["code_fingerprints"] != code:
            parser.error("Pilot is incomplete or policy/dependency code changed")
    source, cases = native.load_data(args.data)
    cases = [case for case in cases if case["variant"] == "Obtainable"]
    if frozen:
        pilot_ids = set(frozen["family_ids"])
        cases = [case for case in cases if case["family_id"] not in pilot_ids]
    if len(cases) != (16 if args.subset == "pilot" else 116):
        parser.error("Expected all 16 pilot or all 116 held-out families")
    config = {"experiment": SCOPE + "-v1", "subset": args.subset, "seed": args.seed,
              "code_fingerprints": code, "parameters": PARAMETERS,
              "dataset_sha256": native.sha_file(source), "request_slots": 2,
              "family_ids": sorted(case["family_id"] for case in cases),
              "policies": list(native.POLICIES), "same_CV_readout_all_policies": True,
              "model_calls": 0, "private_oracle_separate": True}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if {key: previous[key] for key in config} != config:
            raise ValueError("Refusing to change an existing frozen run")
    path.write_text(json.dumps({**config, "complete": False}, indent=2) + "\n")
    rows = []
    with sqlite3.connect(args.output / "episodes.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS episodes "
                   "(case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
        old = {(case_id, policy): json.loads(result)
               for case_id, policy, result in db.execute("SELECT case_id,policy,result FROM episodes")}
        expected_keys = {(case["case_id"], policy) for case in cases for policy in native.POLICIES}
        if not set(old) <= expected_keys:
            raise ValueError("Unexpected saved episode")
        for case in cases:
            for policy in native.POLICIES:
                key = case["case_id"], policy
                if key in old:
                    row = old[key]
                else:
                    row, trace = episode(case, policy, args.seed)
                    with db:
                        db.execute("INSERT INTO episodes VALUES (?,?,?,?)",
                                   (*key, json.dumps(row), json.dumps(trace)))
                rows.append(row)
    summary = summarize(rows, len(cases) * len(native.POLICIES))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    path.write_text(json.dumps({**config, "complete": summary["complete"]}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
