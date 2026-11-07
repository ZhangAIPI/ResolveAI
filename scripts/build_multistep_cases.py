"""Freeze paired multi-step simulator cases from audited Blender view banks.

This is a separate procedural track. Finite-world identifiability and
renderer-derived regions are not independent human judgments on photographs.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import random

from PIL import Image

from resolveai.environment import Environment
from resolveai.multistep import bank_digest, certify_bank, observation_token

SLOTS = ("left", "center", "right")
VIEWS = ("original", "front_survey", "rear_left", "rear_center", "rear_right")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def physical_label(world):
    position = world["permutation"].index(0)
    return "Supported" if world["back_states"][position] else "Refuted"


def validate_render_bank(bank_path, raw):
    """Validate saved first-hit masks and actual counterfactual re-renders."""
    images = raw["images"]
    expected = {"reference", "initial", *("front_permutation_" + str(i) for i in range(6)),
                *("rear_" + slot + "_" + state for slot in SLOTS for state in ("intact", "scratched"))}
    if set(images) != expected:
        raise ValueError("Incomplete rendered view bank")
    for key, row in images.items():
        path = bank_path.parent / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Rendered asset file checksum changed")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if list(rgb.size) != row["size"]:
                raise ValueError("Rendered asset dimensions changed")
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
                raise ValueError("Rendered RGB checksum disagrees with independent PNG decoding")
        check = row["visibility_checks"]
        mask_path = bank_path.parent / check["mask_path"]
        if hashlib.sha256(mask_path.read_bytes()).hexdigest() != check["mask_sha256"]:
            raise ValueError("Visibility mask checksum changed")
        with Image.open(mask_path) as image:
            mask = image.convert("RGB")
            if list(mask.size) != row["size"]:
                raise ValueError("Visibility mask dimensions disagree")
            counts = {name: sum(value > 127 for value in mask.getchannel(channel).tobytes())
                      for channel, name in enumerate(("front", "rear", "scratch"))}
        if counts != check["pixel_counts"]:
            raise ValueError("Stored visibility counts disagree with the saved first-hit mask")
        if key == "initial":
            good = not any(counts.values())
        elif key == "reference" or key.startswith("front_permutation_"):
            good = counts["front"] > 0 and counts["rear"] == counts["scratch"] == 0
        else:
            good = counts["front"] == 0 and counts["rear"] > 0
            good = good and (counts["scratch"] > 0 if key.endswith("scratched") else counts["scratch"] == 0)
        if not good:
            raise ValueError("A supposedly hidden surface is visible or a diagnostic surface is missing")
    checks = raw["counterfactual_rgb_checks"]
    if len(checks) != 3 or {c["base_image"] for c in checks} != {"initial", "front_permutation_0", "rear_left_intact"}:
        raise ValueError("Expected three independent hidden-state re-render checks")
    for check in checks:
        if (not check.get("equal")
                or check["base_rgb_sha256"] != check["alternative_rgb_sha256"]
                or check["base_rgb_sha256"] != images[check["base_image"]]["decoded_rgb_sha256"]):
            raise ValueError("A hidden change alters public RGB pixels")


def canonicalize(bank_path, raw):
    """Strip PNG metadata before freezing file hashes and public observations."""
    for row in raw["images"].values():
        path = bank_path.parent / row["path"]
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.save(path, format="PNG")
        row.setdefault("renderer_file_sha256", old_hash)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["size"] = list(image.size)
    raw["png_canonicalization"] = "RGB PNG without EXIF or text metadata"
    bank_path.write_text(json.dumps(raw, indent=2) + "\n")


def selected_rows(raw, permutation, states):
    images = raw["images"]
    front = next(row for key, row in images.items()
                 if key.startswith("front_permutation_") and row["permutation"] == list(permutation))
    return {
        "reference": images["reference"], "initial": images["initial"],
        "original": images["initial"], "front_survey": front,
        **{"rear_" + slot: images["rear_" + slot + "_" + ("scratched" if bit else "intact")]
           for slot, bit in zip(SLOTS, states)},
    }


def evidence_rows(rows, bank_path, asset_root):
    evidence = []
    for view, row in rows.items():
        width, height = row["size"]
        reference = view == "reference"
        evidence.append({
            "id": view, "path": str((bank_path.parent / row["path"]).relative_to(asset_root)),
            "source_id": "reference-front" if reference else "inspection-" + ("overview" if view in {"initial", "original"} else view),
            "party": "A" if reference else "B",
            "object": "reference crate" if reference else "inspection scene",
            "time": "reference" if reference else "inspection",
            "view": "reference_front" if reference else "overview" if view == "initial" else view,
            "camera_view": "overview" if view in {"initial", "original"} else view,
            "source_size": [width, height], "source_bbox": [0, 0, width, height],
            "available": True,
        })
    return evidence


def view_bank(raw, bank_path, asset_root):
    cache, worlds = {}, []
    for permutation in permutations(range(3)):
        for states in product((0, 1), repeat=3):
            rows = selected_rows(raw, permutation, states)
            evidence = evidence_rows(rows, bank_path, asset_root)
            shell = {"case_id": "private-render-audit", "family_id": "private-render-audit",
                     "claim": "", "initial": list(rows), "evidence": evidence,
                     "annotation": {"verdict": "Need more evidence", "minimal_evidence_sets": {}}}
            env = Environment(shell, asset_root, budget=6)
            observations = {}
            for row in evidence:
                key = row["id"], row["path"]
                if key not in cache:
                    cache[key] = observation_token(env._image(row["id"]), view=row["id"])
                observations[row["id"]] = cache[key]
            worlds.append({
                "world_id": "p" + "".join(map(str, permutation)) + "-s" + "".join(map(str, states)),
                "permutation": list(permutation), "back_states": list(states),
                "observations": observations,
            })
    return {"version": "factorized-render-bank-v1", "worlds": worlds}


def history_key(world, views):
    return tuple((view, digest(world["observations"][view])) for view in views)


def compatible_labels(bank, world, views):
    key = history_key(world, views)
    return {physical_label(candidate) for candidate in bank["worlds"]
            if history_key(candidate, views) == key}


def audit_bank(bank):
    """Independent exhaustive check of every 0/1-request history."""
    worlds = bank["worlds"]
    if len(worlds) != 48:
        raise ValueError("Expected all six identity permutations and eight state vectors")
    initial = ("reference", "initial")
    for world in worlds:
        if len(compatible_labels(bank, world, initial)) != 2:
            raise ValueError("Initial observation leaks the claim label")
        for view in VIEWS:
            if len(compatible_labels(bank, world, (*initial, view))) != 2:
                raise ValueError("A single request already identifies the claim")
        target = "rear_" + SLOTS[world["permutation"].index(0)]
        if compatible_labels(bank, world, (*initial, "front_survey", target)) != {physical_label(world)}:
            raise ValueError("The intended two-step path does not identify the claim")
    fixed = {}
    for pair in combinations(VIEWS, 2):
        fixed[" -> ".join(pair)] = sum(
            len(compatible_labels(bank, w, (*initial, *pair))) == 1 for w in worlds
        ) / len(worlds)
    return {"worlds": len(worlds), "all_zero_or_one_request_histories_ambiguous": True,
            "all_target_two_request_histories_identified": True,
            "best_fixed_two_view_identification_rate": max(fixed.values()),
            "fixed_two_view_identification_rates": fixed,
            "scope": "Finite rendered world bank and public request menu; not human visual review"}


def endpoint(image_id, bbox):
    if bbox is None or len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Missing or empty visible evidence region")
    return {"image_id": image_id, "time": "reference" if image_id == "reference" else "inspection",
            "bbox": [int(x) for x in bbox]}


def visible_region(row, bank_path, kind):
    """Use actual first-hit pixels rather than a projected-box visibility proxy."""
    mask_path = bank_path.parent / row["visibility_checks"]["mask_path"]
    with Image.open(mask_path) as image:
        mask = image.convert("RGB").getchannel(2 if kind == "defect" else 1)
        width, height = mask.size
        pixels = mask.load()
        runs = []
        for y in range(height):
            start = None
            for x in range(width + 1):
                visible = x < width and pixels[x, y] > 127
                if visible and start is None:
                    start = x
                if not visible and start is not None:
                    runs.append([y, start, x])
                    start = None
    if not runs:
        raise ValueError("The diagnostic surface has no visible rendered pixels")
    box = [min(run[1] for run in runs), min(run[0] for run in runs),
           max(run[2] for run in runs), max(run[0] for run in runs) + 1]
    return box, {"size": [width, height], "runs": runs,
                 "source_mask_sha256": row["visibility_checks"]["mask_sha256"],
                 "channel": "scratch" if kind == "defect" else "rear_panel"}


def make_case(raw, bank, bank_path, asset_root, world, pair_id, member):
    position = world["permutation"].index(0)
    target = "rear_" + SLOTS[position]
    rows = selected_rows(raw, world["permutation"], world["back_states"])
    reference = endpoint("reference", rows["reference"]["boxes"]["front"])
    front = endpoint("front_survey", rows["front_survey"]["boxes"][SLOTS[position]])
    rear = endpoint(target, rows[target]["boxes"]["object"])
    label = physical_label(world)
    kind = "defect" if label == "Supported" else "panel"
    diagnostic_box, pixel_mask = visible_region(rows[target], bank_path, kind)
    diagnostic = endpoint(target, diagnostic_box)
    links = [{"relation": "same_object", "left": reference, "right": front},
             {"relation": "same_object", "left": front, "right": rear}]
    state_requirements = [diagnostic, *deepcopy(links)]
    annotation = {
        "protocol": "evidence-chain-v1", "status": "procedural",
        "annotation_origin": "Renderer identities and visible surface masks; finite-world certificate",
        "verdict": label,
        "subclaims": [
            {"id": "identity", "kind": "identity", "truth": "Supported",
             "minimal_evidence_sets": {"Supported": [[deepcopy(links[0])]]}},
            {"id": "condition", "kind": "state", "truth": label,
             "minimal_evidence_sets": {label: [state_requirements]}},
        ],
    }
    family = "sim-search-" + digest([raw["seed"], pair_id, member])[:20]
    evidence = evidence_rows(rows, bank_path, asset_root)
    base = {
        "family_id": family, "paired_history_id": pair_id, "task": "identity_condition",
        "category": "procedural_multistep", "official_split": "held-out-procedural",
        "asset_root": str(asset_root), "provenance_truth": label,
        "claim": "The crate shown in the reference photograph has a scratch on its rear panel at inspection.",
        "claim_parts": [
            {"id": "identity", "text": "Identify the reference crate among the inspection crates."},
            {"id": "condition", "text": "Its rear panel has a scratch at inspection."},
        ],
        "task_instructions": (
            "Position names left, center and right address inspection locations as seen in the front survey, "
            "rather than object identities. Match the reference using visible appearance. "
            "Photographs show one fixed inspection; requests only release recorded views. "
            "Cite the corresponding object regions across photographs and the diagnostic rear-panel region. "
            "The original request returns the initial camera at its existing resolution."
        ),
        "request_options": {"objects": ["inspection scene"], "times": ["inspection"], "views": list(VIEWS)},
        "annotation": annotation,
        "multistep_regions": {"reference": reference, "front_target": front, "rear_target": rear,
                              "diagnostic": diagnostic, "links": deepcopy(links)},
        "multistep_certificate": {
            "version": "finite-world-v1", "bank": bank, "bank_sha256": bank_digest(bank),
            "world_id": world["world_id"], "target_rear_view": target,
            "required_links": deepcopy(links),
            "diagnostic_region": {**diagnostic, "kind": kind, "pixel_mask": pixel_mask},
            "require_pixel_mask": True,
            "available_views": list(VIEWS),
            "renderer_manifest_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        },
    }
    cases = []
    for variant in ("Sufficient", "Obtainable", "Missing", "Unavailable"):
        case = deepcopy(base)
        case.update({"case_id": family + "-" + variant, "variant": variant,
                     "initial": ["reference", "initial"], "evidence": deepcopy(evidence)})
        if variant == "Sufficient":
            case["initial"] += ["front_survey", target]
        if variant in {"Missing", "Unavailable"}:
            case["multistep_certificate"]["available_views"].remove("front_survey")
            if variant == "Missing":
                case["evidence"] = [e for e in case["evidence"] if e["id"] != "front_survey"]
            else:
                next(e for e in case["evidence"] if e["id"] == "front_survey")["available"] = False
        Environment(case, asset_root, budget=6)
        cases.append(case)
    return cases


def build(banks, output, limit_banks=None):
    bank_paths = sorted(banks.glob("*/bank.json"))
    if limit_banks is not None:
        if limit_banks < 1 or len(bank_paths) < limit_banks:
            raise ValueError("The requested number of completed banks is not available")
        bank_paths = bank_paths[:limit_banks]
    if not bank_paths:
        raise ValueError("No completed rendered banks")
    asset_root = banks.parent.resolve()
    loaded = [(path, json.loads(path.read_text())) for path in bank_paths]
    seeds = [raw["seed"] for _, raw in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate scene seed; exclude preview or repeated banks")
    if len({raw["renderer_source_sha256"] for _, raw in loaded}) != 1:
        raise ValueError("Do not mix renderer revisions in one frozen cohort")
    if len({digest(raw["render_settings"]) for _, raw in loaded}) != 1:
        raise ValueError("Do not mix render configurations in one frozen cohort")
    cases, audits, hashes = [], [], {}
    for index, (bank_path, raw) in enumerate(loaded):
        if raw.get("target_identity", 0) != 0:
            raise ValueError("This construction expects reference identity zero")
        validate_render_bank(bank_path, raw)
        canonicalize(bank_path, raw)
        bank = view_bank(raw, bank_path, asset_root)
        audits.append({"seed": raw["seed"], **audit_bank(bank), "certificate_audit": certify_bank(bank)})
        pair = list(combinations(range(3), 2))[index % 3]
        orientation = index % 2
        rng = random.Random(20261107 + raw["seed"])
        states = [rng.randrange(2) for _ in SLOTS]
        states[pair[0]], states[pair[1]] = orientation, 1 - orientation
        pair_id = "sim-bank-" + digest(raw["seed"])[:16]
        for member, position in enumerate(pair):
            candidates = [w for w in bank["worlds"] if w["permutation"].index(0) == position and w["back_states"] == states]
            world = rng.choice(candidates)
            cases.extend(make_case(raw, bank, bank_path, asset_root, world, pair_id, member))
        hashes[str(bank_path.relative_to(asset_root))] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs in constructed dataset")
    encoded = json.dumps(cases, indent=2) + "\n"
    manifest = {
        "schema_version": "procedural-search-v1", "annotation_status": "procedural",
        "families": len(cases) // 4, "cases": len(cases), "paired_scene_banks": len(bank_paths),
        "dataset_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "bank_manifests": hashes,
        "renderer_source_sha256": loaded[0][1]["renderer_source_sha256"],
        "render_settings": loaded[0][1]["render_settings"],
        "budget": 6, "minimum_informative_photo_requests": 2,
        "counterfactual_worlds_per_bank": 48,
        "scope": "Controlled procedural identity and condition search; not independently reviewed real photographs",
        "split_group": "paired_history_id; both matched worlds remain in the same split and bootstrap group",
        "selection": "All completed banks in sorted order; positions cycle through all unordered pairs; each paired bank has one positive and one negative case with the same mixed state vector",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cases.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Frozen dataset differs; choose a new output directory")
    path.write_text(encoded)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "construction_audit.json").write_text(json.dumps({"banks": audits, "manifest": manifest}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "bank_manifests"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-banks", type=int)
    args = parser.parse_args()
    build(args.banks.resolve(), args.output.resolve(), args.limit_banks)


if __name__ == "__main__":
    main()
