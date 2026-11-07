"""Finite-world identifiability and simulator-grounded scoring.

The certificate concerns an explicit rendered world bank, not independently
reviewed visual sufficiency. Public observations, never hidden IDs, determine
which worlds remain compatible with an episode.
"""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from itertools import combinations, permutations, product
import json

from PIL import Image

from .grounding import iou, link_matches
from .tools import ActionError, validate_action

INITIAL_VIEWS = ("reference", "initial")
REQUEST_VIEWS = ("original", "front_survey", "rear_left", "rear_center", "rear_right")
REAR_VIEWS = ("rear_left", "rear_center", "rear_right")
VIEWS = (*INITIAL_VIEWS, *REQUEST_VIEWS)
METADATA_FIELDS = (
    "image_id", "view_id", "source_id", "party", "time", "object",
    "camera_view", "source_bbox", "source_size", "display_size", "target_bbox",
)
VERSION = "finite-world-v1"
THRESHOLDS = {
    "identity_iou": .5, "defect_iou": .25, "defect_precision": .25,
    "defect_coverage": .5, "panel_iou": .5, "panel_precision": .5,
    "panel_coverage": .9,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def bank_digest(bank):
    return sha256(_canonical(bank).encode()).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _signature(observation):
    values = tuple(observation[k] for k in ("rgb_sha256", "metadata_sha256"))
    if not all(_valid_hash(v) for v in values):
        raise ValueError("Observation signatures must be SHA-256 hex digests")
    return values


def observation_token(image, view=None):
    """Fingerprint a real Environment image, including all public image metadata.

    Derived crops/zooms are retained for auditing but add no independent camera
    observation. The builder and runner must use this same function.
    """
    if "target_bbox" in image:
        raise ValueError("Target geometry must remain private in this track")
    if set(image) - set(METADATA_FIELDS) - {"image_png"}:
        raise ValueError("Unregistered public image metadata must be audited")
    raw = image["image_png"]
    if not isinstance(raw, bytes):
        raise ValueError("A real image_png byte string is required")
    with Image.open(BytesIO(raw)) as source:
        rgb = source.convert("RGB")
        size = list(rgb.size)
        pixels = rgb.tobytes()
    metadata = {k: image[k] for k in METADATA_FIELDS if k in image}
    if metadata.get("display_size") != size:
        raise ValueError("Image dimensions disagree with public display_size")
    if not all(k in metadata for k in ("image_id", "view_id", "camera_view", "time",
                                       "source_bbox", "source_size")):
        raise ValueError("Incomplete public image metadata")
    token_view = view if view is not None else metadata["camera_view"]
    if token_view not in VIEWS:
        raise ValueError("Unknown public multistep view")
    return {
        "view": token_view,
        "rgb_sha256": sha256(_canonical(["RGB", *size]).encode() + b"\0" + pixels).hexdigest(),
        "metadata_sha256": sha256(_canonical(metadata).encode()).hexdigest(),
        "metadata": metadata,
        "derived": metadata["view_id"] != metadata["image_id"],
    }


def _worlds(bank):
    worlds = bank["worlds"]
    if not isinstance(worlds, list) or len(worlds) != 48:
        raise ValueError("The bank must contain all 48 finite worlds")
    ids, configurations = set(), set()
    for world in worlds:
        world_id = world["world_id"]
        perm, states = world["permutation"], world["back_states"]
        if not isinstance(world_id, str) or not world_id or world_id in ids:
            raise ValueError("World IDs must be unique and nonempty")
        if (not isinstance(perm, list) or sorted(perm) != [0, 1, 2]
                or any(type(x) is not int for x in perm)):
            raise ValueError("Each world needs a three-identity permutation")
        if (not isinstance(states, list) or len(states) != 3
                or any(type(x) is not int or x not in (0, 1) for x in states)):
            raise ValueError("Each world needs three binary rear states")
        key = tuple(perm), tuple(states)
        if key in configurations:
            raise ValueError("Duplicate finite-world configuration")
        if set(world["observations"]) != set(VIEWS):
            raise ValueError("Every world must define all public views")
        for observation in world["observations"].values():
            _signature(observation)
            if "metadata" in observation:
                public = observation["metadata"]
                if ("target_bbox" in public
                        or sha256(_canonical(public).encode()).hexdigest() != observation["metadata_sha256"]):
                    raise ValueError("Embedded public metadata is inconsistent or reveals target geometry")
        ids.add(world_id)
        configurations.add(key)
    expected = set(product(permutations(range(3)), product((0, 1), repeat=3)))
    if configurations != expected:
        raise ValueError("The finite-world configuration grid is incomplete")
    return worlds


def _label(world):
    return "Supported" if world["back_states"][world["permutation"].index(0)] else "Refuted"


def _rear(world):
    return REAR_VIEWS[world["permutation"].index(0)]


def _compatible(worlds, observations):
    return [w for w in worlds if all(_signature(w["observations"][v]) == sig
                                     for v, sig in observations.items())]


def _labels(worlds):
    return sorted({_label(w) for w in worlds})


def _minimum_requests(worlds, actual, initial, available):
    initial_signatures = {v: _signature(actual["observations"][v]) for v in initial}
    unseen = sorted(set(available) - set(initial))
    for count in range(len(unseen) + 1):
        for queries in combinations(unseen, count):
            history = {**initial_signatures, **{
                v: _signature(actual["observations"][v]) for v in queries}}
            if len(_labels(_compatible(worlds, history))) == 1:
                return count
    return None


def certify_bank(bank):
    """Prove a two-request lower bound relative to a frozen finite view bank."""
    worlds = _worlds(bank)
    # Strong isolation is intentional: rear state cannot leak into a front view,
    # and a rear view cannot identify the object's front texture.
    for view in VIEWS:
        if len({w["observations"][view]["metadata_sha256"] for w in worlds}) != 1:
            raise ValueError("Public camera metadata must not reveal hidden state or identity")
    for view in (*INITIAL_VIEWS, "original"):
        if len({_signature(w["observations"][view]) for w in worlds}) != 1:
            raise ValueError("Reference/initial/restored frames must be world-independent")
    front = defaultdict(set)
    for w in worlds:
        front[tuple(w["permutation"])].add(_signature(w["observations"]["front_survey"]))
    if any(len(v) != 1 for v in front.values()) or len(set.union(*front.values())) != 6:
        raise ValueError("Front survey must vary only with the identity permutation")
    for slot, view in enumerate(REAR_VIEWS):
        signatures = [set(), set()]
        for w in worlds:
            signatures[w["back_states"][slot]].add(_signature(w["observations"][view]))
        if any(len(v) != 1 for v in signatures) or signatures[0] == signatures[1]:
            raise ValueError("Each rear view must depend only on its own visible state")
    for w in worlds:
        initial = {v: _signature(w["observations"][v]) for v in INITIAL_VIEWS}
        if len(_labels(_compatible(worlds, initial))) != 2:
            raise ValueError("Initial observations already identify the label")
        for view in REQUEST_VIEWS:
            history = {**initial, view: _signature(w["observations"][view])}
            if len(_labels(_compatible(worlds, history))) != 2:
                raise ValueError("A single request identifies the label")
        front_history = {**initial, "front_survey": _signature(w["observations"]["front_survey"])}
        identifying = [
            view for view in REQUEST_VIEWS
            if len(_labels(_compatible(worlds, {
                **front_history, view: _signature(w["observations"][view])}))) == 1
        ]
        if identifying != [_rear(w)]:
            raise ValueError("The useful second request must match the observed target location")
    paired = []
    for left, right in combinations(range(3), 2):
        witness = None
        for a in worlds:
            if a["permutation"].index(0) != left:
                continue
            for b in worlds:
                if (b["permutation"].index(0) == right
                        and a["back_states"] == b["back_states"]
                        and _label(a) == _label(b)
                        and len(set(a["back_states"])) == 2):
                    witness = {"world_a": a["world_id"], "world_b": b["world_id"],
                               "next_view_a": _rear(a), "next_view_b": _rear(b)}
                    break
            if witness:
                break
        if witness is None:
            raise ValueError("Missing paired location-dependent next-action witness")
        paired.append(witness)
    return {
        "version": VERSION, "bank_sha256": bank_digest(bank), "worlds": 48,
        "minimum_request_count": 2, "single_request_ambiguity_checks": 48 * len(REQUEST_VIEWS),
        "two_request_identifiable_worlds": 48, "paired_next_action_witnesses": paired,
        "independent_human_review": False,
        "scope": "Exact observational equivalence in a finite rendered world/view bank; not human visual sufficiency",
    }


def _box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) is int for v in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _region_matches(required, actual, kind):
    if (actual["image_id"] != required["image_id"] or actual["time"] != required["time"]
            or not _box(required["bbox"]) or not _box(actual["bbox"])):
        return False
    a, b = required["bbox"], actual["bbox"]
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return (iou(a, b) >= THRESHOLDS[kind + "_iou"]
            and intersection / _area(b) >= THRESHOLDS[kind + "_precision"]
            and intersection / _area(a) >= THRESHOLDS[kind + "_coverage"])



def _union_length(intervals):
    end, total = None, 0
    for left, right in sorted(intervals):
        start = left if end is None else max(left, end)
        total += max(0, right - start)
        end = right if end is None else max(end, right)
    return total


def _validate_mask(region, required):
    mask = region.get("pixel_mask")
    if mask is None:
        if required:
            raise ValueError("This rendered case requires a nonempty diagnostic pixel mask")
        return None
    size, runs = mask["size"], mask["runs"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or not isinstance(runs, list) or not runs):
        raise ValueError("Invalid diagnostic pixel-mask dimensions or empty runs")
    if not _box(region["bbox"]):
        raise ValueError("Invalid diagnostic region")
    bx0, by0, bx1, by1 = region["bbox"]
    if not 0 <= bx0 < bx1 <= size[0] or not 0 <= by0 < by1 <= size[1]:
        raise ValueError("Diagnostic region lies outside its source image")
    previous, total = {}, 0
    for run in sorted(runs):
        if (not isinstance(run, list) or len(run) != 3
                or any(type(v) is not int for v in run)):
            raise ValueError("Pixel-mask runs must be integer [y,x0,x1] rows")
        y, x0, x1 = run
        if (not 0 <= y < size[1] or not 0 <= x0 < x1 <= size[0]
                or not bx0 <= x0 < x1 <= bx1 or not by0 <= y < by1):
            raise ValueError("Diagnostic pixels must lie inside the image and registered region")
        if x0 < previous.get(y, -1):
            raise ValueError("Overlapping or duplicate diagnostic pixel-mask runs")
        previous[y] = x1
        total += x1 - x0
    return {"size": size, "runs": runs, "pixels": total}


def _diagnostic_coverage(region, citations, mask):
    boxes = [c["bbox"] for c in citations if c["image_id"] == region["image_id"]
             and c["time"] == region["time"] and _box(c["bbox"])]
    if mask is not None:
        covered = 0
        for y, x0, x1 in mask["runs"]:
            intervals = [(max(x0, b[0]), min(x1, b[2])) for b in boxes
                         if b[1] <= y < b[3] and max(x0, b[0]) < min(x1, b[2])]
            covered += _union_length(intervals)
        return covered / mask["pixels"]
    # Software fixtures may use a rectangular diagnostic region. Real rendered
    # cases explicitly require the actual visible-surface/defect mask above.
    x0, y0, x1, y1 = region["bbox"]
    clipped = [[max(x0, b[0]), max(y0, b[1]), min(x1, b[2]), min(y1, b[3])]
               for b in boxes]
    clipped = [b for b in clipped if b[0] < b[2] and b[1] < b[3]]
    edges = sorted({x0, x1, *[x for b in clipped for x in (b[0], b[2])]})
    area = sum((right - left) * _union_length(
        [(b[1], b[3]) for b in clipped if b[0] <= left and b[2] >= right])
        for left, right in zip(edges, edges[1:]))
    return area / _area(region["bbox"])

def _public_citations_valid(decision, metadata):
    endpoints = list(decision["citations"])
    endpoints += [link[k] for link in decision.get("links", []) for k in ("left", "right")]
    for endpoint in endpoints:
        info = metadata.get(endpoint["image_id"])
        if info is None or endpoint["time"] != info["time"] or not _box(endpoint["bbox"]):
            return False
        box, visible = endpoint["bbox"], info["source_bbox"]
        if not (visible[0] <= box[0] < box[2] <= visible[2]
                and visible[1] <= box[1] < box[3] <= visible[3]):
            return False
    return all(link["left"]["image_id"] != link["right"]["image_id"]
               for link in decision.get("links", []))


def evaluate(case, decision, observations):
    """Score actual observed view hashes and a finish, separately from human review.

    Chain scoring requires localized identity links and diagnostic-mask coverage;
    localized scoring additionally requires a tight diagnostic-region rectangle.
    simulator_grounded_correct is the localized metric, not a human review claim.
    No missing observation is reconstructed from the private actual world.
    A malformed/absent decision remains a failed submission in the denominator.
    """
    result = {
        "source_label_correct": False, "episode_target_correct": False, "schema_valid_finish": False,
        "citations_valid": False, "procedural_evidence_sufficient": False,
        "simulator_grounded_correct": False, "localized_grounded_correct": False,
        "chain_grounded_correct": False, "unlocalized_definite": False,
        "unsupported_decision": False,
        "decision_matches_current_evidence": False, "certificate_valid": False,
        "compatible_world_count": 0, "compatible_labels": [], "acquired_views": [],
        "minimum_request_count": None, "current_evidence_verdict": "Need more evidence",
        "independent_human_review": False, "localization_thresholds": dict(THRESHOLDS),
    }
    try:
        cert = case["multistep_certificate"]
        bank = cert["bank"]
        report = certify_bank(bank)
        if cert["version"] != VERSION or cert["bank_sha256"] != report["bank_sha256"]:
            raise ValueError("Certificate bank checksum/version mismatch")
        worlds = _worlds(bank)
        actual = next(w for w in worlds if w["world_id"] == cert["world_id"])
        if cert["target_rear_view"] != _rear(actual):
            raise ValueError("Certificate target location mismatch")
        region = cert["diagnostic_region"]
        kind = "defect" if _label(actual) == "Supported" else "panel"
        if (region.get("kind") != kind or region["image_id"] != _rear(actual)
                or not _box(region["bbox"])):
            raise ValueError("Certificate needs a nonempty target defect or intact visible panel")
        mask = _validate_mask(region, cert.get("require_pixel_mask", False))
        result["diagnostic_coverage_source"] = "visible-pixel-mask" if mask is not None else "rectangular-software-fixture"
        actual_metadata = actual["observations"][_rear(actual)].get("metadata", {})
        if (mask is not None and actual_metadata.get("source_size") is not None
                and mask["size"] != actual_metadata["source_size"]):
            raise ValueError("Diagnostic mask and source-image dimensions differ")
        initial = set(case.get("initial", INITIAL_VIEWS))
        # Case image IDs use the fixed public camera/view tokens.
        if not set(INITIAL_VIEWS) <= initial or not initial <= set(VIEWS):
            raise ValueError("Invalid initial public views")
        available = set(cert.get("available_views", REQUEST_VIEWS))
        if not available <= set(REQUEST_VIEWS):
            raise ValueError("Invalid available public views")
        if not (initial - set(INITIAL_VIEWS)) <= available:
            raise ValueError("An initial image is unavailable")
        minimum = _minimum_requests(worlds, actual, initial, available)
        cap = cert.get("max_requests", 2)
        if type(cap) is not int or cap < 0:
            raise ValueError("Invalid initial request allowance")
        answerable = minimum is not None and minimum <= cap
        expected = _label(actual) if answerable else "Need more evidence"
        result.update(minimum_request_count=minimum, physical_source_verdict=_label(actual),
                      episode_target_verdict=expected, initially_budget_answerable=answerable)
        if isinstance(observations, dict):
            observations = [dict(token, view=view) for view, token in observations.items()]
        if not isinstance(observations, list):
            raise ValueError("Observed view tokens must be a list or view dictionary")
        seen, metadata, derived = {}, {}, []
        for token in observations:
            view = token["view"]
            if view not in VIEWS:
                raise ValueError("Unknown observed view")
            info = token.get("metadata")
            if (not isinstance(info, dict)
                    or sha256(_canonical(info).encode()).hexdigest() != token["metadata_sha256"]):
                raise ValueError("Observed public metadata checksum mismatch")
            if token.get("derived"):
                derived.append(token)
                continue
            sig = _signature(token)
            if sig != _signature(actual["observations"][view]):
                raise ValueError("Observed pixels/metadata disagree with frozen actual world")
            if view in seen and seen[view] != sig:
                raise ValueError("Inconsistent repeated view")
            if view not in initial and view not in available:
                raise ValueError("An unavailable camera observation was supplied")
            seen[view] = sig
            metadata[info["image_id"]] = info
        if not initial <= seen.keys():
            raise ValueError("Actual initial observations are missing")
        if any(t["metadata"]["image_id"] not in metadata for t in derived):
            raise ValueError("Derived view has no actually observed parent")
        if len(set(seen) - initial) > cap:
            raise ValueError("Observed camera acquisitions exceed the initial request allowance")
        compatible = _compatible(worlds, seen)
        labels = _labels(compatible)
        if not compatible:
            raise ValueError("Observed history matches no finite world")
        sufficient = len(labels) == 1
        result.update(certificate_valid=True, compatible_world_count=len(compatible),
                      compatible_labels=labels, acquired_views=sorted(set(seen) - initial),
                      procedural_evidence_sufficient=sufficient,
                      current_evidence_verdict=labels[0] if sufficient else "Need more evidence")
        if decision is None:
            return result
        payload = dict(decision)
        if payload.pop("type", "finish") != "finish":
            return result
        validate_action({"type": "finish", **payload})
        result["schema_valid_finish"] = True
        valid_citations = _public_citations_valid(payload, metadata)
        result["citations_valid"] = valid_citations
        verdict = payload["verdict"]
        result["source_label_correct"] = valid_citations and verdict == _label(actual)
        result["episode_target_correct"] = valid_citations and verdict == expected
        result["decision_matches_current_evidence"] = valid_citations and verdict == result["current_evidence_verdict"]
        if verdict == "Need more evidence":
            for key in ("simulator_grounded_correct", "localized_grounded_correct",
                        "chain_grounded_correct"):
                result[key] = result["episode_target_correct"]
            return result
        required_links = cert["required_links"]
        if (len(required_links) < 2 or any(link["relation"] != "same_object"
                or link.get("min_iou", .5) < THRESHOLDS["identity_iou"] for link in required_links)):
            raise ValueError("Certificate requires localized identity-chain links")
        links = payload.get("links", [])
        links_valid = (all(any(link_matches(expected, actual_link) for actual_link in links)
                           for expected in required_links)
                       and all(any(link_matches(expected, actual_link) for expected in required_links)
                               for actual_link in links))
        covered = _diagnostic_coverage(region, payload["citations"], mask)
        chain = (result["episode_target_correct"] and sufficient and links_valid
                 and covered >= THRESHOLDS[kind + "_coverage"])
        localized = any(_region_matches(region, c, kind) for c in payload["citations"])
        strict = chain and localized
        result["chain_grounded_correct"] = chain
        result["simulator_grounded_correct"] = strict
        result["localized_grounded_correct"] = strict
        result["unlocalized_definite"] = chain and not strict
        result["unsupported_decision"] = verdict != "Need more evidence" and not chain
        return result
    except (ActionError, ValueError, KeyError, TypeError, StopIteration) as error:
        result["audit_error"] = type(error).__name__ + ": " + str(error)
        result["simulator_grounded_correct"] = False
        result["localized_grounded_correct"] = False
        result["chain_grounded_correct"] = False
        result["unlocalized_definite"] = False
        result["source_label_correct"] = False
        result["episode_target_correct"] = False
        result["unsupported_decision"] = bool(
            result["schema_valid_finish"] and decision
            and decision.get("verdict") != "Need more evidence")
        return result


evaluate_case = evaluate
