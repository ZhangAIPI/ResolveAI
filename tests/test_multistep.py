"""Finite-world, localization and real-observation invariants; no model inference."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from itertools import permutations, product
import json
import unittest

from PIL import Image, PngImagePlugin

from resolveai.multistep import (
    INITIAL_VIEWS, REQUEST_VIEWS, REAR_VIEWS, VIEWS, VERSION,
    bank_digest, certify_bank, evaluate, observation_token,
)


def digest(value):
    return sha256(str(value).encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metadata(view):
    return {"image_id": view, "view_id": view, "source_id": view,
            "party": "submitted", "object": "public-camera-slot", "time": "capture",
            "camera_view": view, "source_bbox": [0, 0, 100, 100],
            "source_size": [100, 100], "display_size": [100, 100]}


def token(view, rgb_key):
    info = metadata(view)
    return {"view": view, "rgb_sha256": digest(rgb_key),
            "metadata_sha256": sha256(canonical(info).encode()).hexdigest(),
            "metadata": info, "derived": False}


def make_bank():
    worlds = []
    for perm in permutations(range(3)):
        for states in product((0, 1), repeat=3):
            views = {}
            for view in VIEWS:
                key = ("front", perm) if view == "front_survey" else (
                    (view, states[REAR_VIEWS.index(view)]) if view in REAR_VIEWS else view)
                views[view] = token(view, key)
            worlds.append({
                "world_id": "p" + "".join(map(str, perm)) + "-s" + "".join(map(str, states)),
                "permutation": list(perm), "back_states": list(states),
                "observations": {v: {k: t[k] for k in ("rgb_sha256", "metadata_sha256", "metadata")}
                                 for v, t in views.items()},
            })
    return {"worlds": worlds}


def endpoint(view, box):
    return {"image_id": view, "time": "capture", "bbox": list(box)}


def fixture(positive=True):
    bank = make_bank()
    states = [1, 0, 1] if positive else [0, 1, 1]
    actual = next(w for w in bank["worlds"] if w["permutation"] == [0, 1, 2]
                  and w["back_states"] == states)
    ref = endpoint("reference", [20, 20, 80, 80])
    front = endpoint("front_survey", [5, 10, 30, 85])
    rear = endpoint("rear_left", [10, 10, 90, 90])
    links = [{"relation": "same_object", "left": ref, "right": front, "min_iou": .5},
             {"relation": "same_object", "left": front, "right": rear, "min_iou": .5}]
    region = endpoint("rear_left", [30, 35, 45, 50] if positive else [15, 15, 85, 85])
    case = {
        "initial": list(INITIAL_VIEWS),
        "annotation": {"status": "procedural"},
        "multistep_certificate": {
            "version": VERSION, "bank": bank, "bank_sha256": bank_digest(bank),
            "world_id": actual["world_id"], "target_rear_view": "rear_left",
            "required_links": links,
            "diagnostic_region": {**region, "kind": "defect" if positive else "panel"},
        },
    }
    observed = []
    for view in (*INITIAL_VIEWS, "front_survey", "rear_left"):
        t = token(view, "placeholder")
        t.update(actual["observations"][view])
        observed.append(t)
    decision = {"type": "finish", "verdict": "Supported" if positive else "Refuted",
                "citations": [region],
                "links": [{k: v for k, v in link.items() if k != "min_iou"} for link in links]}
    return case, decision, observed


class CertificateTests(unittest.TestCase):
    def test_complete_grid_proves_two_requests_and_conditional_next_view(self):
        report = certify_bank(make_bank())
        self.assertEqual(report["minimum_request_count"], 2)
        self.assertEqual(report["single_request_ambiguity_checks"], 240)
        self.assertEqual(report["two_request_identifiable_worlds"], 48)
        self.assertFalse(report["independent_human_review"])
        pairs = report["paired_next_action_witnesses"]
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(p["next_view_a"] != p["next_view_b"] for p in pairs))

    def test_missing_or_duplicate_worlds_are_not_certified(self):
        bank = make_bank()
        with self.assertRaises(ValueError):
            certify_bank({"worlds": bank["worlds"][:-1]})
        bank["worlds"][-1] = deepcopy(bank["worlds"][0])
        with self.assertRaises(ValueError):
            certify_bank(bank)

    def test_metadata_or_hidden_state_leaks_break_isolation(self):
        for view in ("initial", "original", "front_survey", "rear_left"):
            bank = make_bank()
            bank["worlds"][0]["observations"][view]["metadata_sha256"] = digest("leak")
            with self.assertRaises(ValueError, msg=view):
                certify_bank(bank)

    def test_world_dependent_metadata_cannot_substitute_for_visual_evidence(self):
        bank = make_bank()
        for w in bank["worlds"]:
            w["observations"]["front_survey"]["metadata_sha256"] = digest(w["permutation"])
        with self.assertRaises(ValueError):
            certify_bank(bank)

    def test_indistinguishable_defect_pixels_are_not_certified(self):
        bank = make_bank()
        zero = next(w for w in bank["worlds"] if w["back_states"][0] == 0)["observations"]["rear_left"]
        for w in bank["worlds"]:
            w["observations"]["rear_left"] = deepcopy(zero)
        with self.assertRaises(ValueError):
            certify_bank(bank)


class ScoringTests(unittest.TestCase):
    def test_two_real_view_tokens_and_localized_chain_are_grounded(self):
        case, decision, observed = fixture()
        result = evaluate(case, decision, observed)
        self.assertTrue(result["simulator_grounded_correct"])
        self.assertEqual(result["compatible_world_count"], 4)
        self.assertEqual(result["minimum_request_count"], 2)
        self.assertFalse(result["independent_human_review"])
        self.assertEqual(case["annotation"]["status"], "procedural")

    def test_no_single_view_resolves_even_a_lucky_rear_request(self):
        case, decision, observed = fixture()
        for view in REQUEST_VIEWS:
            actual = next(w for w in case["multistep_certificate"]["bank"]["worlds"]
                          if w["world_id"] == case["multistep_certificate"]["world_id"])
            t = token(view, "unused")
            t.update(actual["observations"][view])
            result = evaluate(case, decision, observed[:2] + [t])
            self.assertFalse(result["procedural_evidence_sufficient"], view)
            self.assertFalse(result["simulator_grounded_correct"], view)

    def test_wrong_rear_and_repeated_originals_do_not_count_as_search(self):
        case, decision, observed = fixture()
        actual = next(w for w in case["multistep_certificate"]["bank"]["worlds"]
                      if w["world_id"] == case["multistep_certificate"]["world_id"])
        wrong = token("rear_center", "unused")
        wrong.update(actual["observations"]["rear_center"])
        self.assertFalse(evaluate(case, decision, observed[:3] + [wrong])["procedural_evidence_sufficient"])
        original = token("original", "unused")
        original.update(actual["observations"]["original"])
        result = evaluate(case, decision, observed[:2] + [original, original])
        self.assertFalse(result["procedural_evidence_sufficient"])
        self.assertEqual(result["acquired_views"], ["original"])

    def test_whole_image_citation_or_wrong_identity_does_not_pass(self):
        case, decision, observed = fixture()
        whole = deepcopy(decision)
        whole["citations"][0]["bbox"] = [0, 0, 100, 100]
        result = evaluate(case, whole, observed)
        self.assertTrue(result["source_label_correct"])
        self.assertFalse(result["simulator_grounded_correct"])
        self.assertFalse(result["localized_grounded_correct"])
        self.assertTrue(result["chain_grounded_correct"])
        self.assertTrue(result["unlocalized_definite"])
        self.assertFalse(result["unsupported_decision"])
        wrong = deepcopy(decision)
        wrong["links"][0]["right"]["bbox"] = [60, 10, 90, 85]
        result = evaluate(case, wrong, observed)
        self.assertFalse(result["simulator_grounded_correct"])
        self.assertFalse(result["chain_grounded_correct"])
        wrong = deepcopy(decision)
        wrong["links"][0]["right"]["bbox"] = [0, 0, 100, 100]
        result = evaluate(case, wrong, observed)
        self.assertFalse(result["chain_grounded_correct"])
        self.assertFalse(result["localized_grounded_correct"])

    def test_mask_pixels_not_empty_bbox_area_define_diagnostic_coverage(self):
        case, decision, observed = fixture()
        cert = case["multistep_certificate"]
        cert["require_pixel_mask"] = True
        cert["diagnostic_region"]["pixel_mask"] = {
            "size": [100, 100], "runs": [[y, 42, 45] for y in range(35, 50)]}
        blank = deepcopy(decision)
        # This covers two thirds of the bbox but none of the visible defect.
        blank["citations"][0]["bbox"] = [30, 35, 40, 50]
        result = evaluate(case, blank, observed)
        self.assertTrue(result["source_label_correct"])
        self.assertFalse(result["chain_grounded_correct"])
        self.assertFalse(result["localized_grounded_correct"])
        full = deepcopy(decision)
        full["citations"][0]["bbox"] = [0, 0, 100, 100]
        result = evaluate(case, full, observed)
        self.assertTrue(result["chain_grounded_correct"])
        self.assertFalse(result["localized_grounded_correct"])

    def test_diagnostic_mask_coverage_unions_without_counting_overlap_twice(self):
        case, decision, observed = fixture()
        cert = case["multistep_certificate"]
        cert["require_pixel_mask"] = True
        cert["diagnostic_region"]["pixel_mask"] = {
            "size": [100, 100], "runs": [[y, 30, 45] for y in range(35, 50)]}
        fragment = endpoint("rear_left", [30, 35, 34, 50])
        decision["citations"] = [fragment, deepcopy(fragment)]
        self.assertFalse(evaluate(case, decision, observed)["chain_grounded_correct"])
        decision["citations"].append(endpoint("rear_left", [34, 35, 38, 50]))
        result = evaluate(case, decision, observed)
        self.assertTrue(result["chain_grounded_correct"])
        self.assertFalse(result["localized_grounded_correct"])

    def test_required_mask_is_checked_during_no_model_preflight(self):
        for bad in (None, {"size": [100, 100], "runs": []},
                    {"size": [100, 100], "runs": [[35, 0, 5]]},
                    {"size": [100, 100], "runs": [[35, 42, 45], [35, 42, 45]]},
                    {"size": [100, 200], "runs": [[35, 42, 45]]}):
            case, _, observed = fixture()
            case["multistep_certificate"]["require_pixel_mask"] = True
            if bad is not None:
                case["multistep_certificate"]["diagnostic_region"]["pixel_mask"] = bad
            result = evaluate(case, None, observed[:2])
            self.assertFalse(result["certificate_valid"])
            self.assertFalse(result["chain_grounded_correct"])

    def test_negative_requires_visible_panel_and_not_empty_defect_mask(self):
        case, decision, observed = fixture(False)
        result = evaluate(case, decision, observed)
        self.assertTrue(result["simulator_grounded_correct"])
        self.assertTrue(result["chain_grounded_correct"])
        small = deepcopy(decision)
        small["citations"][0]["bbox"] = [30, 35, 45, 50]
        self.assertFalse(evaluate(case, small, observed)["simulator_grounded_correct"])
        case["multistep_certificate"]["diagnostic_region"]["bbox"] = []
        self.assertFalse(evaluate(case, decision, observed)["simulator_grounded_correct"])

    def test_wrong_label_is_not_grounded_despite_complete_localization(self):
        case, decision, observed = fixture()
        decision["verdict"] = "Refuted"
        result = evaluate(case, decision, observed)
        self.assertTrue(result["procedural_evidence_sufficient"])
        self.assertFalse(result["source_label_correct"])
        self.assertFalse(result["simulator_grounded_correct"])

    def test_none_or_malformed_finish_is_a_failure_not_a_dropped_episode(self):
        case, decision, observed = fixture()
        for bad in (None, {"type": "inspect", "image_id": "reference"},
                    {"verdict": "Supported"}, {**decision, "unexpected": True}):
            result = evaluate(case, bad, observed)
            self.assertFalse(result["schema_valid_finish"])
            self.assertFalse(result["source_label_correct"])
            self.assertFalse(result["simulator_grounded_correct"])
            self.assertTrue(result["procedural_evidence_sufficient"])

    def test_missing_or_forged_observation_never_filled_from_truth(self):
        case, decision, observed = fixture()
        self.assertFalse(evaluate(case, decision, observed[1:])["certificate_valid"])
        altered = deepcopy(observed)
        altered[-1]["rgb_sha256"] = digest("different actual image")
        self.assertFalse(evaluate(case, decision, altered)["certificate_valid"])
        altered = deepcopy(observed)
        altered[-1]["metadata"]["time"] = "SECRET"
        self.assertFalse(evaluate(case, decision, altered)["certificate_valid"])

    def test_unreleased_or_out_of_bounds_citations_fail(self):
        case, decision, observed = fixture()
        bad = deepcopy(decision)
        bad["citations"][0]["image_id"] = "rear_right"
        self.assertFalse(evaluate(case, bad, observed)["citations_valid"])
        bad = deepcopy(decision)
        bad["citations"][0]["bbox"] = [0, 0, 101, 100]
        self.assertFalse(evaluate(case, bad, observed)["citations_valid"])

    def test_sufficient_and_unavailable_targets_do_not_change_physical_truth(self):
        case, decision, observed = fixture()
        case["initial"] += ["front_survey", "rear_left"]
        result = evaluate(case, decision, observed)
        self.assertTrue(result["simulator_grounded_correct"])
        self.assertEqual(result["minimum_request_count"], 0)
        case, _, observed = fixture()
        case["multistep_certificate"]["available_views"] = [
            v for v in REQUEST_VIEWS if v != "front_survey"]
        abstain = {"verdict": "Need more evidence", "citations": [], "links": []}
        result = evaluate(case, abstain, observed[:2])
        self.assertEqual(result["episode_target_verdict"], "Need more evidence")
        self.assertEqual(result["physical_source_verdict"], "Supported")
        self.assertTrue(result["episode_target_correct"])
        self.assertTrue(result["simulator_grounded_correct"])
        self.assertFalse(result["source_label_correct"])
        self.assertIsNone(result["minimum_request_count"])

    def test_abstention_is_safe_but_not_search_success_when_evidence_is_obtainable(self):
        case, _, observed = fixture()
        result = evaluate(case, {"verdict": "Need more evidence", "citations": []}, observed[:2])
        self.assertTrue(result["decision_matches_current_evidence"])
        self.assertFalse(result["episode_target_correct"])
        self.assertFalse(result["simulator_grounded_correct"])

    def test_dictionary_tokens_and_derived_zoom_add_no_independent_view(self):
        case, decision, observed = fixture()
        dictionary = {t["view"]: t for t in observed}
        self.assertTrue(evaluate(case, decision, dictionary)["simulator_grounded_correct"])
        derived = deepcopy(observed[0])
        derived["derived"] = True
        derived["metadata"]["view_id"] = "view-zoomed"
        derived["metadata"]["display_size"] = [400, 400]
        derived["metadata_sha256"] = sha256(canonical(derived["metadata"]).encode()).hexdigest()
        result = evaluate(case, None, observed[:2] + [derived])
        self.assertFalse(result["procedural_evidence_sufficient"])
        self.assertEqual(result["acquired_views"], [])


class RealTokenTests(unittest.TestCase):
    def test_decoded_rgb_and_metadata_define_the_observation(self):
        image = Image.new("RGB", (100, 100), (12, 50, 90))
        raw = BytesIO()
        image.save(raw, format="PNG")
        public = {**metadata("reference"), "image_png": raw.getvalue()}
        first = observation_token(public)
        annotated = BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "renderer timestamp is not pixel content")
        image.save(annotated, format="PNG", pnginfo=info)
        second = observation_token({**public, "image_png": annotated.getvalue()})
        self.assertEqual(first["rgb_sha256"], second["rgb_sha256"])
        self.assertEqual(first["metadata_sha256"], second["metadata_sha256"])
        self.assertFalse(first["derived"])
        with self.assertRaises(ValueError):
            observation_token({**public, "display_size": [48, 48]})
        with self.assertRaises(ValueError):
            observation_token({**public, "target_bbox": [5, 10, 30, 85]})
        with self.assertRaises(ValueError):
            observation_token({**public, "private_object_id": "SECRET"})


if __name__ == "__main__":
    unittest.main()
