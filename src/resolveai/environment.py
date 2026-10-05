"""Private case state; only public image observations leave the environment.

Run this module's environment in a separate process from model policies.
Annotations and unreleased paths must never enter policy prompts.
"""
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Costs:
    inspect: int = 1
    crop: int = 1
    compare: int = 2
    request_photo: int = 3


class Environment:
    def __init__(self, case: dict, asset_root: Path, budget: int = 12):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self._case = deepcopy(case)
        self._root = Path(asset_root).resolve()
        self._evidence = {e["id"]: e for e in case["evidence"]}
        if len(self._evidence) != len(case["evidence"]):
            raise ValueError("duplicate evidence IDs")
        self._released = set(case["initial"])
        if not self._released <= self._evidence.keys():
            raise ValueError("unknown initial evidence")
        if any(not self._evidence[i]["available"] for i in self._released):
            raise ValueError("unavailable initial evidence")
        self.budget = budget
        self.costs = Costs()
        self.calls = 0
        self.requests = 0
        self.tool_cost = 0
        self.request_cost = 0
        self.finished = False

    def fork(self):
        """Clone a branch without changing the scene or sibling state."""
        return deepcopy(self)

    def _image(self, evidence_id, bbox=None):
        e = self._evidence[evidence_id]
        path = (self._root / e["path"]).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("asset path escapes root")
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            region = [0, 0, width, height] if bbox is None else bbox
            if (len(region) != 4 or any(type(v) is not int for v in region)
                    or not 0 <= region[0] < region[2] <= width
                    or not 0 <= region[1] < region[3] <= height):
                raise ValueError("bbox must be integer pixel coordinates within source")
            buffer = BytesIO()
            image.crop(tuple(region)).save(buffer, format="PNG")
        return {"image_id": evidence_id, "source_id": e["source_id"],
                "party": e["party"], "time": e["time"],
                "source_bbox": region, "source_size": [width, height],
                "image_png": buffer.getvalue()}

    def observation(self):
        return {"claim": self._case["claim"], "budget": self.budget,
                "images": [self._image(i) for i in sorted(self._released)],
                "finished": self.finished}

    def step(self, action: dict):
        if self.finished:
            raise ValueError("episode already finished")
        kind = action["type"]
        if kind == "finish":
            if action.get("verdict") not in {"Supported", "Refuted", "Need more evidence"}:
                raise ValueError("invalid verdict")
            citations = action.get("citations", [])
            if not isinstance(citations, list):
                raise ValueError("citations must be a list")
            for citation in citations:
                if citation["image_id"] not in self._released:
                    raise ValueError("citation references unreleased evidence")
                self._image(citation["image_id"], citation["bbox"])
            self.finished = True
            return {"decision": deepcopy(action), "budget": self.budget}
        if kind not in {"inspect", "crop", "compare", "request_photo"}:
            raise ValueError("unknown action")
        cost = getattr(self.costs, kind)
        if self.budget < cost:
            raise ValueError("budget exhausted")
        if kind in {"inspect", "crop", "compare"}:
            ids = action.get("image_ids", []) if kind == "compare" else [action["image_id"]]
            if kind == "compare" and (len(ids) != 2 or ids[0] == ids[1]):
                raise ValueError("compare requires two distinct images")
            if any(i not in self._released for i in ids):
                raise ValueError("cannot inspect unreleased evidence")
            images = [self._image(i, action["bbox"] if kind == "crop" else None) for i in ids]
            # The visual model compares these pixels; no oracle text is supplied.
            result = {"images": images}
        else:
            query = action["query"]
            required = {"object", "time", "view"}
            if set(query) != required:
                raise ValueError("request requires object, time and view")
            matches = [e for e in self._evidence.values()
                       if all(e[k] == query[k] for k in required)]
            available = sorted((e for e in matches if e["available"]), key=lambda e: e["id"])
            if available:
                selected = next((e for e in available if e["id"] not in self._released), available[0])
                result = {"status": "provided", "images": [self._image(selected["id"])]}
                self._released.add(selected["id"])
            else:
                # Identical response: material existence is hidden from the agent.
                result = {"status": "unable_to_provide", "images": []}
        self.budget -= cost
        self.calls += 1
        if kind == "request_photo":
            self.requests += 1
            self.request_cost += cost
        else:
            self.tool_cost += cost
        return {**result, "budget": self.budget}

    def evaluate(self, decision: dict):
        """Evaluator-only; never expose this result during an episode."""
        verdict = decision["verdict"]
        citations = decision.get("citations", [])
        valid = True
        cited = set()
        for c in citations:
            try:
                if c["image_id"] not in self._released:
                    raise ValueError("unreleased")
                self._image(c["image_id"], c["bbox"])
                cited.add(c["image_id"])
            except (KeyError, TypeError, ValueError):
                valid = False
        sufficient = valid and any(set(s) <= cited for s in
                self._case["annotation"]["minimal_evidence_sets"].get(verdict, []))
        correct = verdict == self._case["annotation"]["verdict"]
        abstains = verdict == "Need more evidence"
        return {"correct": correct, "grounded_correct": correct and (abstains or sufficient),
                "unsupported_decision": not abstains and not sufficient,
                "coverage": not abstains, "requests": self.requests, "tool_calls": self.calls,
                "tool_cost": self.tool_cost, "request_cost": self.request_cost,
                "remaining_budget": self.budget}
