"""Private state, real pixels, budgets and source-preserving visual tools."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from .grounding import chain_report, sufficient, validate_annotation
from .ocr import RapidOCRBackend
from .tools import ActionError, VERDICTS, validate_action
from .world import EvidenceWorld


@dataclass(frozen=True)
class Costs:
    inspect: int = 1
    crop: int = 1
    zoom: int = 1
    ocr: int = 2
    compare: int = 2
    request_photo: int = 3
    assess_quality: int = 1
    read_metadata: int = 1
    ground_text_to_image: int = 3
    ground_image_to_image: int = 3
    ground_image_to_text: int = 2


class Environment:
    MAX_VIEW_PIXELS = 1024 * 1024

    def __init__(self, case, asset_root, budget=12, *, costs=None, ocr_backend=None, grounding_backend=None):
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        validate_annotation(case["annotation"])
        self._case = deepcopy(case)
        self._root = Path(asset_root).resolve()
        self._evidence = {e["id"]: deepcopy(e) for e in case["evidence"]}
        if len(self._evidence) != len(case["evidence"]):
            raise ValueError("duplicate evidence IDs")
        self._released = set(case["initial"])
        if not self._released <= self._evidence.keys():
            raise ValueError("unknown initial evidence")
        if any(not self._evidence[i]["available"] for i in self._released):
            raise ValueError("unavailable initial evidence")
        self._world = EvidenceWorld.from_case(case)
        self._views = {}
        self._ocr = ocr_backend if ocr_backend is not None else RapidOCRBackend()
        self._grounding = grounding_backend
        self.budget = budget
        self.costs = costs or Costs()
        if any(type(v) is not int or v <= 0 for v in asdict(self.costs).values()):
            raise ValueError("tool costs must be positive integers")
        self.calls = self.requests = self.tool_cost = self.request_cost = 0
        self.finished = False
        self.decision = None

    def fork(self):
        """Branch evidence/history/budget; share only immutable world and OCR backend."""
        branch = object.__new__(type(self))
        branch.__dict__ = {k: (v if k in {"_world", "_ocr", "_grounding"} else deepcopy(v))
                           for k, v in self.__dict__.items()}
        return branch

    @staticmethod
    def _check_box(box, size, coordinate_space="display_pixels"):
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= size[0]
                or not 0 <= box[1] < box[3] <= size[1]):
            raise ActionError("invalid_bbox", {"coordinate_space": coordinate_space,
                "display_size" if coordinate_space == "display_pixels" else "source_size": list(size),
                "valid_bounds": [0, 0, *size]})

    def _load(self, image_id):
        e = self._evidence[image_id]
        path = (self._root / e["path"]).resolve()
        if not path.is_relative_to(self._root):
            raise ActionError("invalid_asset")
        with Image.open(path) as image:
            return image.convert("RGB")

    def _render(self, reference):
        if reference in self._views:
            descriptor = self._views[reference]
            image_id, operations = descriptor["image_id"], descriptor["operations"]
        elif reference in self._released:
            image_id, operations = reference, []
        else:
            raise ActionError("unreleased_image")
        image = self._load(image_id)
        evidence = self._evidence[image_id]
        source_size = evidence.get("source_size", list(image.size))
        box = list(evidence.get("source_bbox", [0, 0, *image.size]))
        for operation in operations:
            if operation["type"] == "crop":
                region = operation["bbox"]
                width, height = image.size
                sx, sy = (box[2] - box[0]) / width, (box[3] - box[1]) / height
                box = [box[0] + region[0] * sx, box[1] + region[1] * sy,
                       box[0] + region[2] * sx, box[1] + region[3] * sy]
                image = image.crop(tuple(region))
            else:
                image = image.resize(tuple(operation["size"]), Image.Resampling.LANCZOS)
        return image, image_id, operations, box, source_size

    def _image(self, reference, bbox=None):
        image, image_id, operations, box, source_size = self._render(reference)
        if bbox is not None:
            self._check_box(bbox, image.size)
            sx, sy = (box[2] - box[0]) / image.width, (box[3] - box[1]) / image.height
            box = [box[0] + bbox[0] * sx, box[1] + bbox[1] * sy,
                   box[0] + bbox[2] * sx, box[1] + bbox[3] * sy]
            image = image.crop(tuple(bbox))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        e = self._evidence[image_id]
        return {"image_id": image_id, "view_id": reference, "source_id": e["source_id"],
                "party": e["party"], "time": e["time"],
                "source_bbox": [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])],
                "source_size": source_size, "display_size": list(image.size),
                "image_png": buffer.getvalue()}

    def observation(self):
        return {"claim": self._case["claim"], "claim_parts": deepcopy(self._case.get("claim_parts", [])),
                "budget": self.budget,
                "images": [self._image(i) for i in sorted(self._released)],
                "request_options": deepcopy(self._case.get("request_options", {})),
                "tool_costs": asdict(self.costs), "ocr_available": bool(self._ocr.available),
                "finished": self.finished}

    def _validate_citations(self, citations):
        for citation in citations:
            image_id = citation["image_id"]
            if image_id not in self._released:
                raise ActionError("unreleased_citation")
            image = self._image(image_id)
            self._check_box(citation["bbox"], image["source_size"], "source_pixels")
            if citation["time"] != image["time"]:
                raise ActionError("citation_time_mismatch")
            if not covers_box(image["source_bbox"], citation["bbox"]):
                raise ActionError("citation_outside_observed_region")

    def _validate_links(self, links):
        for link in links:
            self._validate_citations([link["left"], link["right"]])
            if link["left"]["image_id"] == link["right"]["image_id"]:
                raise ActionError("link_requires_distinct_images")

    def _derive(self, reference, operation):
        _, image_id, operations, _, _ = self._render(reference)
        operations = [*operations, operation]
        key = "view-" + hashlib.sha256(json.dumps([image_id, operations], sort_keys=True).encode()).hexdigest()[:16]
        previously_known = key in self._views
        self._views[key] = {"image_id": image_id, "operations": operations}
        try:
            return self._image(key)
        except Exception:
            if not previously_known:
                self._views.pop(key, None)
            raise

    def step(self, action):
        if self.finished:
            raise ActionError("episode_finished")
        validate_action(action)
        kind = action["type"]
        if kind == "finish":
            self._validate_citations(action["citations"])
            self._validate_links(action.get("links", []))
            self.finished = True
            self.decision = deepcopy(action)
            return {"decision": deepcopy(action), "budget": self.budget, "finished": True}
        cost = getattr(self.costs, kind)
        if self.budget < cost:
            raise ActionError("insufficient_budget")
        if kind == "request_photo":
            selected = self._world.select(action["query"], self._released)
            if selected is None:
                result = {"status": "unable_to_provide", "images": []}
            else:
                previously_released = selected["id"] in self._released
                self._released.add(selected["id"])
                try:
                    images = [self._image(selected["id"])]
                except Exception:
                    if not previously_released:
                        self._released.discard(selected["id"])
                    raise
                result = {"status": "provided", "images": images}
        elif kind.startswith("ground_"):
            result = self._ground(action)
        elif kind == "compare":
            result = {"images": [self._image(i) for i in action["image_ids"]]}
        else:
            reference = action["image_id"]
            image, _, _, source_box, _ = self._render(reference)
            if kind == "inspect":
                result = {"images": [self._image(reference)]}
            elif kind == "crop":
                self._check_box(action["bbox"], image.size)
                result = {"images": [self._derive(reference, {"type": "crop", "bbox": action["bbox"]})]}
            elif kind == "zoom":
                size = [max(1, round(v * action["factor"])) for v in image.size]
                if size[0] * size[1] > self.MAX_VIEW_PIXELS:
                    raise ActionError("view_too_large")
                result = {"images": [self._derive(reference, {"type": "resize", "size": size})]}
            elif kind == "assess_quality":
                gray = image.convert("L")
                histogram = gray.histogram()
                stats = ImageStat.Stat(gray)
                result = {"image_id": self._image(reference)["image_id"], "view_id": reference,
                    "display_size": list(image.size), "luminance_mean": stats.mean[0],
                    "contrast_std": stats.stddev[0],
                    "edge_variance": ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0],
                    "dark_fraction": sum(histogram[:5])/(image.width*image.height),
                    "bright_fraction": sum(histogram[251:])/(image.width*image.height)}
            elif kind == "read_metadata":
                metadata = self._image(reference)
                source_id = metadata["image_id"]
                path = (self._root / self._evidence[source_id]["path"]).resolve()
                with Image.open(path) as original:
                    exif = original.getexif()
                    allowed = {name: str(exif[tag]) for tag, name in [(274,"orientation"),(36867,"DateTimeOriginal")]
                               if tag in exif}
                result = {k: metadata[k] for k in ("image_id","view_id","source_id","time","party","source_bbox","source_size","display_size")}
                result.update(embedded_exif=allowed, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            else:
                if not self._ocr.available:
                    raise ActionError("ocr_unavailable")
                rows = self._ocr.read(image)
                metadata = self._image(reference)
                x0, y0, x1, y1 = source_box
                for row in rows:
                    row["source_polygon"] = [[x0 + x * (x1-x0)/image.width,
                                              y0 + y * (y1-y0)/image.height] for x, y in row["polygon"]]
                result = {"image_id": metadata["image_id"], "view_id": reference,
                          "source_id": metadata["source_id"], "time": metadata["time"],
                          "text_regions": rows, "status": "read"}
        self.budget -= cost
        self.calls += 1
        if kind == "request_photo":
            self.requests += 1
            self.request_cost += cost
        else:
            self.tool_cost += cost
        return {**result, "budget": self.budget, "finished": False}

    @property
    def grounding_available(self):
        kinds = {"ground_text_to_image": "text_image", "ground_image_to_image": "image_image",
                 "ground_image_to_text": "image_text"}
        return [tool for tool, kind in kinds.items() if self._grounding and self._grounding.available(kind)]

    def _ground(self, action):
        kind = action["type"]
        if kind not in self.grounding_available:
            raise ActionError("grounding_unavailable")
        target = action.get("target_image_id", action.get("image_id"))
        image, _, _, box, _ = self._render(target)
        metadata = {k: v for k, v in self._image(target).items() if k != "image_png"}
        if kind == "ground_text_to_image":
            proposals = self._grounding.text_image(image, action["text"])
        elif kind == "ground_image_to_image":
            query, *_ = self._render(action["query_image_id"])
            proposals = self._grounding.image_image(query, image, action.get("top_k", 3))
            metadata["query_view_id"] = action["query_image_id"]
        else:
            proposals = self._grounding.image_text(image, action["candidates"])
        for proposal in proposals:
            a, b, c, d = proposal.get("bbox", [0, 0, image.width, image.height])
            proposal["source_bbox"] = [max(box[0], box[0]+a*(box[2]-box[0])/image.width),
                max(box[1], box[1]+b*(box[3]-box[1])/image.height),
                min(box[2], box[0]+c*(box[2]-box[0])/image.width),
                min(box[3], box[1]+d*(box[3]-box[1])/image.height)]
        return {**metadata, "proposals": proposals, "status": "predicted"}

    def evaluate(self, decision):
        """Evaluator-only terminal supervision, never tool feedback."""
        verdict = decision["verdict"]
        citations = decision.get("citations", [])
        links = decision.get("links", [])
        valid = True
        try:
            # Legacy evaluation records may omit time; runtime finish requires it.
            citations = [dict(c, time=c.get("time", self._evidence[c["image_id"]]["time"])) for c in citations]
            self._validate_citations(citations)
            self._validate_links(links)
        except (KeyError, TypeError, ValueError):
            valid = False
        grounded = valid and sufficient(self._case["annotation"], verdict, citations, links)
        correct = verdict == self._case["annotation"]["verdict"]
        abstains = verdict == "Need more evidence"
        annotation = self._case["annotation"]
        chain = annotation.get("protocol") == "evidence-chain-v1"
        audit = chain_report(annotation, verdict, citations, links) if chain and valid else {}
        requirements = annotation.get("minimal_evidence_sets", {})
        region_protocol = any(isinstance(item, dict) for alternatives in requirements.values()
                              for group in alternatives for item in group)
        return {"correct": correct, "grounded_correct": correct and valid and (abstains or grounded),
                "unsupported_decision": not abstains and not grounded,
                "coverage": not abstains, "requests": self.requests, "tool_calls": self.calls,
                "tool_cost": self.tool_cost, "request_cost": self.request_cost,
                "remaining_budget": self.budget,
                "evidence_audit": audit,
                "grounding_protocol": "evidence-chain-v1" if chain else "region-time-v1" if region_protocol else "image-set-proxy-v1"}


def covers_box(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]
