"""One canonical tool schema shared by agents, validation and training exports."""
from copy import deepcopy
import re
from jsonschema import Draft202012Validator

VERDICTS = ["Supported", "Refuted", "Need more evidence"]
BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
IMAGE = {"type": "string", "minLength": 1, "description": "Released image ID or derived view_id."}


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


SPECS = {
    "inspect": ("Read an already released image/view; cost 1.", obj({"image_id": IMAGE})),
    "crop": ("Crop display-pixel bbox [x0,y0,x1,y1] from an image/view; cost 1. Preserves occlusion and provenance.",
             obj({"image_id": IMAGE, "bbox": BOX})),
    "zoom": ("Resample existing pixels: factor >1 enlarges, <1 shrinks; cost 1. Does not create evidence or recover details.",
             obj({"image_id": IMAGE, "factor": {"type": "number", "minimum": .125, "maximum": 4}})),
    "ocr": ("Read text from an image/view using a pixel-based OCR model; cost 2. Returns boxes and confidence, not truth.",
            obj({"image_id": IMAGE})),
    "compare": ("Return two released images/views for your visual correspondence and condition comparison; cost 2. No oracle description.",
                obj({"image_ids": {"type": "array", "items": IMAGE, "minItems": 2, "maxItems": 2, "uniqueItems": True}})),
    "request_photo": ("Request material matching object/time/view exactly; cost 3 even on failure. Does not change scene state.",
                      obj({"query": obj({k: {"type": "string", "minLength": 1} for k in ("object", "time", "view")})})),
    "finish": ("Stop with a visual-fact verdict and citations. Cite original image_id, source-pixel bbox and source time; cost 0. No liability inference.",
               obj({"verdict": {"type": "string", "enum": VERDICTS},
                    "citations": {"type": "array", "items": obj({
                        "image_id": {"type": "string", "minLength": 1}, "bbox": BOX,
                        "time": {"type": "string", "minLength": 1}})}})),
}
SPECS.update({
    "assess_quality": ("Measure resolution, contrast, exposure and edge strength; cost 1. Scores are advisory, not defect/visibility truth.", obj({"image_id": IMAGE})),
    "read_metadata": ("Read submitted source metadata and allowed embedded EXIF; cost 1. Timestamps are not authenticated.", obj({"image_id": IMAGE})),
    "ground_text_to_image": ("Propose regions matching text using a frozen detector; cost 3. No match does not prove absence.", obj({"image_id": IMAGE, "text": {"type": "string", "minLength": 1, "maxLength": 500}})),
    "ground_image_to_image": ("Propose corresponding regions in a target image from a released reference image/view; cost 3. Feature similarity is not identity truth.", obj({"query_image_id": IMAGE, "target_image_id": IMAGE, "top_k": {"type": "integer", "minimum": 1, "maximum": 5}}, ["query_image_id", "target_image_id"])),
    "ground_image_to_text": ("Rank supplied candidate text descriptions against image pixels; cost 2. Cosine scores are not probabilities or verdicts.", obj({"image_id": IMAGE, "candidates": {"type": "array", "minItems": 1, "maxItems": 32, "items": {"type": "string", "minLength": 1, "maxLength": 500}}})),
})
TOOLS = [{"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
         for name, (description, schema) in SPECS.items()]


class ActionError(ValueError):
    """Stable public code; never include private paths or annotations."""
    def __init__(self, code, details=None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get("type"), str) or action.get("type") not in SPECS:
        raise ActionError("unknown_action")
    args = {k: v for k, v in action.items() if k != "type"}
    errors = list(Draft202012Validator(SPECS[action["type"]][1]).iter_errors(args))
    if errors:
        raise ActionError("invalid_arguments", {"validation_errors": [
            {"path": list(e.absolute_path), "rule": e.validator, "message": e.message}
            for e in errors[:3]]})


def tool_schemas(ocr_available=True, grounding_available=(), costs=None):
    result = deepcopy([t for t in TOOLS
                     if (ocr_available or t["function"]["name"] != "ocr")
                     and (not t["function"]["name"].startswith("ground_")
                          or t["function"]["name"] in grounding_available)])
    if costs is not None:
        for tool in result:
            name = tool["function"]["name"]
            tool["function"]["description"] = re.sub(r"cost \d+", "cost " + str(costs.get(name, 0)), tool["function"]["description"])
    return result
