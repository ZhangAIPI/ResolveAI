"""Persistent native assistant/tool conversations; evaluator state stays private."""
import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .tools import ActionError, tool_schemas

SYSTEM = """Verify visual facts from released images using the available tools.
Call exactly one function per turn. You may inspect, crop, zoom, read text,
compare images, or request a specific object/time/view. Use real image pixels;
never infer observations from source identity or tool failure. Zoom/crop cannot
recover occluded details or create independent evidence. OCR is fallible text
recognition. Requests do not change the world. Costs and remaining budget are
shown in tool responses. Retain previous observations; finish when evidence is
sufficient or more useful material cannot be obtained. Cite released image_id,
source-pixel rectangles and the source time, not source_id or derived view_id.
source_id records provenance and may name an unreleased original. A preview is
referenced by its own image_id. Its visible pixels may suffice for a judgment,
but never claim details that require an unseen original. Need more evidence may use an empty citations list. For a
conjunction, support all subclaims; one decisive counterexample can refute it.
For cross-image identity or time claims, provide explicit finish links between
original source regions (same_object/different_object/earlier_than/same_time).
A similarity tool proposes matches; it does not establish object identity.
Use Need more evidence when uncertain. State changes do not establish liability.
Tools never mutate the original or earlier views. Crop/zoom return a NEW view_id;
use that view_id to operate on the returned pixels. Crop boxes use display_size,
not source_size or normalized coordinates. Request choices are in request_options.
Choose informative actions rather than trying every tool or repeating the same
observation; stop when evidence is sufficient or useful requests are exhausted.
Always finish via the finish tool; do not return an unstructured final answer.
"""


def content_for(result):
    """Wire-format text and pixels; only tool-visible fields are serialized."""
    metadata = {k: v for k, v in result.items() if k != "images"}
    content = [{"type": "text", "text": json.dumps(metadata)}]
    for image in result.get("images", []):
        content.append({"type": "text", "text": json.dumps({k: v for k, v in image.items() if k != "image_png"})})
        content.append({"type": "image", "image_png": base64.b64encode(image["image_png"]).decode("ascii")})
    return content


class Conversation:
    def __init__(self, environment):
        self._env = environment
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content_for(environment.observation())}]
        self.tools = tool_schemas(environment._ocr.available, environment.grounding_available, asdict(environment.costs))
        self.errors = 0
        self.turns = 0
        self.termination = None
        self.branch_origin = None
        self.prefix_length = 0

    def public(self):
        return {"mode": "tools", "messages": deepcopy(self.messages), "tools": deepcopy(self.tools)}

    def fork(self):
        """Trusted trainer operation, not a policy-visible tool."""
        branch = object.__new__(type(self))
        branch.__dict__ = {k: deepcopy(v) for k, v in self.__dict__.items() if k != "_env"}
        branch._env = self._env.fork()
        state = {"messages": self.messages, "budget": self._env.budget,
                 "images": sorted(self._env._released)}
        branch.branch_origin = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        branch.prefix_length = len(self.messages)
        return branch

    def call(self, call, text=""):
        """Record the actual native call and actual tool response, including errors."""
        if self._env.finished:
            raise ActionError("episode_finished")
        function = {"name": call.get("name"), "arguments": call.get("arguments")}
        call_id = f"call-{self.turns:04d}"
        self.messages.append({"role": "assistant", "content": text,
                              "tool_calls": [{"id": call_id, "type": "function", "function": deepcopy(function)}]})
        self.turns += 1
        try:
            if function["name"] not in {t["function"]["name"] for t in self.tools}:
                raise ActionError("tool_not_available")
            if not isinstance(function["arguments"], dict) or "type" in function["arguments"]:
                raise ActionError("invalid_arguments")
            result = self._env.step({**function["arguments"], "type": function["name"]})
        except ActionError as error:
            self.errors += 1
            result = {"error": error.code, "constraints": deepcopy(error.details),
                "available_tools": [t["function"]["name"] for t in self.tools],
                "budget": self._env.budget, "finished": self._env.finished}
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            self.errors += 1
            result = {"error": "tool_backend_error", "budget": self._env.budget, "finished": self._env.finished}
        self.messages.append({"role": "tool", "name": function["name"],
                              "tool_call_id": call_id, "content": content_for(result)})
        if self._env.finished:
            self.termination = "finished"
        return result

    def invalid_output(self, raw):
        self.turns += 1
        self.errors += 1
        self.messages.extend([{"role": "assistant", "content": raw},
            {"role": "user", "content": ("Invalid tool format. Required: <tool_call>{\"name\":\"REGISTERED_NAME\","
                "\"arguments\":{}}</tool_call>. Supply the actual schema arguments. Do not use cost, tool or function keys. "
                + "Available names: " + ", ".join(t["function"]["name"] for t in self.tools)
                + f". Remaining budget: {self._env.budget}.")}])

    def record(self):
        """Trainer-only record: separate public prompt from terminal supervision."""
        return {"schema_version": "resolveai-trajectory-v1", "messages": deepcopy(self.messages),
            "tools": deepcopy(self.tools), "branch_origin": self.branch_origin,
            "prefix_length": self.prefix_length,
            "metadata": {"case_id": self._env._case.get("case_id"),
                "family_id": self._env._case.get("family_id"),
                "world_fingerprint": self._env._world.fingerprint,
                "annotation_status": self._env._case.get("annotation", {}).get("status", "unreviewed"),
                "termination": self.termination, "turns": self.turns, "errors": self.errors},
            "decision": deepcopy(self._env.decision),
            "evaluation": self._env.evaluate(self._env.decision) if self._env.finished else None}


class TrajectoryStore:
    """One JSONL trace file and content-addressed image blobs, shared across branches."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.assets = self.output.parent / "images"
        self.assets.mkdir(exist_ok=True)

    def write(self, record):
        record = deepcopy(record)
        for message in record["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for item in message["content"]:
                if "image_png" in item:
                    pixels = base64.b64decode(item.pop("image_png"), validate=True)
                    digest = hashlib.sha256(pixels).hexdigest()
                    path = self.assets / (digest + ".png")
                    if not path.exists():
                        path.write_bytes(pixels)
                    item["image_ref"] = "images/" + path.name
        with self.output.open("a") as out:
            out.write(json.dumps(record) + "\n")
        return record


def sft_example(record, allow_proxy=False):
    """Select correct completed traces; never include private evaluation in prompts."""
    evaluation = record.get("evaluation")
    if (record["metadata"]["termination"] != "finished" or not evaluation
            or not evaluation["grounded_correct"] or record["metadata"]["errors"]):
        return None
    if not allow_proxy and (evaluation["grounding_protocol"] not in {"region-time-v1", "evidence-chain-v1"}
                            or record["metadata"]["annotation_status"] != "reviewed"):
        return None
    return {"messages": deepcopy(record["messages"]), "tools": deepcopy(record["tools"]),
            "assistant_loss_only": True}


def preference_pair(left, right, allow_proxy=False):
    """Compare completed sibling outcomes; supervise only their first differing action."""
    if not left["branch_origin"] or left["branch_origin"] != right["branch_origin"]:
        raise ValueError("preferences require sibling branches from the same state")
    if left["metadata"]["world_fingerprint"] != right["metadata"]["world_fingerprint"]:
        raise ValueError("branches belong to different worlds")
    if not allow_proxy and any(not r.get("evaluation")
            or r["evaluation"]["grounding_protocol"] not in {"region-time-v1", "evidence-chain-v1"}
            or r["metadata"]["annotation_status"] != "reviewed" for r in (left, right)):
        return None
    n = left["prefix_length"]
    if n != right["prefix_length"] or left["messages"][:n] != right["messages"][:n] or left["tools"] != right["tools"]:
        raise ValueError("branch prompts differ")
    if any(r["metadata"]["termination"] != "finished" or not r["evaluation"] for r in (left, right)):
        return None
    def score(r):
        e = r["evaluation"]
        return (e["grounded_correct"], not e["unsupported_decision"], -e["request_cost"], -e["tool_cost"])
    if score(left) == score(right):
        return None
    chosen, rejected = (left, right) if score(left) > score(right) else (right, left)
    if not chosen["evaluation"]["grounded_correct"]:
        return None
    # Do not reward hindsight about a provider failure hidden at the common prefix.
    if (chosen["decision"]["verdict"] == rejected["decision"]["verdict"] == "Need more evidence"
            and chosen["evaluation"]["grounded_correct"] == rejected["evaluation"]["grounded_correct"]):
        observed_failure = any(m["role"] == "tool" and isinstance(m.get("content"),list)
            and any(item.get("type")=="text" and '"status": "unable_to_provide"' in item.get("text","")
                    for item in m["content"]) for m in chosen["messages"][:n])
        if not observed_failure:
            return None
    a, b = chosen["messages"][n], rejected["messages"][n]
    if a == b or a["role"] != "assistant" or b["role"] != "assistant":
        return None
    return {"prompt": deepcopy(left["messages"][:n]), "tools": deepcopy(left["tools"]),
            "chosen": deepcopy(a), "rejected": deepcopy(b)}
