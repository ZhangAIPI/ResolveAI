"""Check image-role binding and ordering on visually obvious development controls.

Only public pixels/roles enter the model. Responses are retained verbatim, not
used as truth labels or training observations.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from resolveai.benchmark import source_digest
from resolveai.conversation import content_for
from resolveai.diagnostic import text_direct
from resolveai.environment import Environment
from resolveai.pilot import ModelClient
from check_repaired_benchmark import select


class Observed:
    def __init__(self, observation):
        self.value = observation

    def observation(self):
        return deepcopy(self.value)


def plain(images, captions, question):
    content = []
    for image, caption in zip(images, captions):
        content.extend([{"type": "text", "text": caption},
                        {"type": "image", "image_png": __import__("base64").b64encode(image["image_png"]).decode()}])
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run on the allocated GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((args.data / "cases.json").read_text())
    bottle = next(c for c in cases if c["variant"] == "Obtainable" and c["task"] == "state"
                  and c["provenance"]["sample_id"] == "6621d76a324f6e05d5838d62")
    identity = next(c for c in select(cases) if c["task"] == "identity" and c["category"] == "bench"
                    and c["provenance"]["construction"] == "easy-negative")
    client = ModelClient(args.model, tool_adapter="auto", assistant_prefix=False)
    manifest = {"protocol": "image-binding-check-v1", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5", "worker": client.ready,
                "source_sha256": source_digest(), "data_sha256": hashlib.sha256((args.data / "cases.json").read_bytes()).hexdigest(),
                "families": [bottle["family_id"], identity["family_id"]],
                "scope": "qualitative input-role controls; no reviewed benchmark score"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    def save(kind, case, response):
        row = {"kind": kind, "family_id": case["family_id"], "response": response}
        rows.append(row)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    try:
        for case in [identity, bottle]:
            env = Environment(case, case["asset_root"], 12)
            target = "original" if case["task"] == "state" else next(
                e["id"] for e in case["evidence"] if e["object"] == "subject-B" and e["view"] == "view-02")
            row = next(e for e in case["evidence"] if e["id"] == target)
            result = env.step({"type": "request_photo", "query": {k: row[k] for k in ["object", "time", "view"]}})
            if result["status"] != "provided":
                raise RuntimeError("Fixed retrieval failed")
            obs = env.observation()
            for kind, pictures in [("fifo_metadata", obs["images"]),
                                   ("sorted_metadata", sorted(obs["images"], key=lambda p: p["image_id"]))]:
                record = text_direct(Observed({**obs, "images": pictures}), case, client)
                save(case["task"] + "_" + kind, case, {"raw": record["messages"][-1]["content"],
                                                     "verdict": record["decision"], "termination": record["termination"]})
            if case["task"] == "identity":
                a = next(p for p in obs["images"] if p["object"] == "subject-A")
                b = next(p for p in obs["images"] if p["image_id"] == target)
                for letter, picture in [("A", a), ("B", b)]:
                    response = client.ask_conversation({"mode": "text", "messages": plain([picture], [f"Image {letter}."],
                        "Name and describe the main foreground object in one short sentence."), "max_new_tokens": 96})
                    save("identity_single_" + letter, case, response)
                for name, images, captions in [
                    ("ab", [a, b], ["Image A.", "Image B."]),
                    ("ba", [b, a], ["Image B.", "Image A."]),
                ]:
                    response = client.ask_conversation({"mode": "text", "messages": plain(images, captions,
                        "Name the foreground object in A and the foreground object in B. Are they the same kind of object?"), "max_new_tokens": 128})
                    save("identity_kind_" + name, case, response)
                response = client.ask_conversation({"mode": "text", "messages": plain([a, b], ["Target A.", "Target B."],
                    "Claim: A and B are the same physical item. Reply exactly Supported if supported, "
                    "Refuted if visibly contradicted, or Need more evidence if it cannot be judged."), "max_new_tokens": 96})
                save("identity_plain_verdict", case, response)
            else:
                reference = next(p for p in obs["images"] if p["image_id"] == "reference")
                query = next(p for p in obs["images"] if p["image_id"] == "original")
                for name, images, captions in [
                    ("reference_first", [reference, query], ["Normal reference, a separate specimen.", "Target bottle."]),
                    ("target_first", [query, reference], ["Target bottle.", "Normal reference, a separate specimen."]),
                    ("single_target", [query], ["Target bottle."]),
                ]:
                    response = client.ask_conversation({"mode": "text", "messages": plain(images, captions,
                        "Claim: The target bottle has a chipped or broken rim. Reply exactly Supported if a chip is visible, "
                        "Refuted if the rim is clearly intact, or Need more evidence if it cannot be judged."), "max_new_tokens": 96})
                    save("state_plain_" + name, case, response)
    finally:
        client.close()
    (args.output / "summary.json").write_text(json.dumps({"complete": True, "records": len(rows), "manifest": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
