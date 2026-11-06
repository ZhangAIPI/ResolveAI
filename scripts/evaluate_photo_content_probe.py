"""Selected-history photo-content probe; counterfactuals are not benchmark scores.

Run REPO RUNTIME OUTPUT_ROOT --shards 4 --shard N [--limit-families 2].
Resume the same shard without the smoke limit; no completed answer is rerun.
"""
import argparse
import base64
from contextlib import closing
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
from PIL import Image

CONDITIONS = ("real", "gray", "shuffled")
INSTRUCTION = "\nFor this final response, call finish exactly once using the observed evidence. No further tool calls are permitted.\n"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def family_seed(family, seed=20261106):
    return int.from_bytes(hashlib.sha256(f"{seed}|{family}".encode()).digest()[:8], "big")


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def rgb_sha(pixels):
    return digest(canonical(list(pixels.shape)).encode() + pixels.tobytes())


def transformed(pixels, condition, seed):
    if condition == "real":
        return pixels.copy()
    if condition == "gray":
        return np.full_like(pixels, 127)
    if condition == "shuffled":
        flat = pixels.reshape(-1, 3)
        return flat[np.random.default_rng(seed).permutation(len(flat))].reshape(pixels.shape)
    raise ValueError("Unknown pixel condition")


def blob(pixels, directory):
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    value = stream.getvalue()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest(value) + ".png")
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if file_sha(path) != digest(value):
        raise ValueError("Content-addressed blob conflict")
    return path


def image_parts(messages):
    for mi, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            for pi, part in enumerate(message["content"]):
                if part.get("type") == "image":
                    yield mi, pi, part


def selected_prefix(trace, directory):
    """Extract only public history through a real, first-turn provided request."""
    seen = set()
    for i, message in enumerate(trace["messages"]):
        images = list(image_parts([message]))
        fingerprints = [rgb_sha(rgb(directory / p["image_ref"])) for _, _, p in images]
        if message.get("role") == "tool" and message.get("name") == "request_photo":
            fields = [json.loads(x["text"]) for x in message["content"] if x.get("type") == "text"]
            novel = [(pi, p) for (_, pi, p), h in zip(images, fingerprints) if h not in seen]
            if fields[0].get("status") == "provided" and novel:
                prefix = deepcopy(trace["messages"][:i + 1])
                assistants = [m for m in prefix if m["role"] == "assistant"]
                if len(assistants) != 1 or len(novel) != 1:
                    raise ValueError("Expected one first-turn new-photo request")
                call = assistants[0]["tool_calls"][0]["function"]
                if call["name"] != "request_photo":
                    raise ValueError("Prefix does not end in a photo request")
                for _, _, part in image_parts(prefix):
                    path = (directory / part["image_ref"]).resolve()
                    if path.stem != file_sha(path):
                        raise ValueError("Source image blob changed")
                    part["image_ref"] = str(path)
                finish = [t for t in trace["tools"] if t["function"]["name"] == "finish"]
                if len(finish) != 1:
                    raise ValueError("Missing unique finish schema")
                return {"messages": prefix, "tools": deepcopy(finish)}, (i, novel[0][0])
        seen.update(fingerprints)
    raise ValueError("No actual new-image request in source history")


def nonpixel_hash(payload):
    item = deepcopy(payload)
    for _, _, part in image_parts(item["messages"]):
        part.pop("image_ref", None)
        part.pop("image_png", None)
    return digest(canonical(item).encode())


def transport(payload):
    value = deepcopy(payload)
    for _, _, part in image_parts(value["messages"]):
        part["image_png"] = base64.b64encode(Path(part.pop("image_ref")).read_bytes()).decode()
    return value


def encode_audit(processor, payload):
    from resolveai.model_processing import chat_inputs
    messages = deepcopy(payload["messages"])
    for _, _, part in image_parts(messages):
        part["image"] = Image.fromarray(rgb(part.pop("image_ref")))
    inputs = chat_inputs(processor, messages, tools=payload["tools"])
    tensors = {}
    for key, tensor in inputs.items():
        array = tensor.detach().cpu().contiguous().numpy()
        tensors[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": digest(array.tobytes())}
    visual = tensors.pop("pixel_values")
    values = inputs["pixel_values"].detach().cpu().contiguous().numpy()
    offset, blocks = 0, []
    for grid in inputs["image_grid_thw"].tolist():
        count = int(np.prod(grid))
        blocks.append(digest(values[offset:offset + count].tobytes()))
        offset += count
    if offset != len(values):
        raise ValueError("Image grid does not partition pixel tensor")
    return {"nonpixel_tensors": tensors, "pixel_values_shape": visual["shape"],
            "pixel_values_dtype": visual["dtype"], "pixel_values_sha256": visual["sha256"],
            "pixel_blocks_sha256": blocks, "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "input_tokens": int(inputs["input_ids"].shape[1])}


def matched(audits, target):
    ignored = {"pixel_values_sha256", "pixel_blocks_sha256"}
    reference = {k: v for k, v in audits[0].items() if k not in ignored}
    if any({k: v for k, v in x.items() if k not in ignored} != reference for x in audits[1:]):
        raise ValueError("Nonpixel tensors, token grid or shapes differ across conditions")
    fixed = [v for i, v in enumerate(audits[0]["pixel_blocks_sha256"]) if i != target]
    if any([v for i, v in enumerate(x["pixel_blocks_sha256"]) if i != target] != fixed for x in audits[1:]):
        raise ValueError("An initial preview/reference pixel block changed")
    return reference


def prepare(source, directory, seed, processor):
    payload, location = selected_prefix(source["trace"], source["directory"])
    native = deepcopy(payload)
    native["tools"] = source["trace"]["tools"]
    native_audit = encode_audit(processor, native)
    if native_audit["input_tokens"] != source["trace"]["generation"][1]["input_tokens"]:
        raise ValueError("Re-encoded original history differs from frozen source processor")
    payload["messages"][0]["content"] += INSTRUCTION
    payload.update(mode="tools", max_new_tokens=384, tool_adapter="native", assistant_prefix=True)
    slots = [(mi, pi) for mi, pi, _ in image_parts(payload["messages"])]
    target = slots.index(location)
    original = Path(payload["messages"][location[0]]["content"][location[1]]["image_ref"])
    pixels = rgb(original)
    family = source["result"]["family_id"]
    output, audits = {}, []
    for condition in CONDITIONS:
        variant = deepcopy(payload)
        changed = transformed(pixels, condition, family_seed(family, seed))
        path = original if condition == "real" else blob(changed, directory)
        variant["messages"][location[0]]["content"][location[1]]["image_ref"] = str(path)
        audit = encode_audit(processor, variant)
        audit["source_native_input_tokens"] = native_audit["input_tokens"]
        audit["replacement_message_and_part"] = list(location)
        audits.append(audit)
        output[condition] = {"payload": variant, "audit": audit, "rgb_sha256": rgb_sha(changed),
                             "input_sha256": digest(canonical(transport(variant)).encode())}
    matched(audits, target)
    if len({x["pixel_blocks_sha256"][target] for x in audits}) != 3:
        raise ValueError("Interventions coincide after image preprocessing")
    if len({nonpixel_hash(x["payload"]) for x in output.values()}) != 1:
        raise ValueError("Public history, metadata, schema or limits differ")
    if len({x["rgb_sha256"] for x in output.values()}) != 3:
        raise ValueError("Degenerate image intervention")
    return output


def ask_once(client, payload, audit):
    from resolveai.tools import validate_action
    answer = client.ask_conversation(transport(payload), 8192)
    if answer.get("input_tokens") != audit["input_tokens"]:
        raise ValueError("Worker token count differs from exact CPU preflight")
    valid, error = False, None
    if answer.get("halt"):
        error = answer["halt"]
    elif answer.get("parse_error"):
        error = "parse_error"
    else:
        call = answer.get("tool_call", {})
        try:
            if call.get("name") != "finish" or not isinstance(call.get("arguments"), dict):
                raise ValueError("Expected finish")
            if answer["raw"].count("<tool_call>") != 1 or answer["raw"].count("</tool_call>") != 1:
                raise ValueError("Expected exactly one call")
            if "type" in call["arguments"]:
                raise ValueError("Unexpected type argument")
            validate_action({"type": "finish", **call["arguments"]})
            valid = True
        except ValueError:
            error = "invalid_finish"
    return answer, valid, error


def visual_core_compatibility(repo, frozen_commit, expected_digest):
    """Audit every source module; only unrelated paid-API helpers may differ."""
    import subprocess
    root = repo / "src/resolveai"
    old_paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", frozen_commit, "--", "src/resolveai"],
        text=True).splitlines()
    names = sorted(Path(p).name for p in old_paths
                   if str(Path(p).parent) == "src/resolveai" and p.endswith(".py"))
    if names != sorted(p.name for p in root.glob("*.py")):
        raise ValueError("Source module set changed")
    old_digest, changed, modules = hashlib.sha256(), [], {}
    for name in names:
        old = subprocess.check_output(["git", "-C", str(repo), "show",
                                       f"{frozen_commit}:src/resolveai/{name}"])
        old_digest.update(name.encode());old_digest.update(old)
        current = (root / name).read_bytes()
        modules[name] = {"frozen_sha256": digest(old), "probe_sha256": digest(current)}
        if old != current:
            changed.append(name)
    if old_digest.hexdigest() != expected_digest or set(changed) - {"api_budget.py", "api_resume.py"}:
        raise ValueError("Frozen visual inference source differs")
    return {"frozen_core_sha256": expected_digest, "modules": modules,
            "changed_unexecuted_api_helpers": changed,
            "scope": "All visual worker, processor, model-client, tools and environment modules byte-identical; paid API helpers unused."}


def sources(repo, runtime):
    from scripts.evaluate_evidence_controls import model_identity
    from resolveai.benchmark import source_digest
    pool = runtime / "evaluation/cvpr-paper-priority"
    cohort_path = repo / "docs/results/cvpr_20261102_candidate_scores.json"
    cohort = json.loads(cohort_path.read_text())
    data = runtime / "benchmark-v041/cases.json"
    if len(cohort["family_ids"]) != 209 or file_sha(data) != cohort["dataset_sha256"]:
        raise ValueError("Frozen source cohort changed")
    cases = {c["family_id"]: c for c in json.loads(data.read_text())
             if c["family_id"] in cohort["family_ids"] and c["variant"] == "Obtainable"}
    selected, manifests, shared, source_worker = {}, [], None, None
    for shard in range(2):
        directory = pool / f"evidence-controls-v1/qwen3_8b_sftrl/budget3/shard-{shard}"
        manifest = json.loads((directory / "manifest.json").read_text())
        cfg = manifest["configuration"]
        families = cohort["family_ids"][shard::2]
        if (cfg["families"] != families or cfg["cohort_sha256"] != file_sha(cohort_path)
                or cfg["dataset_sha256"] != file_sha(data)
                or (cfg["budget"], cfg["max_turns"], cfg["max_new_tokens"], cfg["max_context_tokens"],
                    cfg["image_max_pixels"]) != (3, 8, 384, 8192, 262144)):
            raise ValueError("Source inference/cohort settings changed")
        identity = {k: cfg[k] for k in ("model", "source_sha256", "storage_script_sha256",
                                       "terminal_script_sha256", "seed")}
        if shared is not None and identity != shared:
            raise ValueError("Source shard identity mismatch")
        shared = identity
        worker = {k: manifest["worker"][k] for k in ("tool_adapter", "processor", "image_processor",
            "max_pixels", "min_pixels", "processor_options", "image_settings", "native_tool_format",
            "model_type", "base_model", "lora_adapter")}
        if source_worker is not None and worker != source_worker:
            raise ValueError("Source worker processor identity differs")
        source_worker = worker
        with closing(sqlite3.connect(f"file:{directory}/episodes.sqlite?mode=ro", uri=True)) as db:
            expected = {(f + "-" + v, a) for f in families for v in cfg["variants"] for a in cfg["arms"]}
            if set(db.execute("SELECT case_id,arm FROM episodes")) != expected:
                raise ValueError("Source evaluation is incomplete or contains unexpected keys")
            for raw, trace_raw in db.execute(
                    "SELECT result,trace FROM episodes WHERE arm='adaptive' ORDER BY case_id"):
                row = json.loads(raw)
                if row["variant"] != "Obtainable" or not row["requests"]:
                    continue
                case = cases[row["family_id"]]
                if row["requests"] != 1 or row["task"] != "state" or row["source_label"] != case["provenance_truth"]:
                    raise ValueError("Unexpected selected source episode")
                trace = json.loads(trace_raw)
                selected_prefix(trace, directory)
                selected[row["family_id"]] = {"result": row, "trace": trace, "directory": directory,
                    "record_sha256": digest((raw + "\n" + trace_raw).encode())}
        manifests.append({"path": str(directory / "manifest.json"), "sha256": file_sha(directory / "manifest.json")})
    if len(selected) != 68:
        raise ValueError("Selected source history count changed")
    compatibility = visual_core_compatibility(repo, manifest["source_commit"], shared["source_sha256"])
    model = shared["model"]
    if model_identity(Path(model["path"]), model["model_id"], model["base_revision"]) != model:
        raise ValueError("SFT+RL checkpoint changed")
    provenance = {"cohort_sha256": file_sha(cohort_path), "dataset_sha256": file_sha(data),
                  "source_manifests": manifests, "model": model, "core_sha256": shared["source_sha256"],
                  "source_worker": source_worker, "probe_core_sha256": source_digest(),
                  "visual_source_compatibility": compatibility,
                  "source_records": {f: x["record_sha256"] for f, x in sorted(selected.items())}}
    return selected, provenance


def summary(db, expected):
    rows = [json.loads(x[0]) for x in db.execute("SELECT result FROM episodes")]
    if {(r["family_id"], r["condition"]) for r in rows} != expected:
        return {"complete": False, "completed": len(rows), "expected": len(expected)}
    groups = {}
    for condition in CONDITIONS:
        entries = [r for r in rows if r["condition"] == condition]
        groups[condition] = {"attempts": len(entries),
            "source_label_matches": sum(r["source_label_match"] for r in entries),
            "schema_valid_finish": sum(r["schema_valid_finish"] for r in entries),
            "abstained": sum(r["verdict"] == "Need more evidence" for r in entries),
            "mean_final_call_s": sum(r["generation"].get("latency_s", 0) for r in entries) / len(entries),
            "mean_input_tokens": sum(r["generation"].get("input_tokens", 0) for r in entries) / len(entries),
            "mean_output_tokens": sum(r["generation"].get("output_tokens", 0) for r in entries) / len(entries)}
    return {"complete": True, "completed": len(rows), "expected": len(expected),
            "scope": "Selected-history counterfactual content probe; not native-agent or benchmark accuracy.",
            "validation": "Finish function and arguments schema only; no independently reviewed sufficiency metrics.",
            "conditions": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--limit-families", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.shards != 4 or not 0 <= args.shard < 4 or (args.limit_families is not None and args.limit_families < 1):
        parser.error("Use four shards and a positive optional smoke limit")
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    selected, provenance = sources(args.repo, args.runtime)
    families = sorted(selected)[args.shard::4]
    directory = args.output / f"shard-{args.shard}"
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (directory / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {"scope": "selected-history-content-probe-v1", "families": families,
              "conditions": CONDITIONS, "shards": 4, "shard": args.shard, "seed": args.seed,
              "source": provenance, "script_sha256": file_sha(__file__), "max_new_tokens": 384,
              "max_context_tokens": 8192, "continuation_calls": 1, "do_sample": False,
              "gray_rgb": [127, 127, 127], "shuffle": "joint RGB pixels; SHA256(seed|family) first 64 bits",
              "instruction": INSTRUCTION}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if canonical(manifest["configuration"]) != canonical(config):
            raise ValueError("Probe configuration changed on resume")
    else:
        manifest = {"configuration": config}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    db = sqlite3.connect(directory / "episodes.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS episodes (family_id TEXT, condition TEXT, result TEXT, trace TEXT, PRIMARY KEY(family_id,condition))")
    db.commit()
    expected = {(f, c) for f in families for c in CONDITIONS}
    completed = set(db.execute("SELECT family_id,condition FROM episodes"))
    if not completed <= expected:
        raise ValueError("Unplanned saved result")
    for family, condition, raw, trace_raw in db.execute("SELECT * FROM episodes"):
        row, saved = json.loads(raw), json.loads(trace_raw)
        for _, _, part in image_parts(saved["messages"]):
            part["image_ref"] = str((directory / part["image_ref"]).resolve())
        if ((row["family_id"], row["condition"]) != (family, condition)
                or row["source_record_sha256"] != selected[family]["record_sha256"]
                or digest(canonical(transport(saved)).encode()) != row["input_sha256"]
                or nonpixel_hash(saved) != row["nonpixel_public_sha256"]):
            raise ValueError("Saved result input/source changed")

    if completed == expected:
        (directory / "summary.json").write_text(json.dumps(summary(db, expected), indent=2) + "\n")
        db.close()
        lock.close()
        print(json.dumps({"complete": True, "resumed_without_inference": True}))
        return
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoProcessor
    from resolveai.model_processing import processor_options, image_settings
    from resolveai.pilot import ModelClient
    torch.set_num_threads(1)
    model_path = provenance["model"]["path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, max_pixels=262144,
                                              min_pixels=2304, **processor_options(model_path))
    pending = [f for f in families if any((f, c) not in completed for c in CONDITIONS)]
    if args.limit_families:
        pending = pending[:args.limit_families]
    prepared = {f: prepare(selected[f], args.output / "images", args.seed, processor) for f in pending}
    audit = {f: {c: {**x["audit"], "input_sha256": x["input_sha256"],
                     "returned_rgb_sha256": x["rgb_sha256"],
                     "nonpixel_public_sha256": nonpixel_hash(x["payload"])} for c, x in trio.items()}
             for f, trio in prepared.items()}
    audit_path = directory / "input-audit.json"
    previous_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if any(previous_audit[f] != audit[f] for f in set(previous_audit) & set(audit)):
        raise ValueError("Stored processor audit differs on resume")
    previous_audit.update(audit)
    audit_path.write_text(json.dumps(previous_audit, indent=2) + "\n")
    print(json.dumps({"preflight": True, "families": len(pending), "matched_conditions": 3}), flush=True)
    if args.validate_only:
        db.close();lock.close()
        return
    client = ModelClient(model_path, 0, "auto", assistant_prefix=True)
    ready = client.ready
    if (any(ready.get(k) != v for k, v in provenance["source_worker"].items())
            or ready["tool_adapter"] != "native" or ready["processor"] != type(processor).__name__
            or ready["max_pixels"] != 262144 or ready["min_pixels"] != 2304
            or ready["image_settings"] != image_settings(processor)):
        client.close()
        raise ValueError("Worker processor/settings differ from preflight")
    manifest["worker"] = ready
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for family in pending:
            for condition in CONDITIONS:
                if (family, condition) in completed:
                    continue
                item = prepared[family][condition]
                answer, valid, error = ask_once(client, item["payload"], item["audit"])
                verdict = answer["tool_call"]["arguments"]["verdict"] if valid else None
                row = {"family_id": family, "condition": condition, "schema_valid_finish": valid,
                       "verdict": verdict, "source_label": selected[family]["result"]["source_label"],
                       "source_label_match": valid and verdict == selected[family]["result"]["source_label"],
                       "error": error, "generation": answer, "family_seed": family_seed(family, args.seed),
                       "source_record_sha256": selected[family]["record_sha256"], **audit[family][condition]}
                saved = deepcopy(item["payload"])
                for _, _, part in image_parts(saved["messages"]):
                    part["image_ref"] = os.path.relpath(part["image_ref"], directory)
                with db:
                    db.execute("INSERT INTO episodes VALUES (?,?,?,?)",
                               (family, condition, json.dumps(row), json.dumps(saved)))
                completed.add((family, condition))
                print(json.dumps({"completed": len(completed), "expected": len(expected),
                                  "family_id": family, "condition": condition, "schema_valid_finish": valid}), flush=True)
    finally:
        client.close()
        db.close()
    with closing(sqlite3.connect(f"file:{directory}/episodes.sqlite?mode=ro", uri=True)) as saved_db:
        result = summary(saved_db, expected)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lock.close()


if __name__ == "__main__":
    main()
