"""Native, resumable two-request evaluation on the procedural multistep suite.

Acquisition policies share public request options, two request slots and a tool
budget. Every policy then receives the same single finish-only continuation.
The oracle alone reads the certified target view and is an upper reference.
These are simulator-grounded results, never independently human-reviewed data.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sqlite3
import time

from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import Conversation, TrajectoryStore, content_for
from resolveai.environment import Environment
from resolveai.pilot import ModelClient
from resolveai.tools import validate_action

POLICIES = ("adaptive_acquisition", "fixed_restoration", "fixed_view_order",
            "random_request", "oracle")
REQUEST_VIEWS = ("original", "front_survey", "rear_left", "rear_center", "rear_right")
ACQUISITION_INSTRUCTION = (
    "\nThis is a two-request evidence-acquisition task. During acquisition only "
    "request_photo is available. Choose exactly one request_photo call per turn, "
    "using the public request_options. You have two request slots, including any "
    "repeated or unhelpful requests. A final-answer phase follows the requests.\n"
)
TERMINAL_INSTRUCTION = (
    "\nAcquisition has ended. For this final response, call finish exactly once "
    "using only the images and observations already supplied. No further requests "
    "or other tools are permitted. Cite released image IDs, source-pixel regions "
    "and times; include identity links where the claim requires them. Use Need "
    "more evidence when the observed evidence does not justify a definite answer.\n"
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class NoOCR:
    available = False


def load_data(path):
    source = path / "cases.json" if path.is_dir() else path
    document = json.loads(source.read_text())
    cases = document["cases"] if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases:
        raise ValueError("A nonempty procedural case list is required")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if not case.get("family_id") or not case.get("multistep_certificate"):
            raise ValueError("Cases must have family IDs and multistep certificates")
        if not case.get("multistep_regions"):
            raise ValueError("Cases must contain certified source-region mappings")
    return source.resolve(), cases


def asset_fingerprint(cases, data_directory):
    assets = {}
    for case in cases:
        root = Path(case.get("asset_root", data_directory)).resolve()
        for item in case["evidence"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing asset or path outside the case asset root")
            name = str(path)
            if name not in assets:
                assets[name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    return assets


def model_fingerprint(model, adapter):
    base = model.resolve()
    if not (base / "config.json").is_file():
        raise ValueError("The local base-model directory needs config.json")
    weights = sorted(base.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("The local base checkpoint has no complete weight shards")
    result = {
        "base_path": str(base),
        "configuration": {p.name: sha_file(p) for p in sorted(base.glob("*.json"))
                          if p.name != "tokenizer.json"},
        "weight_shards": {p.name: p.stat().st_size for p in weights},
        "weight_sha256_recomputed": False,
    }
    if adapter:
        saved = adapter.resolve()
        config_path = saved / "adapter_config.json"
        config = json.loads(config_path.read_text())
        if Path(config["base_model_name_or_path"]).resolve() != base:
            raise ValueError("The LoRA adapter points to a different local base model")
        weight = saved / "adapter_model.safetensors"
        if not weight.is_file():
            raise ValueError("The adapter has no adapter_model.safetensors")
        result["adapter"] = {
            "path": str(saved), "configuration_sha256": sha_file(config_path),
            "weight_sha256": sha_file(weight),
            "processor_configuration": {
                p.name: sha_file(p) for p in sorted(saved.glob("*.json"))
                if p.name not in {"tokenizer.json", "adapter_config.json"}},
        }
    return result


def public_queries(observation):
    options = observation["request_options"]
    objects, times = options.get("objects", []), options.get("times", [])
    views = options.get("views", [])
    if len(objects) != 1 or len(times) != 1 or set(views) != set(REQUEST_VIEWS):
        raise ValueError("The controlled suite requires one public object/time and five request views")
    queries = [{"object": obj, "time": stamp, "view": view}
               for obj, stamp, view in itertools.product(objects, times, REQUEST_VIEWS)]
    return {query["view"]: query for query in queries}


def scheduled_views(case, policy, seed):
    if policy == "fixed_restoration":
        return ["original", "original"]
    if policy == "fixed_view_order":
        return ["front_survey", "rear_left"]
    if policy == "random_request":
        seed_value = int.from_bytes(hashlib.sha256(
            f"{seed}|{case['paired_history_id']}".encode()).digest()[:8], "big")
        return random.Random(seed_value).sample(list(REQUEST_VIEWS), 2)
    if policy == "oracle":
        target = case["multistep_certificate"]["target_rear_view"]
        if target not in REQUEST_VIEWS[2:]:
            raise ValueError("The certified oracle target is not a rear view")
        return ["front_survey", target]
    raise ValueError("Adaptive acquisition has no private or scripted schedule")


def offered_call(client, session, phase, args, generations):
    try:
        answer = client.ask_conversation(
            {**session.public(), "max_new_tokens": args.max_new_tokens},
            args.max_context_tokens)
    except (RuntimeError, OSError, EOFError, json.JSONDecodeError) as error:
        generations.append({"phase": phase, "transport_error": type(error).__name__})
        raise
    generations.append({"phase": phase, **deepcopy(answer)})
    return answer


def parsed_call(answer, name):
    if answer.get("halt"):
        raise ValueError("halt:" + str(answer["halt"]))
    if answer.get("parse_error"):
        raise ValueError("parse_error")
    call = answer.get("tool_call")
    if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
            or call["name"] != name or not isinstance(call["arguments"], dict)
            or "type" in call["arguments"]):
        raise ValueError("invalid_" + name)
    validate_action({"type": name, **call["arguments"]})
    return call


def run_episode(case, data_directory, client, policy, args, observation_token, evaluate):
    started = time.perf_counter()
    env = Environment(case, case.get("asset_root", data_directory), args.budget,
                      ocr_backend=NoOCR())
    world_fingerprint = env._world.fingerprint
    initial = env.observation()
    queries = public_queries(initial)
    session = Conversation(env)
    if session.messages[1]["content"] != content_for(initial):
        raise ValueError("Initial observed images changed during conversation creation")
    all_tools = deepcopy(session.tools)
    session.tools = [t for t in all_tools if t["function"]["name"] == "request_photo"]
    session.messages[0]["content"] += ACQUISITION_INSTRUCTION
    initial_public_sha256 = sha_value(session.public())
    tokens = [observation_token(image, view=image["image_id"]) for image in initial["images"]]
    seen_rgb = {token["rgb_sha256"] for token in tokens}
    new_image_requests, new_pixel_images = 0, 0
    generations, executed = [], []
    selection_error = None
    runtime_error = None
    schedule = None if policy == "adaptive_acquisition" else scheduled_views(case, policy, args.seed)
    try:
        for index in range(2):
            if session.turns >= args.max_turns - 1:
                selection_error = "no_turn_reserved_for_terminal"
                break
            if policy == "adaptive_acquisition":
                answer = offered_call(client, session, f"acquisition-{index + 1}", args, generations)
                try:
                    call = parsed_call(answer, "request_photo")
                    query = call["arguments"]["query"]
                    if query not in queries.values():
                        raise ValueError("request_not_in_public_options")
                except ValueError as error:
                    selection_error = str(error)
                    # Failed selector prose is audit-only as well; it must not
                    # become extra reasoning for the common terminal readout.
                    session.invalid_output("")
                    break
                # Selector prose is retained in raw generation audit, but no
                # policy gets private self-commentary in the shared answer call.
                assistant_text = ""
            else:
                call = {"name": "request_photo",
                        "arguments": {"query": deepcopy(queries[schedule[index]])}}
                assistant_text = ""
            result = session.call(call, assistant_text)
            returned = [observation_token(image, view=image["image_id"]) for image in result.get("images", [])]
            new_rgb = {token["rgb_sha256"] for token in returned} - seen_rgb
            new_image_requests += bool(new_rgb)
            new_pixel_images += len(new_rgb)
            seen_rgb.update(token["rgb_sha256"] for token in returned)
            tokens.extend(returned)
            executed.append({"call": deepcopy(call), "status": result.get("status"),
                             "error": result.get("error"),
                             "image_ids": [image["image_id"] for image in result.get("images", [])]})
            if result.get("error"):
                selection_error = result["error"]
                break
        # No prior final answer is generated or replayed. All policies use this
        # same instruction, canonical finish schema and one terminal model call.
        session.tools = [t for t in all_tools if t["function"]["name"] == "finish"]
        session.messages[0]["content"] += TERMINAL_INSTRUCTION
        terminal_public_sha256 = sha_value(session.public())
        answer = offered_call(client, session, "terminal", args, generations)
        try:
            call = parsed_call(answer, "finish")
        except ValueError:
            session.invalid_output(answer.get("raw", ""))
            session.termination = "invalid_terminal"
        else:
            result = session.call(call, answer.get("text", ""))
            if result.get("error") or not env.finished:
                session.termination = "invalid_terminal"
    except (RuntimeError, OSError, EOFError, json.JSONDecodeError) as error:
        runtime_error = type(error).__name__
        session.termination = "model_runtime_error"
        terminal_public_sha256 = None
    if env._world.fingerprint != world_fingerprint:
        raise ValueError("An episode mutated the hidden evidence world")
    if session.termination is None:
        session.termination = "no_valid_final_answer"
    trace = session.record()
    wall_latency_s = time.perf_counter() - started
    score = evaluate(case, env.decision if env.finished else None, tokens)
    trace.update(
        generation=generations, wall_latency_s=wall_latency_s,
        observation_tokens=tokens, multistep_evaluation=deepcopy(score),
        native_evaluation=deepcopy(trace["evaluation"]),
        scope="Procedural simulator-grounded evaluation; not human-reviewed evidence sufficiency.")
    trace["metadata"].update(policy=policy, privileged_policy=(policy == "oracle"))
    trace["acquisition"] = {
        "executed_requests": executed, "selection_error": selection_error,
        "selection_complete": len(executed) == 2 and selection_error is None,
        "scheduled_prefix": policy != "adaptive_acquisition",
        "public_views": list(REQUEST_VIEWS),
    }
    row = {
        "case_id": case["case_id"], "family_id": case["family_id"], "policy": policy,
        "privileged_policy": policy == "oracle", "termination": session.termination,
        "finished": bool(env.finished), "selection_complete": trace["acquisition"]["selection_complete"],
        "selection_error": selection_error, "runtime_error": runtime_error,
        "verdict": env.decision["verdict"] if env.finished else None,
        "multistep_evaluation": score, "requests": env.requests, "tool_calls": env.calls,
        "tool_cost": env.tool_cost, "request_cost": env.request_cost,
        "new_image_requests": new_image_requests, "new_pixel_images": new_pixel_images,
        "token_accounting_complete": all("transport_error" not in g for g in generations),
        "assistant_turns": session.turns, "model_calls": len(generations),
        "input_tokens": sum(g.get("input_tokens", 0) for g in generations),
        "output_tokens": sum(g.get("output_tokens", 0) for g in generations),
        "wall_latency_s": wall_latency_s, "initial_public_sha256": initial_public_sha256,
        "terminal_public_sha256": terminal_public_sha256,
    }
    return row, trace


def save_record(db, store, row, trace):
    saved = store.write(trace)
    row["trace_sha256"] = sha_value(saved)
    with db:
        db.execute("INSERT INTO episodes VALUES (?,?,?,?)",
                   (row["case_id"], row["policy"], canonical(row), canonical(saved)))
    store.output.unlink(missing_ok=True)


def validate_saved(db, expected, cases, output, evaluate):
    existing, initial_hashes = set(), {}
    for case_id, policy, raw, trace_raw in db.execute("SELECT * FROM episodes"):
        key = case_id, policy
        row, trace = json.loads(raw), json.loads(trace_raw)
        if key not in expected or (row["case_id"], row["policy"]) != key:
            raise ValueError("Unexpected saved episode")
        if sha_value(trace) != row["trace_sha256"]:
            raise ValueError("A saved trace changed")
        for message in trace["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part.get("type") != "image":
                    continue
                path = (output / part["image_ref"]).resolve()
                if not path.is_relative_to(output.resolve()) or path.stem != sha_file(path):
                    raise ValueError("A saved image blob changed")
        score = evaluate(cases[case_id], trace.get("decision"), trace["observation_tokens"])
        if canonical(score) != canonical(row["multistep_evaluation"]):
            raise ValueError("Saved procedural scoring no longer agrees with the frozen input")
        old = initial_hashes.setdefault(case_id, row["initial_public_sha256"])
        if row["initial_public_sha256"] != old:
            raise ValueError("Policies did not receive the same initial public input")
        existing.add(key)
    return existing, initial_hashes


def summarize(db, cases, policies):
    rows = [json.loads(row[0]) for row in db.execute("SELECT result FROM episodes")]
    planned = len(cases) * len(policies)
    complete = len(rows) == planned
    groups = {}
    for policy in policies:
        entries = [row for row in rows if row["policy"] == policy]
        counts = {
            "attempted": len(entries),
            "source_label_correct": sum(bool(r["multistep_evaluation"]["source_label_correct"]) for r in entries),
            "simulator_grounded_correct": sum(bool(r["multistep_evaluation"]["simulator_grounded_correct"]) for r in entries),
            "chain_grounded_correct": sum(bool(r["multistep_evaluation"]["chain_grounded_correct"]) for r in entries),
            "localized_grounded_correct": sum(bool(r["multistep_evaluation"]["localized_grounded_correct"]) for r in entries),
            "evidence_sufficient": sum(bool(r["multistep_evaluation"]["procedural_evidence_sufficient"]) for r in entries),
            "finished": sum(r["finished"] for r in entries),
            "selection_complete": sum(r["selection_complete"] for r in entries),
            "unsupported_decisions": sum(bool(r["multistep_evaluation"]["unsupported_decision"]) for r in entries),
        }
        group = {"privileged_reference": policy == "oracle", "counts": counts}
        if complete:
            n = len(entries)
            group["rates_pct"] = {key: 100 * value / n for key, value in counts.items()
                                  if key != "attempted"}
            group["mean"] = {key: sum(r[key] for r in entries) / n for key in
                             ("requests", "new_image_requests", "new_pixel_images",
                              "tool_calls", "tool_cost", "request_cost",
                              "model_calls", "input_tokens", "output_tokens", "wall_latency_s")}
            if not all(r["token_accounting_complete"] for r in entries):
                for key in ("input_tokens", "output_tokens"):
                    group["mean"][key] = None
                group["token_accounting_complete"] = False
        groups[policy] = group
    return {
        "complete": complete, "episodes": len(rows), "expected_episodes": planned,
        "scope": "Procedural simulator-grounded two-request suite, separate from native public-photo benchmark.",
        "metric_scope": "procedural", "independent_human_review": False,
        "denominator": "Every attempted episode, including invalid selection and terminal outputs.",
        "partial_rates_withheld": not complete, "policies": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--variants", nargs="+", choices=("Obtainable",), default=["Obtainable"],
                        help="This controlled two-request comparison supports Obtainable only.")
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Process at most this many pending families; remove the limit to resume.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Load the local model after dataset freeze approval.")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run inside an existing GPU-node allocation")
    if (args.budget < 6 or args.max_turns < 3 or args.max_new_tokens < 1
            or args.max_context_tokens < 1 or args.shards < 1
            or not 0 <= args.shard < args.shards
            or (args.limit is not None and args.limit < 1)
            or len(set(args.policies)) != len(args.policies)):
        parser.error("Invalid budget, turn/token limit, shard or duplicate policy")
    from resolveai.multistep import observation_token, evaluate
    source, all_cases = load_data(args.data)
    eligible = [case for case in all_cases if case.get("variant", "Obtainable") in args.variants]
    families = sorted({case["family_id"] for case in eligible})[args.shard::args.shards]
    selected = sorted([case for case in eligible if case["family_id"] in families],
                      key=lambda case: (case["family_id"], case["case_id"]))
    if not selected:
        parser.error("This shard has no cases")
    indexed = {case["case_id"]: case for case in selected}
    expected = {(case["case_id"], policy) for case in selected for policy in args.policies}
    config = {
        "experiment": "procedural-multistep-two-request-v1",
        "dataset": str(source), "dataset_sha256": sha_file(source),
        "assets": asset_fingerprint(all_cases, source.parent),
        "model": model_fingerprint(args.model, args.adapter),
        "model_id": args.model_id, "revision": args.revision,
        "source_sha256": source_digest(), "script_sha256": sha_file(__file__),
        "families": families, "case_ids": [case["case_id"] for case in selected],
        "policies": args.policies, "variants": args.variants,
        "shards": args.shards, "shard": args.shard,
        "budget": args.budget, "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens, "max_context_tokens": args.max_context_tokens,
        "seed": args.seed, "do_sample": False, "request_slots": 2,
        "request_views": list(REQUEST_VIEWS),
        "random_seed_group": "paired_history_id",
        "random_rule": "SHA256(seed|paired_history_id) first 64 bits; uniform ordered sample of two distinct public views",
        "acquisition_instruction": ACQUISITION_INSTRUCTION,
        "terminal_instruction": TERMINAL_INSTRUCTION,
        "terminal_calls": 1, "oracle_privileged": True,
        "acquisition_assistant_text": "withheld-for-all-policies",
        "accounting": "Scheduled actions count as tool calls but generate no model tokens; adaptive selector calls remain in total compute.",
        "input_tokens": "Offered native input tokens, including context-limit rejections.",
        "scope": "Procedural certificates only; not human-reviewed public-photograph ground truth.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = locked_manifest(args.output / "manifest.json", config)
        with sqlite3.connect(args.output / "episodes.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (case_id TEXT, policy TEXT, result TEXT, trace TEXT, PRIMARY KEY(case_id,policy))")
            db.commit()
            completed, initial_hashes = validate_saved(db, expected, indexed, args.output, evaluate)
            pending = [f for f in families if any(
                (c["case_id"], p) not in completed for c in selected
                if c["family_id"] == f for p in args.policies)]
            if args.limit is not None:
                pending = pending[:args.limit]
            # CPU preflight uses only actually initial public images. It cannot
            # invent observations from the world bank or private certificate.
            for case in selected:
                if case["family_id"] not in pending:
                    continue
                env = Environment(case, case.get("asset_root", source.parent), args.budget,
                                  ocr_backend=NoOCR())
                observation = env.observation()
                public_queries(observation)
                tokens = [observation_token(image, view=image["image_id"]) for image in observation["images"]]
                initial_score = evaluate(case, None, tokens)
                if (not initial_score.get("certificate_valid")
                        or initial_score.get("minimum_request_count") != 2
                        or initial_score.get("procedural_evidence_sufficient")):
                    raise ValueError("Case is not a certified initially unresolved two-request task")
                for policy in args.policies:
                    if policy != "adaptive_acquisition":
                        scheduled_views(case, policy, args.seed)
            if not args.execute or not pending:
                result = summarize(db, selected, args.policies)
                result.update(preflight=True, model_loaded=False)
                (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"preflight": True, "model_loaded": False,
                                  "complete": result["complete"], "pending_families": len(pending)}))
                return
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            path = args.adapter or args.model
            client = ModelClient(path, 0, "auto", assistant_prefix=True)
            manifest["worker"] = client.ready
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            store = TrajectoryStore(args.output / "pending-trace.jsonl")
            try:
                for case in selected:
                    if case["family_id"] not in pending:
                        continue
                    for policy in args.policies:
                        if (case["case_id"], policy) in completed:
                            continue
                        row, trace = run_episode(case, source.parent, client, policy, args,
                                                 observation_token, evaluate)
                        previous = initial_hashes.setdefault(case["case_id"], row["initial_public_sha256"])
                        if previous != row["initial_public_sha256"]:
                            raise ValueError("Initial public input differs across policies")
                        save_record(db, store, row, trace)
                        completed.add((case["case_id"], policy))
                        print(json.dumps({"completed": len(completed), "expected": len(expected),
                                          "case_id": case["case_id"], "policy": policy,
                                          "finished": row["finished"]}), flush=True)
                        if row["runtime_error"]:
                            raise RuntimeError("Model worker failed; the attempted episode was retained")
            finally:
                client.close()
                result = summarize(db, selected, args.policies)
                (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
