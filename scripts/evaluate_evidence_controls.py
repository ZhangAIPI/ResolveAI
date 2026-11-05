"""Compare public evidence-request schedules with a frozen terminal answerer.

Every scheduled request runs through the real environment. The model receives the
resulting native conversation and the same finish schema in every arm. Fixed
schedules and a native adaptive arm share the same environment and allowances.
"""
import argparse
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import random
import time

from scripts.evaluate_controls import OutputBudget, TERMINAL_SYSTEM
from scripts.evaluate_study import Records, normalize_agent_record
from resolveai.action_menu import public_state
from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding
from resolveai.ocr import RapidOCRBackend
from resolveai.pilot import ModelClient
from resolveai.rollout import run

ARMS = ("adaptive", "finish_only", "fixed_original", "fixed_order", "random_order")
VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_schedule(public, arm, seed, case_id):
    """Use only the request options released to the policy, never annotations."""
    if arm not in ARMS:
        raise ValueError("Unknown control arm")
    if arm in ("adaptive", "finish_only"):
        return []
    _, state = public_state(public)
    options = state.get("request_options", {})
    dimensions = []
    for key in ("objects", "times", "views"):
        values = options.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("Public request options must be string lists")
        dimensions.append(list(dict.fromkeys(values)))
    objects, times, views = dimensions
    if arm == "fixed_original":
        views = ["original"] if "original" in views else []
    calls = [{"name": "request_photo",
              "arguments": {"query": {"object": obj, "time": stamp, "view": view}}}
             for obj, stamp, view in product(objects, times, views)]
    if arm == "random_order":
        # Access-version suffixes must not change the schedule of the same world.
        key = case_id
        for variant in VARIANTS:
            key = key.removesuffix("-" + variant)
        digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
        random.Random(int.from_bytes(digest, "big")).shuffle(calls)
    return calls


def controlled_episode(env, client, arm, *, seed=20261105, case_id="",
                       max_turns=8, tokens=384):
    """Charge scheduled calls to the same turn, cost and wall-time allowance."""
    if max_turns < 1 or tokens < 1:
        raise ValueError("Positive turn and token allowances are required")
    session = Conversation(env)
    if arm != "adaptive":
        session.messages[0]["content"] = (
            TERMINAL_SYSTEM
            + f"\nEpisode limit: {max_turns} assistant turns, including any recorded "
              "request calls; finish within this limit.\n"
        )
    started = time.perf_counter()
    schedule = request_schedule(session.public(), arm, seed, case_id)
    executed = []
    for call in schedule:
        # Always reserve at least one model turn for the terminal answer.
        if session.turns >= max_turns - 1:
            break
        _, state = public_state(session.public())
        if state.get("budget", 0) < state.get("tool_costs", {}).get("request_photo", 3):
            break
        result = session.call(call)
        executed.append({"call": call, "status": result.get("status"),
                         "image_ids": [x["image_id"] for x in result.get("images", [])]})
    # Past requests remain in history. Only the terminal action is offered now;
    # all arms therefore share the exact same answer schema and output cap.
    if arm != "adaptive":
        session.tools = [t for t in session.tools if t["function"]["name"] == "finish"]
    trace = normalize_agent_record(
        run(session, OutputBudget(client, tokens), max_turns - session.turns,
            add_turn_limit=(arm == "adaptive")), env
    )
    trace["wall_latency_s"] = time.perf_counter() - started
    trace["scripted_prefix"] = {
        "arm": arm, "seed": seed, "scheduled_calls": len(schedule),
        "executed_calls": executed, "assistant_turns": len(executed),
        "policy_inputs": ("native model public conversation" if arm == "adaptive"
                          else "public request_options only"),
        "model_generated_prefix_tokens": 0,
        "skip_reason": ("original_view_not_offered"
                        if arm == "fixed_original" and not schedule else None),
    }
    return trace


def select_cases(plan, cohort, payload, shards, shard, variants):
    """Keep the frozen primary cohort and all requested access versions."""
    if hashlib.sha256(payload).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    if cohort["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("Cohort belongs to another dataset")
    families = cohort["family_ids"]
    if len(set(families)) != len(families) or not set(families) <= set(plan["family_ids"]):
        raise ValueError("Cohort families differ from the frozen plan")
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    selected = families[shard::shards]
    cases = [c for c in json.loads(payload)
             if c["family_id"] in selected and c["variant"] in variants]
    keys = [(c["family_id"], c["variant"]) for c in cases]
    expected = {(f, v) for f in selected for v in variants}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Missing or duplicate family/access-version cases")
    if any(c.get("provenance_truth") not in ("Supported", "Refuted") for c in cases):
        raise ValueError("Primary controls require known provisional source labels")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return selected, sorted(cases, key=lambda c: c["case_id"])


def validate_records(db, cases, arms, *, complete=False):
    """Never resume into another cohort or silently accept unexpected rows."""
    by_id = {c["case_id"]: c for c in cases}
    expected = {(c["case_id"], arm) for c in cases for arm in arms}
    actual = set()
    for cid, arm, raw, trace_raw in db.execute("SELECT * FROM episodes"):
        row, trace = json.loads(raw), json.loads(trace_raw)
        if (cid, arm) not in expected:
            raise ValueError("Saved result lies outside this control configuration")
        case = by_id[cid]
        if (row["case_id"], row["arm"], row["family_id"], row["variant"]) != (
                cid, arm, case["family_id"], case["variant"]):
            raise ValueError("Saved result identity differs")
        if row["source_label"] != case["provenance_truth"]:
            raise ValueError("Saved source label differs")
        if trace["metadata"]["case_id"] != cid or trace["metadata"]["arm"] != arm:
            raise ValueError("Saved trace identity differs")
        actual.add((cid, arm))
    if complete and actual != expected:
        raise ValueError("Control evaluation is incomplete")
    return actual, expected


def model_identity(path, model_id, revision):
    result = {"path": str(path.resolve()), "model_id": model_id,
              "base_revision": revision}
    config = path / "adapter_config.json"
    if config.exists():
        weights = path / "adapter_model.safetensors"
        result.update(adapter_sha256=sha256(weights), adapter_config_sha256=sha256(config),
                      adapter_base_path=json.loads(config.read_text())["base_model_name_or_path"])
    elif (path / "config.json").exists():
        result["config_sha256"] = sha256(path / "config.json")
    else:
        raise ValueError("No local model or adapter configuration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("model", type=Path, help="Pinned local base model or LoRA directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path, help="Override model with this saved LoRA adapter")
    parser.add_argument("--grounding-root", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run on an allocated GPU node")
    if len(set(args.arms)) != len(args.arms) or len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate arms or access versions")
    if args.model_spec:
        spec = json.loads(args.model_spec.read_text())
        model_id, revision = spec["model_id"], spec["revision"]
        if args.model_id not in (None, model_id) or args.revision not in (None, revision):
            parser.error("Model metadata differs from the supplied model spec")
    else:
        model_id, revision = args.model_id, args.revision
    if "adaptive" in args.arms and args.grounding_root is None:
        parser.error("--grounding-root is required for the full-tool adaptive arm")
    if not model_id or not revision:
        parser.error("Provide --model-spec or --model-id and --revision")
    path = args.adapter or args.model
    identity = model_identity(path, model_id, revision)
    plan = json.loads(args.plan.read_text())
    cohort = json.loads(args.cohort.read_text())
    payload = (Path(plan["data"]) / "cases.json").read_bytes()
    families, cases = select_cases(
        plan, cohort, payload, args.shards, args.shard, args.variants)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "experiment": "public-request-schedules-v1",
        "plan_sha256": sha256(args.plan), "dataset_sha256": plan["dataset_sha256"],
        "cohort_sha256": sha256(args.cohort), "source_sha256": source_digest(),
        "script_sha256": sha256(__file__),
        "storage_script_sha256": sha256(Path(__file__).with_name("evaluate_study.py")),
        "terminal_script_sha256": sha256(Path(__file__).with_name("evaluate_controls.py")),
        "model": identity,
        "model_spec_sha256": sha256(args.model_spec) if args.model_spec else None,
        "families": families, "cases": len(cases), "arms": args.arms,
        "variants": args.variants, "shards": args.shards, "shard": args.shard,
        "expected_episodes": len(cases) * len(args.arms), "seed": args.seed,
        "random_seed_rule": "SHA256(seed|family_id); identical order across access versions",
        "budget": plan["budget"], "max_turns": plan["max_turns"],
        "max_new_tokens": 384, "max_context_tokens": 8192,
        "image_max_pixels": 512 * 512, "tool_adapter": "auto", "assistant_prefix": True,
        "answer_schema": "same native finish schema; adaptive retains all intermediate tools",
        "grounding_manifest": (json.loads((args.grounding_root / "manifest.json").read_text())
                               if args.grounding_root else None),
        "fixed_original_scope": "Only requests literal original view when publicly offered",
        "prefix_accounting": "Actual calls count toward shared turns, costs and latency",
        "token_accounting": "Generated answer tokens; scheduled calls cost zero generated tokens, but remain in model input",
        "scope": "Unreviewed source-label and behavior controls, not grounded accuracy",
    }
    manifest = locked_manifest(args.output / "manifest.json", configuration)
    records = Records(args.output, arms=args.arms)
    validate_records(records.db, cases, args.arms)
    expected_count = configuration["expected_episodes"]
    if records.summary()["episodes"] == expected_count:
        validate_records(records.db, cases, args.arms, complete=True)
        summary = records.summary()
        summary.update(complete=True, expected=expected_count, scope=configuration["scope"])
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        records.db.close()
        print(json.dumps({"complete": True, "episodes": expected_count,
                          "resumed_without_inference": True}), flush=True)
        return
    # Reuse the OCR capability flag without repeatedly loading its backend.
    ocr = RapidOCRBackend()
    backend = (FrozenGrounding(args.grounding_root, device="cuda:0")
               if args.grounding_root else None)
    client = ModelClient(path, 0, "auto", assistant_prefix=True)
    manifest["worker"] = client.ready
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verified_complete = False
    try:
        for case in cases:
            for arm in args.arms:
                if records.has(case["case_id"], arm):
                    continue
                env = Environment(case, case["asset_root"], plan["budget"],
                                  ocr_backend=ocr, grounding_backend=backend)
                fingerprint = env._world.fingerprint
                trace = controlled_episode(
                    env, client, arm, seed=args.seed, case_id=case["case_id"],
                    max_turns=plan["max_turns"])
                if env._world.fingerprint != fingerprint:
                    raise RuntimeError("Evidence world changed")
                trace.update(evaluation=None, schema_version="resolveai-diagnostic-v1")
                trace["metadata"].update(
                    family_id=case["family_id"], case_id=case["case_id"], arm=arm,
                    official_split=case["official_split"],
                    annotation_status="unreviewed-development")
                verdict = trace["decision"]["verdict"] if trace.get("decision") else None
                row = {
                    "case_id": case["case_id"], "family_id": case["family_id"],
                    "task": case["task"], "variant": case["variant"], "arm": arm,
                    "verdict": verdict, "termination": trace["termination"],
                    "errors": trace["errors"], "requests": env.requests,
                    "tool_calls": env.calls, "released": sorted(env._released),
                    "source_label": case["provenance_truth"],
                    "source_label_match": verdict == case["provenance_truth"],
                    "tool_cost": env.tool_cost, "request_cost": env.request_cost,
                    "latency_s": trace["wall_latency_s"],
                    "scripted_calls": trace["scripted_prefix"]["assistant_turns"],
                    "assistant_turns": trace["metadata"]["turns"],
                    "output_tokens": sum(g.get("output_tokens", 0) for g in trace["generation"]),
                    "input_tokens": sum(g.get("input_tokens", 0) for g in trace["generation"]),
                }
                records.save(row, trace)
                print(json.dumps({"completed": records.summary()["episodes"],
                                  "expected": expected_count, "arm": arm,
                                  "case_id": case["case_id"],
                                  "termination": row["termination"]}), flush=True)
        validate_records(records.db, cases, args.arms, complete=True)
        if model_identity(path, model_id, revision) != identity:
            raise RuntimeError("Model or adapter changed during evaluation")
        verified_complete = True
    finally:
        client.close()
        summary = records.summary()
        summary.update(complete=verified_complete,
                       expected=expected_count, scope=configuration["scope"])
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        records.db.close()


if __name__ == "__main__":
    main()
