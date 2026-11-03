"""Budgeted closed-model evaluation; dry-run is the default and makes no API calls."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import stat

from resolveai.api_budget import ApiBudget, BudgetStop
from resolveai.benchmark import locked_manifest, source_digest
from resolveai.conversation import Conversation
from resolveai.diagnostic import text_direct
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding
from resolveai.openai_client import OpenAIClient
from resolveai.rollout import run
from scripts.evaluate_study import ARMS, Records, normalize_agent_record


def select_cases(plan, cohort, stage, variants):
    if plan["dataset_sha256"] != cohort["dataset_sha256"]:
        raise ValueError("Cohort dataset mismatch")
    data = Path(plan["data"])
    if hashlib.sha256((data/"cases.json").read_bytes()).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    families = set(cohort["family_ids"])
    if not families <= set(plan["family_ids"]) or len(families) != 209:
        raise ValueError("Expected the frozen 209-family paper cohort")
    cases = [c for c in json.loads((data/"cases.json").read_text())
             if c["family_id"] in families and c["variant"] in variants]
    if len(cases) != len(families)*len(variants):
        raise ValueError("Missing or duplicate paper cases")
    order = {name:i for i,name in enumerate(("Sufficient","Obtainable","Missing","Unavailable"))}
    cases.sort(key=lambda c:(order[c["variant"]],c["case_id"]))
    if stage == "pilot":
        # Deterministic and label-blind: equal task counts, all requested access variants.
        picked = set()
        sufficient = [c for c in cases if c["variant"]=="Sufficient"]
        for task in ("state","identity"):
            # The smallest two IDs per task, not hand-picked successful examples.
            picked.update(sorted(c["family_id"] for c in sufficient if c["task"]==task)[:2])
        cases = [c for c in cases if c["family_id"] in picked]
    return cases


def configure_api_key(path):
    if path is None:
        return
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("API key file must be a private, owner-only regular file")
    lines = path.read_text().splitlines()
    if len(lines) != 1 or not lines[0].startswith("OPENAI_API_KEY="):
        raise ValueError("Expected one OPENAI_API_KEY assignment")
    key = lines[0].split("=", 1)[1].strip()
    if not key.startswith("sk-") or len(key) < 40 or any(c.isspace() for c in key):
        raise ValueError("Invalid API key format")
    os.environ["OPENAI_API_KEY"] = key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("cohort",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--config",type=Path,default=Path("docs/experiments/openai-budget.json"))
    p.add_argument("--ledger",type=Path,required=True,help="Reuse one external ledger across all models and stages")
    p.add_argument("--grounding-root",type=Path,required=True)
    p.add_argument("--stage",choices=("pilot","paper"),default="pilot")
    p.add_argument("--all-variants",action="store_true")
    p.add_argument("--key-file",type=Path,help="Owner-only credential file outside Git; never a key value")
    p.add_argument("--budget-usd",type=float,default=60,help="Prefunded ledger cap, never above USD 60")
    p.add_argument("--stage-name",help="Persistent budget-stage identifier")
    p.add_argument("--stage-budget-usd",type=float,help="A smaller stage limit within the shared USD 60 cap")
    p.add_argument("--execute",action="store_true")
    args = p.parse_args()
    configure_api_key(args.key_file)
    config = json.loads(args.config.read_text())
    plan, cohort = json.loads(args.plan.read_text()), json.loads(args.cohort.read_text())
    variants = ["Sufficient","Obtainable","Missing","Unavailable"] if args.all_variants else ["Sufficient"]
    cases = select_cases(plan,cohort,args.stage,variants)
    arms = ARMS[:2] if config.get("endpoint")=="decisions" else ARMS
    stage_cap = args.stage_budget_usd if args.stage_budget_usd is not None else (1 if args.stage=="pilot" else 60)
    if not 0 < stage_cap <= 60:
        p.error("Stage budget must be within USD 60")
    preview = {"stage":args.stage,"model":config["model_id"],"cases":len(cases),
               "episodes":len(cases)*len(arms),"variants":variants,"tasks":dict(Counter(c["task"] for c in cases)),
               "shared_cap_usd":60,"ledger_cap_usd":args.budget_usd,"stage_cap_usd":stage_cap,"arms":list(arms),"budget_stage":args.stage_name or args.stage,
               "key_configured":bool(os.environ.get("OPENAI_API_KEY")),"execute":args.execute,
               "scope":"Unreviewed source-label comparison; formal_grounded_accuracy is null"}
    print(json.dumps(preview,indent=2),flush=True)
    if not args.execute:
        return
    if not os.getenv("SLURM_JOB_ID") or os.getenv("SLURMD_NODENAME", socket.gethostname().split(".")[0]) != "macula":
        p.error("Execute on the allocated macula GPU node")
    if not os.environ.get("OPENAI_API_KEY"):
        p.error("OPENAI_API_KEY is missing; no paid request sent")
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_config = {**preview,"dataset_sha256":plan["dataset_sha256"],"api_config":config,
                       "family_ids":sorted({c["family_id"] for c in cases}),"budget":plan["budget"],
                       "max_turns":plan["max_turns"],"max_context_tokens":8192,"source_sha256":source_digest(),
                       "model_id":config["model_id"],"revision":"API service alias; actual returned model retained per call",
                       "ledger":str(args.ledger.resolve()),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grounding_manifest":json.loads((args.grounding_root/"manifest.json").read_text())}
    locked_manifest(args.output/"manifest.json",manifest_config)
    ledger = ApiBudget(args.ledger,args.budget_usd)
    records = Records(args.output)
    backend = FrozenGrounding(args.grounding_root,device="cpu")
    client = OpenAIClient(config,ledger,stage=preview["budget_stage"],stage_limit_usd=preview["stage_cap_usd"])
    stopped = None
    try:
        for case in cases:
            for arm in arms:
                if records.has(case["case_id"],arm):
                    continue
                client.begin_episode(case["case_id"],arm)
                env = Environment(case,case["asset_root"],plan["budget"],grounding_backend=backend)
                fingerprint = env._world.fingerprint
                if arm=="full_available_text":
                    env._released.update(e["id"] for e in case["evidence"] if e["available"])
                trace = (text_direct(env,case,client) if arm.endswith("text") else
                         normalize_agent_record(run(Conversation(env),client,plan["max_turns"]),env))
                if fingerprint != env._world.fingerprint:
                    raise RuntimeError("World changed")
                trace.update(evaluation=None,schema_version="resolveai-diagnostic-v1")
                trace["metadata"] = {"family_id":case["family_id"],"case_id":case["case_id"],
                                     "arm":arm,"official_split":case["official_split"],
                                     "annotation_status":"unreviewed-development"}
                verdict = trace["decision"]["verdict"] if trace.get("decision") else None
                row = {"case_id":case["case_id"],"family_id":case["family_id"],"task":case["task"],
                       "variant":case["variant"],"arm":arm,"verdict":verdict,
                       "termination":trace["termination"],"errors":trace["errors"],
                       "requests":env.requests,"tool_calls":env.calls,"released":sorted(env._released),
                       "source_label":case["provenance_truth"],
                       "source_label_match":verdict==case["provenance_truth"] if case["provenance_truth"] else None,
                       "tool_cost":env.tool_cost,"request_cost":env.request_cost,"latency_s":trace["wall_latency_s"]}
                records.save(row,trace)
                print(json.dumps({"completed":records.summary()["episodes"],**ledger.status()}),flush=True)
    except BudgetStop as error:
        stopped = str(error)
    except Exception as error:
        stopped = type(error).__name__
        raise
    finally:
        summary = {**records.summary(),"expected":len(cases)*len(arms),"budget":ledger.status(),
                   "stop_reason":stopped,"formal_grounded_accuracy":None}
        summary["complete"] = summary["episodes"]==summary["expected"]
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        records.db.close()
        ledger.close()
    if stopped:
        raise SystemExit(stopped)


if __name__=="__main__":
    main()
