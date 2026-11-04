"""Small single-GPU online GRPO trial with real multi-turn visual tool rollouts.

The policy sees Conversation.public() only. Source annotations stay in the
environment and enter the optimizer solely as a scalar terminal reward.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.grounding_tools import FrozenGrounding
from resolveai.model_processing import chat_inputs, processor_options
from resolveai.policy_gradient import group_advantages, grpo_token_loss, trajectory_reward
from resolveai.vlm_worker import decode_tool_call, generation_eos_ids, materialize, portable_tools


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activate(model, name):
    model.set_adapter(name)
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(name == "default" and
            (".lora_A.default." in key or ".lora_B.default." in key))


def token_logps(model, turn):
    batch = {key: value.to(model.device) for key, value in turn["batch"].items()}
    target = turn["tokens"].to(model.device).unsqueeze(0)
    batch["input_ids"] = torch.cat([batch["input_ids"], target], -1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(target)], -1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in batch:
            batch[key] = torch.cat([batch[key], torch.zeros_like(target)], -1)
    length = target.shape[-1]
    logits = model(**batch, use_cache=False, logits_to_keep=length+1).logits[:, -length-1:-1]
    return logits.float().log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze()



def policy_inputs(processor, public):
    """Use exactly the native/portable transport selected by the evaluation worker."""
    template = processor.chat_template or ""
    native = isinstance(template, str) and "tools" in template and "tool_calls" in template
    adapter = "native" if native else "portable"
    messages = materialize(public["messages"])
    if not native:
        messages = portable_tools(messages, public["tools"], assistant_prefix=True)
    batch = chat_inputs(processor, messages, tools=public["tools"] if native else None)
    prefix = '' if native else '<tool_call>\n{"name":'
    if prefix:
        ids = processor.tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
        batch["input_ids"] = torch.cat([batch["input_ids"], ids], -1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(ids)], -1)
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in batch:
                batch[key] = torch.cat([batch[key], torch.zeros_like(ids)], -1)
    return batch, prefix, adapter


def assert_unadapted(model):
    matrices = [value for name,value in model.named_parameters() if ".lora_B.default." in name]
    if not matrices or any(torch.count_nonzero(value).item() for value in matrices):
        raise ValueError("RL-only initialization must have zero-output LoRA updates")

def rollout(model, processor, case, grounding, *, sample, keep_tokens):
    env = Environment(case, case["asset_root"], 12, grounding_backend=grounding)
    session = Conversation(env)
    session.messages[0]["content"] += "\nEpisode limit: 8 assistant turns; finish within this limit.\n"
    turns, raw_calls = [], []
    model.eval()
    for _ in range(8):
        # No case metadata, source label or reward is passed to the processor.
        public = session.public()
        batch, prefix, adapter = policy_inputs(processor, public)
        if batch["input_ids"].shape[1] > 8192:
            session.termination = "context_limit"
            break
        with torch.no_grad():
            generated = model.generate(**batch.to(model.device), max_new_tokens=384,
                do_sample=sample, **({"temperature":1.0,"top_p":1.0,"top_k":0} if sample else {}),
                use_cache=True, repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
                eos_token_id=generation_eos_ids(model.generation_config.eos_token_id, processor.tokenizer.eos_token_id),
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
        tokens = generated[0, batch["input_ids"].shape[1]:].detach().cpu()
        if keep_tokens:
            turns.append({"batch":{key:value.detach().cpu() for key,value in batch.items()}, "tokens":tokens})
        raw = prefix + processor.tokenizer.decode(tokens, skip_special_tokens=True)
        raw_calls.append(raw)
        try:
            call, text = decode_tool_call(raw, public["tools"], adapter, model.config.model_type, True)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            session.invalid_output(raw)
        else:
            session.call(call, text)
        if session.termination == "finished":
            break
    if session.termination is None:
        session.termination = "max_turns"
    reward, correct = trajectory_reward(session)
    return {"turns":turns, "reward":reward, "correct":correct, "raw":raw_calls,
            "termination":session.termination, "requests":env.requests, "errors":session.errors}


def save_policy(model, processor, directory, group):
    temporary = directory.with_name(directory.name+".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, selected_adapters=["default"], safe_serialization=True)
    processor.save_pretrained(temporary)
    (temporary/"checkpoint_group.json").write_text(json.dumps({"group":group})+"\n")
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model_spec", type=Path)
    parser.add_argument("sft_run", type=Path)
    parser.add_argument("grounding", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--from-base", action="store_true", help="Skip SFT; use - for the sft_run argument")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("Run on a retained allocated GPU")
    if args.groups < 1 or args.group_size < 2:
        parser.error("Positive groups and at least two samples are required")
    torch.set_num_threads(4)
    torch.manual_seed(101)
    random.seed(101)
    spec = json.loads(args.model_spec.read_text())
    initial = None
    sft = None
    rank = 32
    if not args.from_base:
        waiting = time.monotonic()
        while True:
            try:
                sft = json.loads((args.sft_run/"complete.json").read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                if time.monotonic()-waiting > 21600:
                    raise TimeoutError("SFT checkpoint was not ready within six hours")
                print(json.dumps({"stage":"waiting_for_sft_checkpoint"}), flush=True)
                time.sleep(20)
        sft_manifest = json.loads((args.sft_run/"training_manifest.json").read_text())
        if sft_manifest["configuration"]["base_revision"] != spec["revision"]:
            raise ValueError("SFT and RL base revisions differ")
        initial = Path(sft["adapter"])
        if sha(initial/"adapter_model.safetensors") != sft["adapter_sha256"]:
            raise ValueError("SFT adapter changed")
        if Path(sft_manifest["configuration"]["base_model"]).resolve() != Path(spec["path"]).resolve():
            raise ValueError("SFT and RL base paths differ")
        rank = sft_manifest["configuration"]["rank"]
    processor_path = initial or Path(spec["path"])
    processor = AutoProcessor.from_pretrained(processor_path,max_pixels=512*512,min_pixels=48*48,
                                              **processor_options(processor_path))
    template = processor.chat_template or ""
    adapter = "native" if isinstance(template,str) and "tools" in template and "tool_calls" in template else "portable"
    if sft and adapter != sft_manifest["configuration"].get("tool_adapter",adapter):
        raise ValueError("SFT and RL tool transports differ")
    data_manifest = json.loads((args.data/"manifest.json").read_text())
    for split,checksum in data_manifest["files"].items():
        if sha(args.data/(split+"_cases.json")) != checksum:
            raise ValueError("Private RL worlds changed")
    cases = json.loads((args.data/"train_cases.json").read_text())
    validation = json.loads((args.data/"validation_cases.json").read_text())
    rng = random.Random(101)
    rng.shuffle(cases)
    # Eight held-out source families, balanced by task and label; all access variants.
    buckets = {}
    for case in validation:
        if case["variant"] == "Sufficient":
            buckets.setdefault((case["task"],case["annotation"]["verdict"]),[]).append(case["family_id"])
    val_families = [family for _,group in sorted(buckets.items()) for family in rng.sample(sorted(group),2)]
    validation = [case for family in val_families for case in validation if case["family_id"]==family]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"base_model":spec["path"],"base_revision":spec["revision"],
        "initial_adapter_sha256":sft["adapter_sha256"] if sft else None,
        "initialization":"base" if args.from_base else "sft","data_sha256":sha(args.data/"manifest.json"),
        "groups":args.groups,"group_size":args.group_size,"ppo_epochs":2,"clip":.2,"beta":.02,
         "learning_rate":1e-5,"seed":101,"reward_scaling":"none; group mean baseline",
        "sampling":{"temperature":1.0,"top_p":1.0,"top_k":0,"repetition_penalty":1.0},
        "budget":12,"image_max_pixels":512*512,"tool_adapter":adapter,"dropout":0.0,
        "lora_rank":rank,
        "grounding_manifest":json.loads((args.grounding/"manifest.json").read_text()),"max_turns":8,"max_input_tokens":8192,"max_output_tokens":384,
        "validation_cases":[case["case_id"] for case in validation],
        "checkpoint_selection":"Held-out source success, then cost-aware reward; no benchmark feedback",
        "reward":"source_grounded_correct:+1/otherwise:-1; -0.01*tool_cost-0.02*request_cost-0.02*errors",
        "scope":"Online sampled multi-turn GRPO pilot; source-derived reward, not reviewed sufficiency",
        "loss_code_sha256":sha(Path(__file__).parents[1]/"src/resolveai/policy_gradient.py")}
    manifest_path = args.output/"training_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["configuration"] != config or previous["script_sha256"] != sha(Path(__file__)):
            raise ValueError("RL resume configuration or code differs")
    if (args.output/"complete.json").exists():
        complete = json.loads((args.output/"complete.json").read_text())
        if sha(Path(complete["adapter"])/"adapter_model.safetensors") != complete["adapter_sha256"]:
            raise ValueError("Completed RL adapter changed")
        print(json.dumps(complete), flush=True)
        return
    base = AutoModelForImageTextToText.from_pretrained(spec["path"],
        dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa")
    latest = args.output/"latest"
    if args.from_base:
        initial = args.output/"initial"
        if latest.exists():
            if not initial.exists():
                raise ValueError("RL-only resume lost its reference initialization")
            model = PeftModel.from_pretrained(base,latest,is_trainable=True)
        elif initial.exists():
            model = PeftModel.from_pretrained(base,initial,is_trainable=True)
            assert_unadapted(model)
        else:
            model = get_peft_model(base,LoraConfig(r=rank,lora_alpha=64,lora_dropout=0.,
                bias="none",task_type="CAUSAL_LM",
                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"))
            assert_unadapted(model)
            save_policy(model,processor,initial,0)
        if manifest_path.exists() and previous["initialization"]["reference_adapter_sha256"] != sha(initial/"adapter_model.safetensors"):
            raise ValueError("RL-only reference initialization changed")
    else:
        model = PeftModel.from_pretrained(base,latest if latest.exists() else initial,is_trainable=True)
    model.load_adapter(initial,adapter_name="reference",is_trainable=False)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p = 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    activate(model,"default")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0)
    state = {"group":0,"optimizer_updates":0,"best_validation":-float("inf"),"best_validation_success":-1.0,"best_group":0}
    state_path = args.output/"optimizer.pt"
    if state_path.exists():
        payload = torch.load(state_path,map_location="cpu",weights_only=False)
        state.update(payload["state"])
        if json.loads((latest/"checkpoint_group.json").read_text())["group"] != state["group"]:
            raise ValueError("RL adapter and optimizer groups differ")
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    manifest_path.write_text(json.dumps({"configuration":config, "data":{**data_manifest,"adaptation_name":"RL only" if args.from_base else "Online GRPO"},
        "script_sha256":sha(Path(__file__)),"initialization":{"sft_used":not args.from_base,
            "reference_adapter":str(initial),"reference_adapter_sha256":sha(initial/"adapter_model.safetensors")},
        "gpu":torch.cuda.get_device_name(0),
        "trainable_parameters":sum(p.numel() for p in parameters),
        "versions":{"torch":torch.__version__,"transformers":__import__("transformers").__version__,
                    "peft":__import__("peft").__version__}},indent=2)+"\n")
    grounding = FrozenGrounding(args.grounding)
    def validate():
        activate(model,"default")
        results = [rollout(model,processor,case,grounding,sample=False,keep_tokens=False) for case in validation]
        return {"reward":sum(row["reward"] for row in results)/len(results),
                "source_success":sum(row["correct"] for row in results)/len(results)}
    if state["group"] == 0:
        score = validate()
        state["best_validation"] = score["reward"]
        state["best_validation_success"] = score["source_success"]
        save_policy(model,processor,args.output/"best",0)
        print(json.dumps({"stage":"initial_validation",**score}),flush=True)
    print(json.dumps({"stage":"ready","config":config}),flush=True)
    for group_index in range(state["group"],args.groups):
        group_started = time.monotonic()
        case = cases[group_index % len(cases)]
        activate(model,"default")
        samples = [rollout(model,processor,case,grounding,sample=True,keep_tokens=True)
                   for _ in range(args.group_size)]
        advantages = group_advantages([row["reward"] for row in samples])
        updates, loss_total, kl_total = 0,0.,0.
        if advantages.abs().sum() > 0:
            for row in samples:
                for turn in row["turns"]:
                    model.eval()
                    activate(model,"default")
                    with torch.no_grad():
                        turn["old"] = token_logps(model,turn).detach().cpu()
                    activate(model,"reference")
                    with torch.no_grad():
                        turn["ref"] = token_logps(model,turn).detach().cpu()
            for _ in range(2):
                activate(model,"default")
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_total, kl_total = 0.,0.
                for row,advantage in zip(samples,advantages):
                    count = sum(turn["tokens"].numel() for turn in row["turns"])
                    if not count:
                        continue
                    for turn in row["turns"]:
                        logps = token_logps(model,turn)
                        losses,kl = grpo_token_loss(logps,turn["old"].to(model.device),
                                                  turn["ref"].to(model.device),advantage.to(model.device))
                        loss = losses.sum()/(count*args.group_size)
                        loss.backward()
                        loss_total += loss.detach().item()
                        kl_total += kl.detach().sum().item()/(count*args.group_size)
                norm = torch.nn.utils.clip_grad_norm_(parameters,1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite policy gradient")
                if norm.item()==0:
                    continue
                optimizer.step()
                state["optimizer_updates"] += 1
                updates += 1
        state["group"] = group_index+1
        metrics = {"group":state["group"],"optimizer_updates":state["optimizer_updates"],
            "case_id":case["case_id"],"task":case["task"],"variant":case["variant"],
            "rewards":[row["reward"] for row in samples],"advantages":advantages.tolist(),
            "updates_this_group":updates,"loss":loss_total,"kl":kl_total,
            "source_success":sum(row["correct"] for row in samples)/len(samples),
            "requests":sum(row["requests"] for row in samples),
            "sampled_assistant_tokens":sum(turn["tokens"].numel() for row in samples for turn in row["turns"]),
            "sampled_input_tokens":sum(turn["batch"]["input_ids"].shape[-1] for row in samples for turn in row["turns"]),
            "gpu_peak_gib":torch.cuda.max_memory_allocated()/1024**3}
        if state["group"]%8==0 or state["group"]==args.groups:
            score = validate()
            metrics["validation"] = score
            if (score["source_success"],score["reward"]) > (state["best_validation_success"],state["best_validation"]):
                state["best_validation"],state["best_group"] = score["reward"],state["group"]
                state["best_validation_success"] = score["source_success"]
                save_policy(model,processor,args.output/"best",state["group"])
            activate(model,"default")
            save_policy(model,processor,latest,state["group"])
            temporary = state_path.with_suffix(".tmp")
            torch.save({"state":state,"optimizer":optimizer.state_dict(),
                        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},temporary)
            temporary.replace(state_path)
        metrics["group_elapsed_s"] = time.monotonic()-group_started
        with (args.output/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics)+"\n")
        with (args.output/"rollouts.jsonl").open("a") as stream:
            stream.write(json.dumps({"group":state["group"],"case_id":case["case_id"],
                "rollouts":[{k:v for k,v in row.items() if k!="turns"} for row in samples]})+"\n")
        print(json.dumps(metrics),flush=True)
    if state["optimizer_updates"] == 0:
        raise RuntimeError("No RL updates: sampled groups had no reward variance; retain diagnostics")
    (args.output/"complete.json").write_text(json.dumps({"state":state,
        "adapter":str(args.output/"best"),"adapter_sha256":sha(args.output/"best/adapter_model.safetensors"),
        "selection":"Held-out source-family success, then reward; benchmark performance unknown"},indent=2)+"\n")


if __name__ == "__main__":
    main()
