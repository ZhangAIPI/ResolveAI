"""Train a small, resumable LoRA acquisition pilot with assistant-target-only loss."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(messages):
    result = deepcopy(messages)
    for message in result:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for item in message.get("content", []):
            if "image_path" in item:
                item["image"] = Image.open(item.pop("image_path")).convert("RGB")
    return result


def encode(example, processor, max_context=8192):
    prefix = materialize(example["messages"])
    full = prefix + materialize([example["target"]])
    options = dict(tools=example["tools"])
    prefix_text = processor.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True, **options)
    full_text = processor.apply_chat_template(
        full, tokenize=False, add_generation_prompt=False, **options)
    if not full_text.startswith(prefix_text):
        raise ValueError("Native template changes the public prefix when appending a target")
    # Tokenizing the whole string can merge the prompt's final newline with the
    # assistant target. Keep the exact inference prefix and tokenize its suffix
    # separately, as generation does. No prompt or image token receives loss.
    batch = processor.apply_chat_template(prefix, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt", **options)
    target = processor.tokenizer(full_text[len(prefix_text):],
        add_special_tokens=False, return_tensors="pt").input_ids
    if target.shape[-1] == 0:
        raise ValueError("Empty assistant target")
    length = batch["input_ids"].shape[-1]
    if length + target.shape[-1] > max_context:
        raise ValueError("Context limit exceeded; never silently truncate images")
    batch["input_ids"] = torch.cat([batch["input_ids"], target], dim=-1)
    batch["attention_mask"] = torch.cat(
        [batch["attention_mask"], torch.ones_like(target)], dim=-1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in batch:
            batch[key] = torch.cat([batch[key], torch.zeros_like(target)], dim=-1)
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][:, :length] = -100
    return batch


def save_adapter(model, processor, directory, step):
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    (temporary / "checkpoint_step.json").write_text(json.dumps({"step": step}) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-examples", type=int, default=32)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("Run inside an allocated GPU worker")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    random.seed(17)
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    for split, expected in data_manifest["files"].items():
        if file_hash(args.data / (split + ".jsonl")) != expected:
            raise ValueError("Training data changed")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model": str(args.model.resolve()), "base_revision": args.base_revision, "data_sha256": file_hash(args.data / "manifest.json"),
        "max_steps": args.max_steps, "accumulation": args.accumulation,
        "validation_every": args.validate_every, "validation_examples": args.validation_examples,
        "rank": 32, "alpha": 64, "learning_rate": 1e-4, "seed": 17,
        "target_modules": "language_model attention q/k/v/o projections",
        "image_max_pixels": 512 * 512, "max_context": 8192,
        "scope": "Source-supervised identity acquisition pilot; no frozen benchmark examples",
    }
    manifest_path = args.output / "training_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["configuration"] != config or previous["script_sha256"] != file_hash(Path(__file__)):
            raise ValueError("Resume configuration or training code differs")
    complete_path = args.output / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if file_hash(Path(complete["adapter"]) / "adapter_model.safetensors") != complete["adapter_sha256"]:
            raise ValueError("Completed adapter changed")
        print(json.dumps({"stage": "training_already_complete", "step": complete["state"]["step"]}), flush=True)
        return
    manifest_path.write_text(json.dumps({
        "configuration": config, "data": data_manifest,
        "script_sha256": file_hash(Path(__file__)),
        "gpu": torch.cuda.get_device_name(0), "job": os.environ["SLURM_JOB_ID"],
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
    }, indent=2) + "\n")
    train = [json.loads(line) for line in (args.data / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (args.data / "validation.jsonl").read_text().splitlines()]
    random.Random(17).shuffle(train)
    random.Random(23).shuffle(validation)
    validation = validation[:args.validation_examples]
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 512, min_pixels=48 * 48)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    base.config.use_cache = False
    latest = args.output / "latest"
    if latest.exists():
        model = PeftModel.from_pretrained(base, latest, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("language_model" not in name for name, _ in trainable):
        raise ValueError("Unexpected trainable vision or base parameters")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=.01)
    state_path = args.output / "optimizer.pt"
    state = {"step": 0, "cursor": 0, "best_validation": math.inf, "bad_validations": 0}
    if state_path.exists():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        state.update(payload["state"])
        if json.loads((latest / "checkpoint_step.json").read_text())["step"] != state["step"]:
            raise ValueError("Adapter and optimizer checkpoint steps differ")
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    max_steps = min(args.max_steps, 2 * len(train) // args.accumulation)
    print(json.dumps({"stage": "ready", "trainable_parameters": sum(p.numel() for _, p in trainable),
                      "train_examples": len(train), "validation_examples": len(validation),
                      "max_steps": max_steps, "resume_step": state["step"]}), flush=True)
    validation_batches = [encode(row, processor) for row in validation]

    def validation_loss():
        model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for batch in validation_batches:
                n = int((batch["labels"][:, 1:] != -100).sum())
                loss = model(**{k: v.to("cuda") for k, v in batch.items()}).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                total += loss.item() * n
                tokens += n
        model.train()
        return total / tokens

    log = args.output / "metrics.jsonl"
    if state["step"] == 0:
        initial = validation_loss()
        with log.open("a") as out:
            out.write(json.dumps({"step": 0, "validation_loss": initial}) + "\n")
        print(json.dumps({"stage": "initial_validation", "loss": initial}), flush=True)
    model.train()
    started = time.monotonic()
    while state["step"] < max_steps:
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accumulation):
            row = train[state["cursor"] % len(train)]
            batch = encode(row, processor)
            loss = model(**{k: v.to("cuda") for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            total += loss.item()
            (loss / args.accumulation).backward()
            state["cursor"] += 1
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm) or norm.item() == 0:
            raise ValueError("Invalid or absent adapter gradient")
        optimizer.step()
        state["step"] += 1
        row = {"step": state["step"], "training_loss": total / args.accumulation,
               "gradient_norm": norm.item(), "elapsed_s": time.monotonic() - started,
               "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
        validate = state["step"] % args.validate_every == 0 or state["step"] == max_steps
        if validate:
            row["validation_loss"] = validation_loss()
            if row["validation_loss"] < state["best_validation"]:
                state["best_validation"] = row["validation_loss"]
                state["bad_validations"] = 0
                save_adapter(model, processor, args.output / "best", state["step"])
            else:
                state["bad_validations"] += 1
            save_adapter(model, processor, latest, state["step"])
            temporary = state_path.with_suffix(".tmp")
            torch.save({"state": state, "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
            temporary.replace(state_path)
        with log.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if validate and state["bad_validations"] >= 3:
            break
    (args.output / "complete.json").write_text(json.dumps({
        "state": state, "adapter": str(args.output / "best"),
        "adapter_sha256": file_hash(args.output / "best/adapter_model.safetensors"),
        "benchmark_performance": None, "selection": "Held-out source-disjoint teacher-forced validation loss",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
