# Fixed public benchmark v0.2

Assume no additional self-collected photographs. Build a fixed public benchmark
rather than deferring verification to a future collection. MVTec AD supports
anomaly recognition and localization; it cannot validate all cross-view identity
or temporal claims in the proposal. Those claims need an independently annotated
public source that actually records shared identity/time, or controlled simulation.
The current Blender scene is integration-only, not a substitute real-photo test.

## Frozen sample

Select ten good and ten anomalous official-test originals in each of all 15
categories: **300 source images**. Exclude all 24 previous pilot originals and the one compatibility-development original.
Sampling is hash-deterministic within category and binary label. Preserve dataset
revision, original path, original split, original label, dimensions and file hash.
The source unit is an original image; physical-object identity is not provided,
so confidence intervals must not be described as independent-object intervals.
MVTec's original noncommercial share-alike license governs the experiment.

Every source image has four correlated versions: original initially visible
(Sufficient), uniformly downsampled preview with an obtainable original
(Obtainable), preview with no original in the evidence pool (Missing), and preview
with original unavailable (Unavailable). This yields **1,200 cases per policy**.
Preview construction uses no defect masks. Missing and Unavailable share public
request options and failure responses. A high-quality original is an operational
sufficiency proxy, not an independently annotated minimal semantic evidence set.

```bash
python -m resolveai.public_data /p62/benchmark-data --categories all --per-label 10 \
  --exclude-cases /p62/development_cases.json
python -m resolveai.benchmark /p62/benchmark-data /p62/model /p62/runs/model-name \
  --model-id OWNER/MODEL --revision COMMIT --grounding-root /p62/frozen-tools
```

## Model and policy comparisons

Pinned local models: Qwen/Qwen3-VL-8B-Instruct,
Qwen/Qwen2.5-VL-7B-Instruct and OpenGVLab/InternVL3_5-8B-HF.
Use bfloat16, SDPA, greedy decoding, at most 384 generated tokens per turn,
16 turns, 8,192 context tokens and budget 12. Tool schemas, prices and frozen
perception providers are shared across interactive models. Checkpoint-native
image preprocessing is recorded; it is not assumed identical across architectures.
Qwen input is bounded by 512² pixels; InternVL's standard image processor uses
448² without dynamic patch tiling. Original and preview remain the same source
pixels regardless of model resizing.

Initial-only baseline exposes finish and the initial evidence, without additional
acquisition. Interactive agent exposes the full tool pool and preserves every
actual call, result and image across turns. These are explicitly different access
policies. Do not describe the initial baseline as having made equal tool calls.
Each model runs both policies on all cases: **2,400 episodes/model**, planned
**7,200 episodes across three models**.

Qwen2.5-VL and InternVL official chat templates omit structured calls and tool
definitions. All three benchmark models therefore use the same portable adapter:
JSON tool schemas in system instructions, one `<tool_call>` JSON per assistant
turn, and identified tool results represented as user-role typed content. A fixed assistant syntax prefix `<tool_call>{"name":` is supplied to every model;
the model generates the tool name and arguments. A complete JSON object is accepted
if the model omits the closing transport tag; missing functions or arguments are never inferred. The
stored canonical conversation retains actual assistant/tool roles. This preserves
history and pixels without claiming native function-call training for every model.
Qwen3-VL's native template remains available for integration tests.

Execution may partition source families into disjoint `--shards N --shard-index I`
runs on separate GPUs. Each episode still uses one GPU; shared source families
remain in one shard. Reporting merges only complete, disjoint shards with identical
model/data/tool settings and retains their manifests.

## Metrics and integrity

Report source-label accuracy, original-image-grounded proxy accuracy, unsupported
proxy decisions, coverage, selective error, completion, requests, tool use/cost,
input/output tokens, model latency and full episode latency. Unfinished episodes
are retained with no invented finish decision and contribute zero to headline
accuracy; report their count separately. Safe abstention is not a correct binary
visual classification, so report both accuracy and unsupported/coverage behavior.
Report each availability condition and category separately. Paired improvements
and bootstrap confidence intervals resample source-image families, preserving
all four versions. One fixed budget supplies a coverage/risk point, not a full
confidence or budget curve.

Runs resume only if dataset hashes, source-code digest, pinned models/providers,
decoding, tools and limits match exactly. Each run stores a manifest, compact
results JSONL and canonical trajectories with content-addressed image blobs outside
Git. Keep compact summaries and manifests in the repository. Do not tune policies
on this frozen test set; later training uses official training material and
simulation with separate family/asset groups.

This verifies environment/tool usefulness and frozen policy behavior. It does
not establish LoRA/DPO gains, independently grounded accuracy, real dispute data
or simulation-to-real training transfer. Those claims require separate experiments.

Official sources: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
[InternVL3.5](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF).


## Repartitioning completed evaluations

Stop the original writers before using `python -m resolveai.repartition SOURCE DATA OUTPUT --shards 8 --name qwen3`. The helper checks the dataset hash, preserves each completed result and its exact matching trajectory, hardlinks immutable image blobs, keeps availability variants together by family, and aborts if a writer changes source files. Model weights, source code and rollout settings remain unchanged; only execution shard fields differ. Keep the serial archive for provenance and exclude it from merged episode counts.

`scripts/evaluate_local_models.py qwen3 RUNTIME DATA OUTPUT --code FROZEN_CODE --gpus 4 --shards 8` queues the shards by remaining episode count across four GPUs. Finished results are reused. Final reporting requires all shards with no overlapping episodes. The node count and GPU count are execution resources, not independent statistical units; confidence intervals still resample source families.
