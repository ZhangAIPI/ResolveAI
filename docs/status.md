# Execution status

Current: the 7,200-episode candidate evaluation and the subsequent 2,160-episode paired diagnostic are complete and published. These are separate experiments, not one score. See the [plain Chinese diagnosis](benchmark_diagnosis.zh-CN.md) for the next benchmark fixes. Allocation entries below describe execution history, not live resources.

Repository: https://github.com/ZhangAIPI/ResolveAI
Checkout: `/home/cxu-serve/p62/zzh136/experiments/ResolveAI`.
Runtime/data: `/home/cxu-serve/p62/zzh136/experiments/resolveai-poc`.

Initial Slurm allocations 24679 and 24689 each provided two RTX A6000 GPUs
(48 GB each), 16 CPUs and 128 GB host memory. Both allocations are on `sclera`,
hostname `node2x32b.cs.rochester.edu`; they are two allocations on one host.
All implementation, dependencies, rendering and model runs execute inside these
allocations using `srun --jobid=JOB --overlap`, not on the control host. p62 had
approximately 62 TB free at initialization; p61 was nearly full. Never cancel or
replace these allocations without authorization for the other experiments.

Isolated runtime reuses PyTorch 2.4.0+cu121 with Transformers 4.57.6,
Torchvision 0.19.0+cu121, Pillow 12.2.0 and RapidOCR 1.4.4. Portable Blender 4.3.2
Cycles successfully rendered six multi-view/time images on CUDA. Frozen grounding
providers use Grounding DINO tiny, DINOv2 base and CLIP-B/32. Models and datasets
remain outside Git. CLIP original tensors are converted to safetensors without
changing values; conversion and source hashes are recorded.

The environment supports persistent function-call conversations, source-coordinate
crop/zoom/OCR, three grounding tools, quality/metadata, exact requests, immutable
world state, isolated branches, region/time sufficient-evidence evaluation and
training-compatible trajectories/preferences. See [contract](environment.md).

The earlier pilot completed 384 episodes on 24 public images; its
[results](results/public_pilot.md) remain a historical resolution-release baseline.
The fixed evaluation now contains 300 public source images and 1,200 availability
cases, excludes previous pilot originals and compares three local frozen MLLMs.
[Protocol](benchmark.md) specifies paired policies, bounds and scope. Do not read
benchmark construction or a running process as completed evaluation results.

No new self-collected photographs are assumed. Region-level independent annotation,
LoRA/DPO training, broad multi-view simulation construction and verified public
identity/time adaptation remain research work. The six-image Blender scene is
unreviewed integration data. Neither it nor a quality-proxy public score establishes
simulation-training transfer or actual dispute-case accuracy. Conference dates from the supplied
proposal have not been independently checked in this implementation task.

## GPU renewal: 2026-10-05

- `24845`: InternVL resume, running on `sclera`; new episodes verified.
- `24846`: Qwen resume, submitted with `afterany:24689` to avoid concurrent result writers.
- Each allocation: 2×A6000, 16 CPUs, 128GB RAM, 12 hours, partition `macula`.
- Outputs and scheduler logs remain on p62 under the existing runtime. Evaluation code stays frozen at `4810a1e`; completed episodes are reused. Jobs exit after their evaluation/reporting work finishes.

## Qwen3 parallel continuation: 2026-10-05

`24847` runs Qwen3 on all four A6000 GPUs in `sclera`, with 32 CPUs, 256GB RAM and a four-hour limit. `24846` was intentionally replaced after 1,989 completed episodes and matching native trajectories were verified and migrated. Eight family shards preserve the frozen `4810a1e` evaluation; GPU queues begin with 107/96/104/104 remaining episodes. Model and image contents were not duplicated. The serial archive is excluded from the final counts. The job merges results and publishes the three-model report after completion, then exits.

## Paired capability diagnostic complete: 2026-10-05

Job 24872 ran Qwen3-VL-8B, Qwen2.5-VL-7B and InternVL3.5-8B on the four A6000 GPUs in sclera, with all code/data/results on p62. All 2,160 episodes are recorded; all 120 scripted retrieval checks passed without changing world state. [Results](results/capability_diagnostic.md) separate final label agreement from receiving an original. Frozen model code was f304912; later documentation commits do not change these runs. Next: validate answer/tool adapters and review visual judgeability before using failures as training evidence.
