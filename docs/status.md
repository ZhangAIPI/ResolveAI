# Execution status

Historical: the 7,200-episode candidate evaluation and the subsequent 2,160-episode paired diagnostic are complete and published. These are separate experiments, not one score. See the [plain Chinese diagnosis](benchmark_diagnosis.zh-CN.md) for the next benchmark fixes. Allocation entries below describe execution history, not live resources.

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

## Benchmark repair and interface diagnosis: 2026-10-05

All three models completed the paired 456-episode development diagnosis in job
24895 (50m27s); the allocation exited normally. Job 24911 completed 12 qualitative
Qwen2.5 input-binding checks. These runs explain interface/acquisition failures;
they are not reviewed benchmark scores. See the
[Chinese examples and results](results/repaired_checks.zh-CN.md).

The current v0.4.1 pool on sclera/p62 contains 478 combinations and 1,912 release
configurations: 188 state queries, 32 positive and 32 resolution-matched negative
identity controls, plus 226 unlabelled same-category identity candidates. It
references existing photos and normal-reference files without copying them.
All 478 retrieval/immutable-world preflights and 62 software checks passed.
[Construction audits](results/shortcut_audit.json) show the repaired dimension and
claim-wording rules match 50% of their balanced binary source labels. Independent
visual truth and sufficient-evidence review remains pending; formal scoring and
training exports reject unreviewed material.

## Current human study and model evaluation: 2026-10-06

The short study has 20 participants, each assigned three search trials (75 seconds each) and two staged independent reviews (120 seconds each). The sampled 20 families are source-disjoint within participants; 10 families receive two people per search condition, and all 20 receive two independent reviews. See the [Chinese instructions and design](human_study.zh-CN.md). No real participant responses have been collected at launch.

Jobs 24937–24940 run four independent workers on sclera: two Qwen3-VL-8B shards, one Qwen2.5-VL-7B worker and one InternVL3.5-8B worker. Each worker has one A6000, six CPUs and 56 GB RAM, with a 12-hour limit. Their code is frozen at 5dc8125. The model set has 96 families, four releases and three arms: 1,152 episodes per model, 3,456 total. All 20 human families are included. SQLite commits each completed episode and native trace together for resume. Formal metrics await independent visual reviews; startup failures from jobs 24930–24933 are archived separately and are not model failures or current completed episodes.

73 software tests and a full isolated Chromium study flow passed. Web job 24958 uses a separate CPU-only allocation on sclera, with a three-day limit ending 2026-10-09 00:16 America/New_York. The [public instruction page](https://pathology-decision-solved-mood.trycloudflare.com) and private organizer dashboard passed external HTTPS and real-browser checks; anonymous API access returns 403. Read-only production checks left consent, starts, answers and events empty. Credentials and raw responses remain private on p62. See [deployment checks](results/human_web_deployment.json) and the [timestamped model progress snapshot](results/human_model_progress.json).

## Human UI revision: 2026-10-06

The human-only interface now uses direct photo selection and per-photo buttons, with derived images replacing their photo in place. All human time, points and action caps are removed; elapsed time and action counts are recorded. A Chinese/English toggle preserves selection and unsaved answers. Records are tagged `human-ui-v3-unrestricted`; older responses and fixed model evaluation budgets remain unchanged. Existing invitations and the public hostname are retained.

## Single-answer human study: 2026-10-06

Human UI v4 displays low-resolution photos at a usable size, provides image operations in all five tasks, replaces a preview when its matching original is released, and finishes each review with one submission. Browser zoom is a logged presentation-only action, without resampling pixels. The two reviews now provide full-pool truth/evidence votes (visual-truth-review-v2), not initial/pool availability labels. The analyzer reports two-human full-material label agreement separately and never promotes these partial reviews to the formal evidence-sufficiency benchmark. A private preview invitation uses a separate SQLite database. Existing formal assignments and answers are retained.

### 2026-10-06：人类流程简化与题目多样性

- 新 UI `human-ui-v5-optional-regions`：直接提交判断，取消必选照片；画框可选。提供的目标用橙色虚线，人画的细节用蓝色实线。
- 修复缓存图片在加入页面前加载导致的零尺寸 canvas：按真实显示尺寸绘制并跟随尺寸变化，浏览器实测书类题不再出现整图绿色。裁剪保持照片位置，目标不在裁剪范围时不绘制目标框。
- 审核记录升级为 `visual-truth-review-v3`，只用于完整材料结论共识；可选区域不推导最小充分证据，不产生人的 grounded 引用分数。旧回答按原协议保留。
- 固定题库和各条件次数，只重排未开始者：前三道搜证物品类别互不重复；19份中15份五类不同、4份四类不同。已开始者不更换案件。
- 77项回归测试和真实五题的隔离浏览器检查通过；不向正式/试用库写入测试回答。中英文 GitHub 演示截图已更新。
- 超分不自动用于当前人工实验；低清材料先请求原图。未来超分工具可独立消融，生成细节不作为新增事实证据。

### 2026-10-06：去掉人工页面无明确结果的工具按钮

人工界面去掉“比较”和“检查图片”：照片已并排显示，这些按钮没有提供新的、可理解的结果。裁剪后的“恢复整图”、画框后的“清除我的框”和可用时的“识别文字”直接显示在图片下，不再放在“更多”菜单。模型工具、题单、邀请和答案不变。
