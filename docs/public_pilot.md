# Public-image proof of concept

The initial pilot uses 24 source images from MVTec AD's official test split:
three object categories, four normal and four anomalous images per category.
Sampling is pinned and label-balanced before inference. No policy training or
prompt selection uses pilot results. Original labels are retained privately.

Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
Mirror: https://huggingface.co/datasets/Voxel51/mvtec-ad
Mirror revision: `30a183a3b96e3aef953f230784b123b719b09d97`.
The original CC BY-NC-SA 4.0 license governs the images and derived dataset,
including when mirror metadata differs. Do not commit or redistribute images
as ordinary source code. Cite Bergmann et al., CVPR 2019, MVTec AD.

A 48-pixel overview is produced uniformly from each original using Lanczos,
without masks or labels. Four versions expose the original initially, on
request, nowhere in the accessible pool, or as unavailable. All versions retain
the original independent anomaly label. Abstention is safe but does not count
as label accuracy. This avoids relabeling a visibly anomalous preview as unknown.

The original-image rule is a deliberately strict operational quality proxy,
not independent human annotation of minimal sufficient evidence. Report raw
label accuracy alongside proxy grounded correctness; never label this as the
paper's semantic grounded accuracy. Low-resolution images may already suffice.
The experiment tests resolution/evidence release only: no cross-view identity,
state change, causality, newly photographed view, or simulation-to-real transfer.

## Policies

- `initial`: frozen VLM on initial images, zero requests.
- `fixed_request`: request the original once when it is not already released.
- `agent`: frozen VLM chooses whether to request; one request maximum.
- `oracle`: original-only evaluation reference with privileged access, separate
  from equal-pool policy comparisons. It is not a mathematical upper bound.

Model: official Qwen3-VL-8B-Instruct, pinned revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, bf16, greedy decoding, SDPA.
The model process receives base64 image pixels and public metadata only; neither
labels nor unreleased asset paths enter its prompts. A process boundary prevents
accidental object access; deploy filesystem isolation before untrusted policies.

Each policy has budget 6, request cost 3. Costs, VLM token counts, actual latency,
parse failures, coverage/risk and paired source-family bootstrap intervals are
recorded. Invalid responses abstain and count as parse failures. Failed requests
still incur their cost. No annotation-generated observation text is used.

## Reproduce on an allocated GPU

Keep one shared data/model directory outside Git. Use an isolated `.venv` with
PyTorch 2.4.0+cu121, Torchvision 0.19.0+cu121, Pillow 12.2.0,
Transformers 4.57.6 and tokenizers 0.22.2. Torchvision must match PyTorch.

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
python -m resolveai.public_data /p62/resolveai-poc --per-label 4
# Download one pinned Qwen snapshot into shared storage (no duplicate cache).
python - <<'MODEL'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    local_dir="/p62/resolveai-poc/model",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
MODEL
python -m resolveai.pilot /p62/resolveai-poc /p62/resolveai-poc/model \
  /p62/resolveai-poc/runs/initial-fixed.jsonl --policies initial fixed_request
python -m resolveai.pilot /p62/resolveai-poc /p62/resolveai-poc/model \
  /p62/resolveai-poc/runs/agent-oracle.jsonl --policies agent oracle
python -m resolveai.report /p62/resolveai-poc/runs/*.jsonl \
  --output /p62/resolveai-poc/runs/summary.json
```

The two result files can run on the two existing Slurm allocations with one
GPU per model process. Do not use the jump host for inference. Result files
resume by case/policy key; only resume under the same data, model and settings.
Keep one manifest/log/result per run and do not save decoded-image duplicates.

Source-image families here consist of one original and its resolution variants.
MVTec does not establish the physical identity of objects across images, so these
are not the physical-object families planned for our later collection. Public
pretraining exposure is uncontrolled. No anomaly mask or oracle-generated region
was supplied to the model; region localization is not evaluated by this pilot.
