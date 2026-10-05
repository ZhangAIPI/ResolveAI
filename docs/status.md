# Execution status — 2026-10-04

Repository: https://github.com/ZhangAIPI/ResolveAI
Checkout: `/home/cxu-serve/p62/zzh136/experiments/ResolveAI`.

Existing Slurm allocations 24679 and 24689 each provide 2 RTX A6000 GPUs
(48 GB per GPU). Both are on Slurm node `sclera`, whose hostname is
`node2x32b.cs.rochester.edu`; these are two allocations on one host, not two hosts.
Use `srun --jobid=JOB --overlap -N1 -n1 -c1 COMMAND` for bounded work inside
an allocation. Confirm occupancy before running new models. Never cancel or
replace existing allocations without coordinating their other experiments.

p62 has approximately 62 TB available at initialization; p61 is nearly full.
Keep model weights and image pools in shared p62 storage, outside Git.
The proposed 2×80 GB configuration is not the actual allocation. Measure memory
before selecting image resolution, batch size, student or teacher loading.

First milestone: freeze schema/annotation protocol, implement controlled renders,
collect 50 real families, validate visual perception with two models, then
establish zero-shot and rule baselines. Later: static/simulator SFT, equal-budget
branch preferences, family-level intervals, real transfer and ablations.
Model choices in the supplied plan are candidates, not tested dependencies.
Real-photo collection and two independent annotators require human resources.
Conference dates in the supplied plan have not been independently verified here.

## Validation

Python environment: PyTorch 2.4.0+cu121 and Pillow 12.2.0 already available.
Blender was not found in the current executable path. No model weights or
large data files have been downloaded. `docs/collection.csv` is an empty real
collection inventory template, not fabricated evidence.
