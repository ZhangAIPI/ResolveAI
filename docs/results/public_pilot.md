# Public-image pilot results — 2026-10-04

Evidence release helped on this small test, while the frozen acquisition agent
remained less effective than a fixed request. This is a resolution-release
proof of concept, not validation of cross-view reasoning or simulation training.

24 unique source images from the official MVTec AD test split, balanced
across three categories and normal/defective labels, produce 96 availability
variants. Four policies completed 384 unique episodes and 450 model calls.
No parse failures or invalid citations occurred. All inference ran on the two
existing GPU allocations, using Qwen3-VL-8B-Instruct and 16.51 GB peak allocated
GPU memory per worker. Data and model remain outside Git on p62.

## Cases where the original was obtainable

Accuracy counts only a correct independent dataset label; abstention is not
counted as correct. Coverage counts determinate decisions. Risk is the error
rate among those determinate decisions.

| Policy | Correct / 24 | Coverage | Risk among decisions | Mean requests | Mean model latency |
|---|---:|---:|---:|---:|---:|
| Initial images | 0/24 (0.0%) | 12.5% | 100.0% | 0.00 | 1.15 s |
| Fixed request | 12/24 (50.0%) | 62.5% | 20.0% | 1.00 | 1.25 s |
| Prompted agent | 10/24 (41.7%) | 62.5% | 33.3% | 0.92 | 2.29 s |
| Original-only reference (privileged) | 12/24 (50.0%) | 66.7% | 25.0% | 0.00 | 1.23 s |

The fixed-request gain over initial-only is **+50.0 percentage points**
(paired source-family bootstrap 95% CI: +29.2 to +70.8). The prompted-agent
gain is **+41.7 points** (CI: +20.8 to +62.5). Intervals use 5,000 resamples;
the small, deliberately balanced sample is descriptive rather than a population estimate.

Initial images led to 21 abstentions and three incorrect determinate decisions.
The fixed policy made 15 determinate decisions, of which 12 were correct; the
agent also made 15, of which 10 were correct. The agent requested 22/24 times
and used about twice as many model calls as the fixed policy. Its proxy
unsupported-decision rate was 20.8%, versus 12.5% for the fixed policy. The
pilot therefore does not establish improved efficiency or reduced unsupported
decisions for the current prompted agent.

## Availability and perception

| Version | Initial correct | Fixed correct | Agent correct |
|---|---:|---:|---:|
| Sufficient | 12/24 | 12/24 | 12/24 |
| Obtainable | 0/24 | 12/24 | 10/24 |
| Missing | 0/24 | 0/24 | 0/24 |
| Unavailable | 0/24 | 0/24 | 0/24 |

When a request failed, fixed and agent policies abstained on 22/24 cases
in each failure version. They still incurred request costs; two wrong
determinate decisions remained. Missing and Unavailable share the same public
response so the agent cannot infer whether hidden material exists.

Original-only reference accuracy is 12/24. On obtainable cases the fixed
policy correctly recognizes 9/12 defective images but only 3/12 normal images;
abstention on normal images is a major bottleneck. More pixels alone do not
solve reliable defect verification. The privileged reference is reported
separately and is not an equal-evidence policy or mathematical upper bound.

## What this verifies and what remains

The environment can transmit real image pixels to a separate VLM process,
release matching material, enforce budgets, preserve source provenance,
and produce measurable changes in decisions without annotation-generated
observations. Independent dataset labels provide the evaluation target.

The 48-pixel preview is an artificial controlled degradation. Original and
preview are the same source, not independent photos. Minimal evidence is an
original-image citation proxy, not human semantic sufficiency annotation.
Public pretraining exposure is uncontrolled. Physical identity, before/after
state change, localization, branch-trained policies and transfer to our own
photos are not validated here.

Before scaling self-collection, broaden the untouched public evaluation and
address normal-case abstention, incorrect stopping and unnecessary requests.
Our later physical-object families must include genuine multi-view photos and
independent region/evidence annotation; public-image gains do not replace that.

[Reproduction protocol](../public_pilot.md), [machine-readable summary](public_pilot.json),
[dataset/model/runtime manifests](public_pilot_manifest.json). Raw episode traces
and two logs are retained in the p62 directory listed in the manifest; no
image or model duplicates are checked into Git.

Dataset: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
via [Voxel51 mirror](https://huggingface.co/datasets/Voxel51/mvtec-ad).
Images/derived dataset retain CC BY-NC-SA 4.0; original source license governs.
Model: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
