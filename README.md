# ResolveAI

Interactive multi-image visual fact verification. The research question is whether
branchable simulation training improves evidence acquisition on controlled real
photographs. Verdicts are `Supported`, `Refuted`, or `Need more evidence`;
visual state changes do not establish liability.

## Current implementation

The private environment releases existing photographs by exact object/time/view
queries. Inspect, crop and compare return image pixels with source provenance;
perception belongs to the visual model. Crops retain original coordinates and
never count as new independent evidence. Requests do not alter scene state.
Independent branches retain their own released images and budgets. An evaluator
checks cited image sets against independently annotated sufficient evidence.
Family splits keep shared physical objects/assets in one partition.

The public-image proof of concept uses a frozen Qwen3-VL-8B model and a pinned
24-image MVTec test subset. [Measured results](docs/results/public_pilot.md)
show 50% correct decisions with fixed requests versus 0% from the initial
preview in obtainable cases; the prompted agent reaches 41.7%. This is a small
resolution-release pilot. See [pilot protocol](docs/public_pilot.md) for paired
policies, independent labels, evidence-release conditions and scope limits.
Self-collected cases, region-level semantic sufficiency, Blender rendering and
policy training remain future work. Fixture tests use generated pixels only for
software validation; the public pilot uses original dataset pixels.

## Run

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m resolveai.server /private/case.json /private/assets --budget 12
```

The server emits its initial observation followed by one response per JSON action.
`image_png` is base64 PNG. Send `inspect` with `image_id`, `crop` with `image_id`
and integer pixel `bbox`, `compare` with two `image_ids`, or `request_photo` with
`query: {object, time, view}`. `finish` requires a verdict and citations containing
`image_id` and source-image `bbox`. `fork` takes a unique `new_branch`; subsequent
commands select it using `branch`. Annotation/evaluation never enters the policy
response. Run policies without access to private asset directories; a Python
process boundary alone does not provide filesystem isolation.

See [data protocol](docs/data_protocol.md) and [execution status](docs/status.md).
Keep datasets, checkpoints, caches and logs outside Git. Retain one configuration,
summary and final checkpoint per experiment; avoid duplicate image caches.
