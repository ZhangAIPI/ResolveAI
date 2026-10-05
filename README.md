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

This is infrastructure, not an evaluated model or a dataset. No real-photo or
synthetic research results have been produced. Region-level semantic sufficiency,
confidence intervals, Blender rendering, VLM inference and policy training remain
to be implemented. Fixture tests use generated pixels only for software validation.

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
