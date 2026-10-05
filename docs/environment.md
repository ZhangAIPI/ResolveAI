# Environment contract v0.2

A `Conversation` retains system instructions, initial pixels, assistant function
calls and tool-role responses throughout an episode. Errors include public action constraints (for example display-pixel bounds) and available tool names. They become actual tool
feedback; they do not invent observations or finish decisions. Invalid output,
budget exhaustion and context limits have bounded recovery. Unfinished episodes
remain unfinished and count as failures in headline accuracy.

| Function | Default cost | Pixel-based behavior |
|---|---:|---|
| inspect | 1 | Existing released image or derived view |
| crop | 1 | Display-coordinate crop with source mapping |
| zoom | 1 | Resample by 0.125–4; enlargement does not restore detail |
| ocr | 2 | RapidOCR text, polygons and confidence |
| compare | 2 | Both released pixel views for the MLLM to compare |
| request_photo | 3 | Exact public object/time/view match; failure also costs 3 |
| assess_quality | 1 | Resolution, exposure, contrast and edge statistics |
| read_metadata | 1 | Submitted source/time, file hash and permitted embedded EXIF |
| ground_text_to_image | 3 | Grounding DINO tiny region proposals |
| ground_image_to_image | 3 | DINOv2-B dense-feature region proposals |
| ground_image_to_text | 2 | CLIP-B/32 ranking of supplied candidate descriptions |
| finish | 0 | Three-way verdict and source-region/time citations |

Grounding DINO does not prove absence when it finds no object. DINO similarity
does not establish same-object identity. CLIP scores are cosine similarities,
not calibrated verdict probabilities; this tool ranks candidates rather than
writing free-form captions. RapidOCR confidence does not authenticate text.
Metadata timestamps are submitted claims unless independently corroborated.
DINOv2 is the initial accessible frozen matching backend; DINOv3 remains a planned
provider substitution, not a measured component. Backends advertise availability
and can be replaced without exposing annotations to perception models.

Derived views keep operation history in memory and render from released pixels.
A downsampled view followed by enlargement stays degraded. Tool outputs identify
original `image_id`, derived `view_id`, original dimensions, source rectangle,
display dimensions and time. Crop inputs use display coordinates; finish uses
integer original coordinates. Derived views never count as independent evidence.
Failed requests use the same public response for Missing and Unavailable.

## World simulation

`EvidenceWorld` stores an immutable private state and evidence library. Requests
release exact matching material; forks clone budgets, releases, derived views and
conversation history while sharing the unchanged world. `render_scene.py` uses
Blender Cycles to render different cameras and times from explicitly constructed
state. It does not fabricate observation text from annotations. The first scene
has a procedural wooden table, a rendered scratch, three views and two times.
Its annotation status is unreviewed; it is a renderer/transition test.

The phase-one world model is state-driven rendering plus evidence release.
Generative image editing, super-resolution and hypothetical damage are excluded
from verified evidence. A future generative simulator must keep imagined outputs
separate from submitted photographs and cannot silently become supporting proof.
The current public adapter releases originals from a closed pool; it never claims
to take a new camera view of a public object.

## Training boundary

The model worker receives only public messages and tool schemas. The private
controller executes validated calls and evaluates after finish. `server` exposes
no evaluation or annotation endpoint. Default policy sessions cannot fork;
`--allow-forks` enables a bounded trusted-controller branch interface.
An account/container boundary is required for untrusted policies with arbitrary
code execution because process separation alone does not restrict filesystem reads.

`TrajectoryStore` writes JSONL and deduplicated PNG blobs. `sft_example` selects
completed, correct, grounded, error-free traces. By default it also requires
reviewed region/time annotations. `preference_pair` compares completed sibling
branches with identical public prefixes, schemas and world fingerprints; it
exports only the first differing assistant action, not forced tool observations.
Correct grounded outcomes take priority over request/tool cost. Proxy bootstrap
is explicitly opt-in and must not be described as independently grounded training.

Provider sources: [RapidOCR](https://github.com/RapidAI/RapidOCR),
[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny),
[DINOv2](https://huggingface.co/facebook/dinov2-base),
[CLIP](https://huggingface.co/openai/clip-vit-base-patch32),
[Transformers tool conversations](https://huggingface.co/docs/transformers/en/chat_extras).
