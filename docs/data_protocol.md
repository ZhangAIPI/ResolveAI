# Data and annotation protocol v0.2

Each private case contains `family_id`, `claim`, `initial` image IDs, `evidence`,
public `request_options` and private `annotation`. Evidence rows contain `id`,
relative `path`, `source_id`, `party`, requestable `object`, `time`, `view` and
`available`. Optional `source_size`/`source_bbox` map degraded images to source
coordinates. Never derive public request options from hidden evidence availability.

`minimal_evidence_sets` maps verdicts to alternative conjunctions. Region/time
requirements use `{"image_id":"before","bbox":[0,0,100,100],"time":"before"}`.
The evaluator checks source IDs, valid observed bounds, source time and union
coverage of each required region; overlapping duplicate citations cannot inflate
coverage. Support covers every conjunct; a decisive counterexample may suffice
for Refuted. Legacy lists of image IDs remain a clearly marked quality proxy.

Record subclaims, object identity, condition, visibility, uncertainty, two annotator
decisions, arbitration, protocol version and `status: reviewed` in private
annotations. Uncertain identity, time or state must not be forced into deterministic
labels. Geometric render boxes alone are not independently validated sufficiency.
Default training filters reject unreviewed annotations and image-set proxies.

Each family has Sufficient, Obtainable, Missing and Unavailable versions. Requests
match public object/time/view, never truth labels or the most decisive image.
Source/party should be independently randomized where the source protocol permits.
All crops retain the same original evidence identity and do not become new proof.

Use connected-component family, physical-object, asset and scene groups targeting
70/10/20. `split_families` keeps shared group IDs together; proportions are expected,
not exact. Variants and claim paraphrases stay in one split. Maintain an unseen
category test and bootstrap by the available independent source unit. Public official
splits take precedence over a new split; freeze public test samples before training.

No new self-collected data is assumed in the current project scope. Public datasets
must preserve source, license and original task semantics. MVTec AD provides anomaly
and region labels, not same-object before/after pairs. WebQA can test multi-source
evidence selection. AVerImaTeC can support closed-pool statement verification;
releasing existing evidence is not new photography. Review and independently
annotate any adaptation that extends the original task. State-driven Blender
simulation supplies controlled identity, camera and time; keep simulation and public
photograph results separate and avoid hidden-state-generated observation text.
