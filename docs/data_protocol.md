# Data and annotation protocol v0.1

Each private case contains `family_id`, `claim`, `initial` (image IDs), `evidence`
and `annotation`. An evidence row contains `id`, relative `path`, `source_id`,
`party`, `object` (requestable public name, not a hidden identity label), `time`,
`view`, and `available`. Private annotations contain `verdict` and
`minimal_evidence_sets`, a mapping from each verdict to alternative lists of
original image IDs. A reliable counterexample can suffice for Refuted; Supported
must cover every conjunct. The current evaluator checks image membership only;
annotated region coverage is required before reporting paper grounded accuracy.

Record subclaims, hidden object identity, condition, visibility, region boxes,
uncertainty, annotator decisions, arbitration and annotation protocol version in
private annotations. Collect independent labels from two annotators. Do not force
uncertain identity, time or condition into deterministic labels.

Construct Sufficient, Obtainable, Missing and Unavailable variants for each
family. Missing and Unavailable deliberately share a public failure response.
Decisive material must be obtained by matching a requested view/time/object,
never by selecting evidence based on the truth label. Party/source must be
randomized independently of verdict.

Group split by family, physical object, asset and scene, targeting 70/10/20.
`split_families` uses shared group IDs and connected components; proportions are
expected rather than exact. All variants and claim paraphrases stay together.
Maintain an additional unseen-category evaluation. Bootstrap over families.

Real collection starts with the complete photo pool, then controls release.
Use reversible stains, interchangeable parts or existing damage; log timestamps
and operations. Target 50 real families initially, then 300, with 6–10 views each.
Simulation target: 2,000 families and four availability variants. Never use
hidden state to generate the model's observation text.

Public datasets require source and official-split preservation. WebQA is an
evidence-selection supplement; AVerImaTeC uses closed-pool retrieval; MVTec AD
is an anomaly/localization supplement and different objects must not be relabeled
as before/after pairs. Review licenses before importing data.
