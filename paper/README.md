# Can Vision-Language Models Find the Evidence They Need?

ResolveAI studies how vision-language models inspect submitted photographs, request additional evidence and decide when to answer.

[Main manuscript](main.tex) · [Supplementary material](supplementary.tex) · [Overleaf](https://www.overleaf.com/project/6ae80679a4c7a64df04ac5b8) · [Current experiments and remaining questions](../docs/experiments/reviewer_evidence_search.zh-CN.md)

## Current manuscript

The main text occupies eight CVPR pages, with references on the following page. It contains three figures and four tables. Experimental Setup, Main Results, Post-training Results and Behavior Analysis cover the public-photo study; Photo-request Controls and Two-step Evidence Search isolate image content and observation-dependent view selection.

- **Figure 1:** real-image examples of identity matching and condition-detail requests.
- **Figure 2:** data sources, matched photograph-access settings and the tool environment.
- **Figure 3:** paired procedural cases with identical initial photographs but different required second views.
- **Table 1:** ten open-weight and five API models, plus the additional Luna Decisions interface. Sufficient compares answers with and without tools; Obtainable reports condition/identity accuracy, new images, completion and interaction cost.
- **Table 2:** Base, SFT, RL and SFT+RL for three backbones, with training results kept separate from the base-model comparison.
- **Table 3:** photograph requests, genuinely new displayed pixels and stopping after requests fail.
- **Table 4:** the held-out two-step search study: a public-RGB program with five request policies, and MLLM evidence selection under a shared finite-choice interface.

The former low-information result figure remains removed. Its 204 returned-image content trials are analyzed in the text and supplement. Figure 3 instead displays actual rendered images and the two-step decision. Full configurations, checkpoint comparisons, failure accounting and additional results are supplementary.

## Data and scores

The candidate pool contains 478 families: 188 condition cases, 64 labelled identity controls and 226 unlabelled identity candidates. The fixed quantitative cohort contains 209 labelled families: 153 condition and 56 identity cases. All sibling access versions retain the same claim and object content.

Public-photo accuracy measures agreement with source labels. Those cases admit one-request detail restoration and have no independent sufficiency adjudication. A separate procedural suite adds 132 families in 66 scene pairs; 16 families are reserved for program development and 116 for held-out comparisons. Its finite-world certificates and visible-region masks support exact construction-specific grounded scoring. These checks do not substitute for human review of the public photographs or establish real before/after damage verification.

## Result sources

| Comparison | Public result record | Main table source |
| --- | --- | --- |
| Base models across access settings | [Complete Obtainable results](../docs/results/obtainable_results.json) and [Sufficient results](../docs/results/cvpr_20261102_candidate_scores.json) | [benchmark_results.tex](tables/benchmark_results.tex) |
| Post-training | [All twelve policies](../docs/results/adaptation_obtainable_results.json) | [sft_results.tex](tables/sft_results.tex) |
| Requests and stopping | [Behavior counts](../docs/results/paper_search_behavior.json) | [search_behavior.tex](tables/search_behavior.tex) |
| Request controls | [Scheduled requests](../docs/results/cvpr_20261105_evidence_controls.json), [access controls](../docs/results/cvpr_20261105_evidence_interventions.json), [paired changes](../docs/results/cvpr_20261106_evidence_control_pairs.json) | Supplementary material |
| Returned-image content | [All 204 trials](../docs/results/cvpr_20261106_photo_content_probe.json) | Text and supplement |
| Two-step view selection | [Program feasibility control](../docs/results/multistep_search_program_heldout.json), [assisted model diagnostic](../docs/results/multistep_search_assisted_heldout.json), [native-call pilot](../docs/results/multistep_search_native_pilot.json) | [multistep_search.tex](tables/multistep_search.tex) |

Public reports retain coverage, source hashes and failure denominators. Weights, photographs and raw trajectories stay outside Git. Existing completed episodes are reused when refreshing summaries.

## Build and attribution

Compile main.tex and supplementary.tex separately with pdfLaTeX/BibTeX or Tectonic, placing generated files outside the repository. The official CVPR style files and their provenance are recorded in [template-source.json](template-source.json).

Photographs come from MVTec AD and CO3D; image IDs, source revisions, hashes and crop coordinates are recorded in [figures/provenance.json](figures/provenance.json). Figures retain the source pixels. The manuscript includes its AI-assistance statement.

[Reviews](reviews.zh-CN.md) and [author responses](rebuttals.zh-CN.md) preserve the earlier manuscript self-reviews. They describe the versions reviewed at those dates, rather than new independent conference decisions.
