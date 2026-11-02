# Evidence Before Verdicts

Evidence Before Verdicts: Benchmarking Active Visual Verification. The benchmark is ResolveAI.

## Main paper and supplement

The main paper ends at the benchmark design and has three sections: Introduction, Related Work, and The ResolveAI Benchmark. It follows one story: recognizing damage is different from obtaining enough evidence to verify a particular object claim. The same claim is tested under different evidence-access conditions.

- main.tex: concise narrative, core design and evaluation rationale.
- supplementary.tex: entry point for the separate technical appendix.
- appendix.tex: formal evidence audit, annotation checklist, tool contract, execution settings, metric equations and controlled action comparisons.
- figures/: original diagrams and unchanged CO3D photographs. The logical certificate diagram is used only in the supplement.
- references.bib: primary-source references shared by both documents.
- cvpr.sty and ieeenat_fullname.bst: unmodified official CVPR 2027 files; provenance is in template-source.json.

Compile main.tex and supplementary.tex separately using pdfLaTeX + BibTeX (Overleaf) or tectonic. Keep generated files outside Git. The main paper contains no experiment/result section or training-gain claim.

## Reference-paper structure

The writing uses concrete benchmark examples and a capability question to organize the main text, with reproducibility details moved to the supplement.

| Primary reference | Structural lesson applied here |
| --- | --- |
| [MMMU, CVPR 2024](https://arxiv.org/abs/2311.16502) | Motivation and target capability lead into a compact benchmark section; detailed annotation rules go to the appendix. |
| [MMVP, CVPR 2024](https://arxiv.org/abs/2401.06209) | A concrete visual ambiguity motivates the test and its diagnostic value; implementation settings stay outside the main argument. |
| [OpenEQA, CVPR 2024](https://open-eqa.github.io/) | Connect the observation setting, data and evaluation in one benchmark narrative; agent function signatures and collection interfaces are supplementary. |
| [VSI-Bench, CVPR 2025](https://arxiv.org/abs/2412.14171) | Organize the benchmark around a capability question; put construction parameters and evaluation details in the technical supplement. |

Related-work comparisons additionally cover BLINK, ActiView, AVerImaTeC, AbstainEQA, visual tool benchmarks and CounterCredit. Wording and figure code are original; no prior figures are copied.

## Evidence status and attribution

The 478 families and 1,912 access versions are construction counts, with source reuse tracked. Independent sufficiency admission remains required. The short interface study is separate from certificate annotation.

Public photographs currently support condition and identity candidates. Genuine before/after verification requires paired observations; no simulation-training or transfer gain is claimed.

CO3D photographs: Jeremy Reizenstein et al., Common Objects in 3D, ICCV 2021, CC BY-NC 4.0. Sequence, image IDs and unchanged-file hashes are in figures/provenance.json. AI assistance is disclosed in the manuscript.
