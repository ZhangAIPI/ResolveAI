# Evidence Before Verdicts

Evidence Before Verdicts: Benchmarking Active Visual Verification. The benchmark is ResolveAI.

## Main paper and supplement

The main paper ends at the benchmark design and has three sections: Introduction, Related Work, and The ResolveAI Benchmark. It follows one story: recognizing damage is different from obtaining enough evidence to verify a particular object claim. The main argument retains the access-aware target, sufficient-evidence rationale, construction/validation design and paired evaluation; implementation remains supplementary. The same claim is tested under different evidence-access conditions.

- main.tex: concise narrative, core design and evaluation rationale.
- supplementary.tex: entry point for the separate technical appendix.
- appendix.tex: formal evidence audit, annotation checklist, tool contract, execution settings, metric equations and controlled action comparisons.
- figures/: original diagrams and unchanged CO3D photographs. The logical certificate diagram is used only in the supplement.
- references.bib: primary-source references shared by both documents.
- cvpr.sty and ieeenat_fullname.bst: unmodified official CVPR 2027 files; provenance is in template-source.json.

The current methods-only main text occupies four pages before the reference/disclosure page. It retains three main sections; future experiments follow the benchmark section. The technical supplement remains separate.

Compile main.tex and supplementary.tex separately using pdfLaTeX + BibTeX (Overleaf) or tectonic. Keep generated files outside Git. The main paper contains no experiment/result section or training-gain claim.

## Page allocation checked against published CVPR PDFs

Page numbers below are relative to the first page of the official CVF two-column PDF, including title and figures. They identify section starts; floated result tables can appear before their section heading.

| Published paper | End of benchmark/method discussion and start of evaluation |
| --- | --- |
| [MMMU, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf) | Section 3 continues into page 5; Section 4, Experiments, starts on page 5. |
| [VSI-Bench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.pdf) | Benchmark construction continues into page 4; Section 4, Evaluation on VSI-Bench, starts on page 4. |
| [OpenEQA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.pdf) | Benchmark/evaluation design continues into page 5; Section 3, EQA Agents, starts on page 5; Section 4, Experimental Results, starts on page 6. |
| [MMVP, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) | Section 2.3, Benchmark Results, starts on page 3; a later model-improvement method starts on page 6. This hybrid structure is not a template for a single methods-before-results boundary. |

These examples do not establish a mandatory page count. They show why section count and page count should not be used as substitutes for a complete argument. ResolveAI keeps three main sections and gives the core design enough space before future experiments, while leaving tool schemas, costs, exact matching and execution parameters in the supplement.

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
