# Evidence Before Verdicts

Evidence Before Verdicts: Benchmarking Active Visual Verification. The benchmark is ResolveAI.

## Main paper and supplement

The main paper now has four sections: Introduction, Related Work, The ResolveAI Benchmark, and Initial Benchmark Results. It follows one question: can the agent obtain the observations needed to settle a particular object claim? The main argument retains the access-aware target, sufficient-evidence rationale, construction/validation design and paired evaluation; implementation remains supplementary. Five numbered equations carry the central definitions; the gap indicator is inline. The same claim is tested under different evidence-access conditions.

- main.tex: object-verification motivation, sufficiency logic, evidence gap, grounded outcome, acquisition–conversion decomposition and matched action benefit. Strict family success is defined in the supplement.
- supplementary.tex: entry point for the separate technical appendix.
- appendix.tex: formal evidence audit, annotation checklist, tool contract, execution settings, metric equations and controlled action comparisons.
- figures/: an overview, real-image action examples and a matched-branch diagnostic; original vector code and unchanged CO3D/MVTec inputs. The detailed sufficient-evidence diagram is supplementary.
- references.bib: primary-source references shared by both documents.
- cvpr.sty and ieeenat_fullname.bst: unmodified official CVPR 2027 files; provenance is in template-source.json.

The benchmark discussion continues onto page 6, where Section 4 begins; floated Table 1 appears on page 5. The technical supplement remains separate.

Compile main.tex and supplementary.tex separately using pdfLaTeX + BibTeX (Overleaf) or tectonic. Keep generated files outside Git. Table 1 reports the fixed 209-family measured snapshot in docs/results/cvpr_20261102_candidate_scores.json. Generate its TeX with scripts/paper_result_table.py. It measures unreviewed source-label agreement, not grounded accuracy. Table 1 includes seven completed base models and reports correct-answer percentages with and without tools. Sample counts stay in the evaluation text; the task footnote is removed. The final-response breakdown is supplementary in tables/agent_responses.tex. Table 2 reports completed identity/acquisition SFT: identity agreement improves, condition agreement decreases, and the pooled tool-agent score falls from 82.3% to 78.5%. Both gains and losses are retained. Its source data are in docs/results/cvpr_sft_candidate_scores.json; tables/sft_results.tex is included in the main paper.

## Page allocation checked against published CVPR PDFs

Page numbers below are relative to the first page of the official CVF two-column PDF, including title and figures. They identify section starts; floated result tables can appear before their section heading.

| Published paper | End of benchmark/method discussion and start of evaluation |
| --- | --- |
| [MMMU, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf) | Section 3 continues into page 5; Section 4, Experiments, starts on page 5. |
| [VSI-Bench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.pdf) | Benchmark construction continues into page 4; Section 4, Evaluation on VSI-Bench, starts on page 4. |
| [OpenEQA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.pdf) | Benchmark/evaluation design continues into page 5; Section 3, EQA Agents, starts on page 5; Section 4, Experimental Results, starts on page 6. |
| [MMVP, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) | Section 2.3, Benchmark Results, starts on page 3; a later model-improvement method starts on page 6. This hybrid structure is not a template for a single methods-before-results boundary. |

These examples do not establish a mandatory page count. They show why section count and page count should not be used as substitutes for a complete argument. ResolveAI gives the core design enough space before the experimental comparison, while leaving tool schemas, costs, exact matching and execution parameters in the supplement.

## Reference-paper structure

The writing uses concrete benchmark examples and a capability question to organize the main text, with reproducibility details moved to the supplement.

| Primary reference | Structural lesson applied here |
| --- | --- |
| [MMMU, CVPR 2024](https://arxiv.org/abs/2311.16502) | Motivation and target capability lead into a compact benchmark section; detailed annotation rules go to the appendix. |
| [MMVP, CVPR 2024](https://arxiv.org/abs/2401.06209) | A concrete visual ambiguity motivates the test and its diagnostic value; implementation settings stay outside the main argument. |
| [OpenEQA, CVPR 2024](https://open-eqa.github.io/) | Connect the observation setting, data and evaluation in one benchmark narrative; agent function signatures and collection interfaces are supplementary. |
| [VSI-Bench, CVPR 2025](https://arxiv.org/abs/2412.14171) | Organize the benchmark around a capability question; put construction parameters and evaluation details in the technical supplement. |

Two additional writing references are [VidComposition (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_VidComposition_Can_MLLMs_Analyze_Compositions_in_Compiled_Videos_CVPR_2025_paper.html) and [MMPerspective (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6f17e48da556b38db2906acd7594a0ff-Abstract-Datasets_and_Benchmarks_Track.html), by Yolo Tang and collaborators. Their capability organization, concrete task examples and separation of construction from diagnostic analysis inform the presentation. ResolveAI's diagrams, wording and evaluation definitions are original.

The four benchmark subsections follow the argument from the controlled variable to evidence sufficiency, case construction and evaluation:

1. Evidence access as a controlled variable.
2. From observations to a justified verdict.
3. Building cases with an evidence gap.
4. Separating acquisition from judgment.

A central distinction is that access changes can require different justified verdicts for the same physical truth. Strict family success checks this adaptation jointly (supplement); it does not reward repeating an answer across versions. The equations define the evaluation; they do not report empirical gains. Exact tool schemas and matching rules remain supplementary.

Related-work comparisons additionally cover BLINK, ActiView, AVerImaTeC, AbstainEQA, visual tool benchmarks and CounterCredit. Wording and figure code are original; no prior figures are copied.

## Complete simulated reviews and revisions

[Review records and author responses](reviews.zh-CN.md) cover three full manuscript reviews, including strengths, weaknesses, questions, limitations, recommendation, confidence and changes. These are author-side self-reviews, not independent conference reviews. [Five follow-up rounds, rebuttals and revisions](rebuttals.zh-CN.md) cover Rounds 4–8, including reachability, trajectory diagnosis, alternative evidence, novelty and final prose/layout checks. Data validation and experimental evidence remain open requirements.

## Evidence status and attribution

The 478 families and 1,912 access versions are construction counts, with source reuse tracked. Independent sufficiency admission remains required. The short interface study is separate from evidence-sufficiency annotation.

Public photographs currently support condition and identity candidates. Genuine before/after verification requires paired observations; no simulation-training or transfer gain is claimed.

MVTec photographs: Paul Bergmann et al., MVTec AD, CVPR 2019, CC BY-NC-SA 4.0. The capsule preview and original are copied unchanged from the recorded candidate assets. The orange-bordered detail is a native LaTeX viewport, explicitly marked as a crop; source coordinates, hashes and the upstream revision are recorded in figures/provenance.json. The example is not a semantically admitted test label.

CO3D photographs: Jeremy Reizenstein et al., Common Objects in 3D, ICCV 2021, CC BY-NC 4.0. Sequence, image IDs and unchanged-file hashes are in figures/provenance.json. AI assistance is disclosed in the manuscript.

The argument also draws a writing lesson from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385): identify a specific obstacle, motivate a controlled construction, and specify comparisons that could support or reject the explanation. No prose, equations, figures or empirical claims are copied from that paper.

A subsequent prose rewrite removes reviewer-facing imperatives and checklist paragraphs from the main paper. Construction requirements are explained through their scientific rationale; annotation procedures and unfinished validation work remain in the supplement and review record. The main text retains the candidate-data status and does not imply completed validation.

## Result presentation

Table 1 now separates task performance from interaction outcomes. Its centered first panel reports condition (153 cases), cross-view identity (56), and pooled results (209), each with Answer only and With tools. Its second panel accounts for every agent attempt as a label match, opposite label, abstention, or missing verdict. All-image direct decisions remain in the supplement. The tables use natural column widths and fixed small column gaps, without stretching their rules across the page. Difficulty groups are omitted because no independently established difficulty labels exist.

This follows the task-first presentation in [MMMU Table 2](https://arxiv.org/html/2311.16502v3) and [BLINK's task results](https://arxiv.org/html/2404.12390v2): expose capability differences before an aggregate, state the sample counts, and separate diagnostic outcomes from the main score. No figures, wording, or scores are copied.
