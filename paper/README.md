# ProofAgent manuscript

English CVPR 2027 working draft, limited to the abstract, introduction, related work, task, benchmark construction and evaluation methodology. No experiments/results section, completed-score table or training-gain claim is included. ProofAgent preserves the title of the supplied Overleaf project; the codebase remains ResolveAI.

The story follows one question: can a model obtain enough object-specific evidence, then stop with a justified decision? The protocol distinguishes physical truth, obtainable evidence and released evidence. The four availability variants are intended construction conditions, not automatically certified semantic labels.

## Files and build

- main.tex: manuscript; official review mode hides the existing author field.
- references.bib: primary-source references.
- figures/: two original vector diagrams and three unchanged CO3D example photographs.
- cvpr.sty and ieeenat_fullname.bst: unmodified official CVPR 2027 files; archive source/hash in template-source.json.

Compile main.tex with pdfLaTeX + BibTeX (Overleaf) or tectonic. Keep build outputs outside Git. Do not edit the official style to fit content.

## Writing and figure decisions

- MMMU: define the capability before listing data or models.
- BLINK: anchor the task with a concrete visual example and an explicit capability decomposition.
- MMStar: explain visual dependence and shortcut controls as part of the benchmark design.
- AVerImaTeC: make independent evidence validation central.
- ActiView / OpenEQA: acknowledge active acquisition as an existing evaluation target.
- VisualToolBench / VTC-Bench: distinguish the evaluation target from the tool inventory.
- CounterCredit: separate executing a call from using informative returned evidence.

The figures are new diagrams for this project, not copies of those papers. Figure 1 combines actual public photographs with the observation/action/decision loop and the four controlled access conditions. Figure 2 explains the asymmetry between complete support and decisive refutation. Neither figure assigns a reviewed verdict to the illustrative photographs.

## Evidence status

The 478 families and 1,912 configurations are construction counts. Source labels, unknown identity, optional human marks and fully reviewed sufficiency are kept distinct. Before any result section is added, complete the planned admission process and retain source grouping in uncertainty estimates.

The draft does not claim that simulation training, generative world models, real before/after verification or transfer to new self-collected cases has been demonstrated.

## Attribution

CO3D photographs: Jeremy Reizenstein et al., Common Objects in 3D, ICCV 2021; CC BY-NC 4.0. See figures/provenance.json for sequence, image IDs, hashes and source locations. Files are copied without enhancement, retouching or generative edits; LaTeX changes display size only. Original diagram code and text are AI-assisted, disclosed in the manuscript.
