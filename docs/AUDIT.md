# Audit of public model merges on Hugging Face

Run on 21 November 2026 with `audit_hub.py` (text only, no model weights; Kaggle CPU notebook).
Sample: the 1,500 most-downloaded models tagged `mergekit`. Quantized copies (GGUF, EXL2, ...)
were followed back to the merge they were made from, giving **1,219 distinct merges**.
Recipes were found for **937 (77%)**; **475** use a task-vector method (TIES, DARE, task
arithmetic, DELLA, SCE, Model Stock). Family trees come from model-card metadata, the merges'
own recipes, and two name-based rules for links that cards leave out (each noted per row).

## 1. Hidden duplicates: a model merged with something that already contains it

| | Merges |
|---|---|
| Checkable (2+ ingredients, complete family information) | 371 |
| One ingredient is an ancestor of another ("lineage loop") | **98 (26.4%)** |
| ...of which the repeated model is hidden inside a merged ingredient | **86 (23.2%)** |
| ...of which a model is merged with its own fine-tune (often deliberate) | 12 |
| Hidden duplicates among task-vector merges | 60 of 281 (21.4%) |

In a hidden duplicate, model A's contribution enters the merge twice: once directly and once
inside another ingredient that already contains A. Choosing the right base does not remove it.
Most-downloaded examples: maxcurrent/peagle-9b (9,182 downloads), kukedlc/neuralsynthesis-7b-v0.3
(8,850), novaciano/sentimental_sex-3.2-1b (1,841), mantis2024/dirty-muse-writer-v01 (1,317).
(Model Kinship, arXiv 2410.12613, uses "inbreeding" as a metaphor for merging very similar
models; here the duplicates are found and counted from the family tree.)

## 2. Is the declared `base_model` the closest common ancestor?

Only 54 of the 475 task-vector merges could be judged: the rest lack a published recipe,
name local folders that cannot be matched to Hub models, or have gaps in their family tree.

| Declared base | Merges |
|---|---|
| The closest common ancestor (correct) | **24 (44%)** |
| A relative of the ingredients (sibling, cousin or descendant of their common ancestor) | **13 (24%)** |
| An older ancestor (the double counting of our experiments) | **1 (2%)** |
| Unclear from the cards | 16 (30%) |

For plain task arithmetic, merging from base B instead of the common ancestor C changes the
result by **(sum of weights − 1) × (C − B)**. It vanishes when the base is right or the weights
are normalised. mergekit normalises TIES and DELLA by default and SCE always
(`merge_methods/registry.py`, checked 21 Nov 2026); task arithmetic, DARE-TIES, DARE-linear,
breadcrumbs and DELLA-linear are not normalised by default.

Of the 14 merges with a wrong base, **9 are neutralised by normalisation or weights summing to
1**, 2 over-weight the difference, 2 under-weight it, and 1 uses Model Stock. Normalisation is
what protects most public merges; the Base Checker flags the cases where it is off.

## 3. Coverage is itself a finding

Model cards rarely record lineage well enough to check a merge from its metadata: 282 merges
publish no recipe, 75 recipes name local folders, and 193 have an ingredient whose card lists
no parent. Model weights cannot leave this out, which is why this project recovers the family
tree from the weights (`lineage.py`, `base_checker.py`).

## Method notes and limits
- Card metadata is self-reported; the name-based rules (official instruct model → its base;
  parentless re-upload → the official model of the same name) are kept only when the target
  model exists, and are noted per row.
- A flagged merge is not proven broken: the impact depends on the method, weights and
  normalisation. The accuracy cost of hidden duplicates was not measured.
- The sample is the most-downloaded mergekit models and is dominated by creative-writing and
  role-play merges.
- Full per-merge results: `docs/audit_results.csv`.
