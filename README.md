# Lineage-Aware Model Merging for Large Language Models

**Live demo:** https://huggingface.co/spaces/papachengo/base-checker

Model merging combines several fine-tuned models into one, with no training and no data,
by adding up each model's change from a **base model**. Every merging method and tool
(including [mergekit](https://github.com/arcee-ai/mergekit)) takes that base as given:
the user types `base_model` by hand, and nothing checks it.

We found that this one line can decide whether a merge works. Three models fine-tuned from
Qwen2.5-1.5B-Instruct, merged with Task Arithmetic:

| `base_model` | GSM8K | MBPP |
|---|---|---|
| Qwen2.5-1.5B-Instruct (their closest common ancestor) | **68.0** | **48.0** |
| Qwen2.5-1.5B (the usual choice) | **6.5** | **0.0** |

Same models, same method, same settings. With the wrong base, every model's change also
contains Instruct's own training, and adding three of them counts that training three times.

This repository recovers the **family tree** of models from their weights alone, picks the
closest common ancestor as the base, and detects unsafe merges **before** merging.

B.Tech (AI & Data Science) major project, Sikkim Manipal Institute of Technology, 2026.
Gourabdipta Ghosh and Arko Chakraborty. Guide: Mr. Suraj Dey.

---

## Base Checker (for mergekit configs)

```bash
python base_checker.py --config examples/wrong_base_task_arithmetic.yml \
    --candidates Qwen/Qwen2.5-1.5B-Instruct --out fixed.yml
```

It reads the weights of every model involved (no GPU, no data), builds the family tree and
prints one of:

| Verdict | Meaning |
|---|---|
| `OK` | `base_model` is the closest common ancestor |
| `WRONG_BASE` | the models share a closer ancestor and the method adds task vectors with total weight > 1, so that ancestor's update is counted several times; `--out` writes a corrected config |
| `WARNING` | a closer ancestor exists, but the method is less sensitive (TIES-style trimming, or weights summing to 1) |
| `TOO_FAR` | even the best base is more than 10% away from a model: merging is unsafe from any base |
| `INFO` | the method (linear, slerp) does not use the base for task vectors |

Options: `--discover` adds the `base_model` listed on each Hugging Face model card as a
candidate (metadata only proposes candidates; the decision uses the weights).
`--cache lineage/grams.npz --aliases plans/lineage.json` reuses a saved run, so the check is
instant. Exit codes: 0 OK/INFO, 1 WARNING, 2 WRONG_BASE, 3 TOO_FAR.

## How it works

1. **Distance** between two models: average over the 196 attention/MLP weight matrices of
   the relative size of their difference.
2. **Family tree**: a minimum-cost directed spanning tree (Chu-Liu/Edmonds) over the
   distances, with link direction from weight kurtosis, following MoTHer
   (Horwitz et al., ICLR 2025).
3. **Base choice**: among the models on the tree paths joining the models to merge, the one
   closest to all of them.
4. **Double-counting check**: cosine between the models' task vectors measured from a
   candidate base. Near 1 means they share an ancestor's update; near 0 means independent.
5. **Distance gate**: if even the best base is more than 10% away from a model, do not merge.

## Results (Qwen2.5-1.5B family, Kaggle T4, GSM8K and MBPP, first 200 test items each)

- **Family tree from weights:** 9/9 links and 8/9 parents correct for 10 checkpoints
  (base, Instruct, 4 Instruct fine-tunes, Coder base + Instruct, Math base + Instruct).
- **Base choice:** Task Arithmetic (sum) is better from the recovered ancestor in **5/5**
  merge sets, by 10 to 55 points; TIES is much less sensitive (4/5, mean +0.95 points).
- **Double counting:** task-vector cosine 0.80-0.94 from the usual base vs about 0.00 from
  the recovered ancestor, for every pair.
- **Distance gate:** stepping from Instruct towards Coder stays healthy up to 10%, degrades
  at 20%, fails at 30% and collapses at 50%.
- 42 merges evaluated in total. Full tables: [`docs/RESULTS.md`](docs/RESULTS.md).

## Files

| File | Purpose |
|---|---|
| `base_checker.py` | The Base Checker for mergekit configs |
| `lineage.py` | Family tree from weights, base choice, gate (`python lineage.py plans/lineage.json`) |
| `merge.py`, `mergecore.py` | Task Arithmetic, TIES, DARE, Iso-C merging (NumPy) |
| `eval_model.py` | GSM8K + MBPP evaluation |
| `predictors.py` | Pre-merge predictors (principal angles, variance score, cosine, distance) |
| `run_plan.py`, `plans/` | Resumable experiment runner and every experiment we ran |
| `examples/` | mergekit configs for the four cases above |
| `tests/` | 47 tests, no GPU or downloads needed: `python -m pytest tests` |
| `docs/EXPERIMENTS.md` | How we ran everything on Kaggle |

## Related work

Recovering model family trees from weights: MoTHer (Horwitz et al., ICLR 2025),
modelDNA (arXiv 2607.10617), Model Atlas (arXiv 2503.10633). Grouping a model zoo, merging
within groups and routing between them: Model-GLUE (arXiv 2410.05357). Predicting merge
collapse with a fixed base: arXiv 2610.03199, arXiv 2603.09463. To our knowledge, none of
these uses model lineage to choose the base for merging, which is the focus of this project.

## Licence

MIT (see `LICENSE`).
