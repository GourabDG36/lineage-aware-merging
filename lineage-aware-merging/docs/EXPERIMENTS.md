# merge-lab

Implementation-semester code: merge Qwen2.5-1.5B fine-tunes, score the merges, and compute
weight-only predictors of merge collapse before merging.

| File | What it does |
|---|---|
| `run_plan.py` | **Runs a whole plan with one command**: skips finished steps, sanity-checks every new model on 8 GSM8K questions (0/8 = BROKEN, its steps skipped), logs every step, deletes merged weights after evaluation, refreshes `backup.zip` after every step |
| `plans/phase1.json` | Done (2 Nov): baselines, qwq + mbpp merged five ways, the re-basing test, coder + math against the base |
| `plans/phase2_cpu.json` | Predictors for every phase-2 pair. Run in a **CPU-only** notebook: costs no GPU quota |
| `plans/phase2_gpu.json` | Evaluate business and legal; TA and TIES merges of five pairs of Instruct fine-tunes; four mixed pairs (one far from the base, one near) |
| `collect_results.py` | Puts every finished merge in one table (`results_table.csv` / `.md`): merged scores, parents, change vs parents, collapse flag, predictors |
| `eval_model.py` | Scores a model on GSM8K (maths) and MBPP (Python), first N test items, greedy, float16; answer limits 1024 (GSM8K) and 768 (MBPP) tokens, and reports how many answers hit the limit |
| `merge.py` | Merges 2+ models into a reference: Task Arithmetic (`ta`), TIES (`ties`), DARE (`dare`), Iso-C (`iso_c`) |
| `predictors.py` | Pre-merge predictors for a pair: principal angles vs chance (ours), variance score of arXiv 2610.03199, cosine, norm ratio, reference validity |
| `mergecore.py` | The maths (NumPy), used by `merge.py` and `predictors.py` |
| `hf_io.py` | Reads/writes safetensors weights |
| `tests/` | Three test files, no GPU and no downloads needed: `for t in tests/test_*.py; do python $t; done` |

## Every Kaggle session (GPU T4 x2, Internet on)

```python
import shutil, os
shutil.copytree("/kaggle/input/datasets/gourabdiptaghosh/major-project-merge-data/merge-lab",
                "/kaggle/working", dirs_exist_ok=True)
os.chdir("/kaggle/working")
```

If you have a `backup.zip` from an earlier session, upload it as a dataset (or attach the
earlier notebook version's output) and unzip it into `/kaggle/working` first, so finished
steps are skipped:

```python
!unzip -o -q /kaggle/input/<your-backup-dataset>/backup.zip -d /kaggle/working
```

Then:

```python
!python run_plan.py plans/phase1.json --status    # what is done / to do
!python run_plan.py plans/phase1.json             # run everything not done yet
```

Download `backup.zip` from the Output panel when you stop. Do not `pip install` anything
in the same running notebook; nothing extra is needed (`datasets` is already on Kaggle).

Phase 2 uses two notebooks that can run at the same time:

- **CPU notebook** (Accelerator: None, Internet on): `!python run_plan.py plans/phase2_cpu.json`
- **GPU notebook** (GPU T4 x2, Internet on): restore the phase-1 `backup.zip` first (the
  merges are compared with their parents' phase-1 scores), then
  `!python run_plan.py plans/phase2_gpu.json`

Each produces its own `backup.zip`. Unzip both into one folder (they do not overlap) and run
`python collect_results.py plans/phase1.json plans/phase2_gpu.json` for the full table.
Predictors for a merge are looked up as `predictors/<modelA>_<modelB>__ref-<reference>`.

The base model is never evaluated: it is not chat-tuned and repeats one word on chat prompts
in float16 and float32 alike. It is used only as a reference.

## Models

| Short name | Repository | Small delta of | Status |
|---|---|---|---|
| base | `Qwen/Qwen2.5-1.5B` | — | |
| instruct | `Qwen/Qwen2.5-1.5B-Instruct` | base | GSM8K 75.0, MBPP 38.0 |
| qwq | `micaebe/Qwen2.5-1.5B-Instruct-QwQ` | instruct (r 0.52%; SFT on QwQ maths solutions) | GSM8K 71.0, MBPP 45.0 |
| mbpp | `jahyungu/Qwen2.5-1.5B-Instruct_mbpp` | instruct (r 0.15%) | GSM8K 65.5, MBPP 43.5 |
| coder | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | none (86% of base norm, Phase K) | GSM8K 50.0, MBPP 52.5 |
| math | `Qwen/Qwen2.5-Math-1.5B-Instruct` | none (140% of base norm, Phase K) | GSM8K 85.0, MBPP 24.5 |
| business | `joyboy5656n/business-assistant-llm` | instruct (r 0.70%, merged LoRA, rank ~16) | |
| legal | `Farhanabdul12/legal-qwen2.5-1.5b-finetuned` | instruct (r 0.40%) | |
| ~~r1_math~~ | `Thomas-Chou/Qwen2.5-1.5B-Open-R1-GRPO` | instruct (r 2.9%) | **BROKEN**: gibberish in float16 and float32, also with tied embeddings; dropped |

## Single commands (what run_plan.py runs for you)

```bash
python eval_model.py --model micaebe/Qwen2.5-1.5B-Instruct-QwQ --name qwq --limit 200
python predictors.py --reference Qwen/Qwen2.5-1.5B-Instruct \
    --models micaebe/Qwen2.5-1.5B-Instruct-QwQ jahyungu/Qwen2.5-1.5B-Instruct_mbpp \
    --name qwq_mbpp__ref-instruct
python merge.py --reference Qwen/Qwen2.5-1.5B-Instruct \
    --models micaebe/Qwen2.5-1.5B-Instruct-QwQ jahyungu/Qwen2.5-1.5B-Instruct_mbpp \
    --method ties --out merged/ties__qwq_mbpp__ref-instruct
python eval_model.py --model merged/ties__qwq_mbpp__ref-instruct --limit 200
```

Naming: `<method>[-lam<v>][-d<v>][-p<v>][-a<v>]__<modelA>_<modelB>__ref-<reference>`.

## Notes

- **Re-basing.** With plain averaging (`ta`, lam = 1/K) the merged weights are the same
  whichever reference is used, because the reference cancels out. The reference matters for
  `ta` with lam = 1 (a sum: measured from the base, Instruct's own change is counted twice),
  and for `ties`, `dare` and `iso_c` (non-linear). Re-basing comparisons therefore use those.
- **Config and tokenizer** of a merged model come from `--tokenizer-from` (default: the
  reference). When merging against the base, pass `--tokenizer-from Qwen/Qwen2.5-1.5B-Instruct`
  so every merged model uses the same chat template in evaluation.
- **Iso-C** (Marczak et al., ICML 2025): per attention/MLP matrix, SVD of the summed task
  matrices, every singular value replaced by their mean, scaled by `--alpha` (default 1.0;
  the paper tunes alpha on held-out data, which we do not do). Other tensors are averaged.
  Singular values below 1e-6 of the largest are dropped so rank-deficient sums (merged LoRAs)
  gain no arbitrary directions.
- **Variance score.** Implemented from arXiv 2610.03199 Section 3: norm-scaled task vectors
  (s_k = min_j ||tau_j|| / ||tau_k|| over the merged coordinates), per-coordinate population
  variance I = P − S, sigma_l = sqrt(mean I) per matrix, cancellation ratio rho and coherence
  ratio. The paper does not state how sigma_l is aggregated over layers or its numeric
  threshold, so `predictors.py` reports the coordinate-weighted global sigma, the mean and the
  max of sigma_l, over the MLP projections (paper default) and over all 196 matrices.
- **Prism** (the repair method of arXiv 2610.03199) is not implemented yet: some details
  (the exact scaling coefficient and which tensors stay at base values) must be confirmed
  from the paper first.
- **Speed.** Top-64 subspaces of large matrices use the eigenvectors of the Gram matrix
  (agrees with a full SVD to 0.01°, 4–7× faster), and each model's subspaces are cached in
  `predictors/_subspace_cache/` and reused for every pair it appears in.
- **Number format.** Every evaluation uses float16 (native on the T4). The same model scored
  69.0% in emulated bfloat16 and 74.5% in float16 on the same 200 GSM8K items, so results in
  different formats must never be compared; the script refuses to mix them.

## Phase 3: lineage-aware merging (added 9 Nov 2026)

| Notebook | Command | What it does | Time |
|---|---|---|---|
| CPU (no GPU) | `python lineage.py plans/lineage.json` | Objective 1: family tree of 10 checkpoints from weights only, checked against the model cards; merge reference + merge/route verdict for every merge set | ~30-45 min incl. ~30 GB downloads |
| GPU T4 | `python run_plan.py plans/phase3a_gpu.json` | Objective 3 strength sweep (5/10/20/30% from Instruct towards Coder) + 3-model merge from Instruct vs base | 8 evals |
| GPU T4 | `python run_plan.py plans/phase3b_gpu.json` | Objective 2: Instruct (lineage choice) vs base for 3 more pairs | 9 evals |

`lineage.py` writes `lineage/report.md` (readable summary), `tree.json`, `references.csv`,
`distances.csv` and `grams.npz`. With `grams.npz` the analysis re-runs in seconds without the
weights: `python lineage.py plans/lineage.json --from-grams`.

`merge.py` now accepts a single model with `--method ta --lam x`: W = W_ref + x * (W_model - W_ref),
used for the strength sweep.
