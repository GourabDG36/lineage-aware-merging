# Results

Qwen2.5-1.5B family. Evaluation: GSM8K and MBPP, first 200 test items each, greedy decoding,
float16, Kaggle T4. About ±3 points per score. Phase 3a was run twice and gave identical
scores.

## Parent models

| Model | Hugging Face repo | GSM8K | MBPP |
|---|---|---|---|
| Instruct | Qwen/Qwen2.5-1.5B-Instruct | 75.0 | 38.0 |
| qwq | micaebe/Qwen2.5-1.5B-Instruct-QwQ | 71.0 | 45.0 |
| mbpp | jahyungu/Qwen2.5-1.5B-Instruct_mbpp | 65.5 | 43.5 |
| business | joyboy5656n/business-assistant-llm | 60.5 | 39.5 |
| legal | Farhanabdul12/legal-qwen2.5-1.5b-finetuned | 60.0 | 38.0 |
| coder | Qwen/Qwen2.5-Coder-1.5B-Instruct | 50.0 | 52.5 |
| math | Qwen/Qwen2.5-Math-1.5B-Instruct | 85.0 | 24.5 |

## Family tree from weights (lineage.py, 10 checkpoints, CPU, 851 s)

Links 9/9 correct; parents 8/9 correct. The only miss is the direction between Qwen2.5-1.5B
and Qwen2.5-Math-1.5B (two pre-trained checkpoints).

Distance from Instruct: mbpp 0.0015, legal 0.0040, qwq 0.0052, business 0.0070.
Instruct from the base: 0.0123. Coder-Instruct from Coder base: 0.0145. Math-Instruct from
Math base: 0.1427. Coder family ~0.97 and Math family ~1.22 from the base.

Task-vector cosine between sibling pairs: 0.80-0.94 measured from the base,
0.0001-0.0054 measured from Instruct.

## Base choice: recovered ancestor (Instruct) vs the usual base

Score = (GSM8K + MBPP) / 2.

| Merge set | TA sum, Instruct | TA sum, base | Gain | TIES, Instruct | TIES, base | Gain |
|---|---|---|---|---|---|---|
| qwq+mbpp | 70.0/43.0 | 62.0/31.0 | +10.00 | 72.0/45.5 | 71.0/43.5 | +1.50 |
| qwq+business | 69.5/45.5 | 40.0/37.0 | +19.00 | 75.5/46.5 | 68.0/42.5 | +5.75 |
| mbpp+legal | 58.0/42.5 | 51.0/28.0 | +10.75 | 59.0/45.0 | 62.5/39.5 | +1.00 |
| business+legal | 54.5/37.0 | 41.0/26.5 | +12.00 | 60.0/37.0 | 68.0/40.5 | −5.75 |
| qwq+mbpp+business | 68.0/48.0 | 6.5/0.0 | +54.75 | 73.5/44.0 | 69.0/44.0 | +2.25 |

## Distance gate: stepping from Instruct towards Coder

| Step | GSM8K | MBPP | Answers hitting the token limit (GSM8K / MBPP) |
|---|---|---|---|
| 0% (Instruct) | 75.0 | 38.0 | – |
| 5% | 67.0 | 42.0 | 0 / 0 |
| 10% | 67.5 | 40.5 | 3 / 0 |
| 20% | 61.0 | 34.5 | 17 / 1 |
| 30% | 44.0 | 26.5 | 199 / 200 |
| 50% | 0.0 | 0.0 | – |
| 100% (Coder) | 50.0 | 52.5 | – |

## Merges of distant models (base = Qwen2.5-1.5B), all collapsed

| Merge | GSM8K | MBPP |
|---|---|---|
| TA, coder+instruct | 0.0 | 0.0 |
| TA, coder+math | 3.5 | 0.0 |
| TIES, coder+math | 2.0 | 0.0 |
| TA, coder+qwq | 0.0 | 0.0 |
| TA, math+instruct | 1.5 | 0.0 |
| TA, math+mbpp | 1.0 | 0.0 |

## Merges of Instruct fine-tunes from Instruct (phase 2)

| Merge | GSM8K | MBPP |
|---|---|---|
| TA, business+legal | 66.5 | 38.0 |
| TIES, business+legal | 60.0 | 37.0 |
| TA, mbpp+business | 68.0 | 44.0 |
| TIES, mbpp+business | 62.5 | 47.0 |
| TA, mbpp+legal | 65.5 | 42.0 |
| TIES, mbpp+legal | 59.0 | 45.0 |
| TA, qwq+business | 69.5 | 42.0 |
| TIES, qwq+business | 75.5 | 46.5 |
| TA, qwq+legal | 72.5 | 39.5 |
| TIES, qwq+legal | 73.0 | 39.5 |
| DARE, qwq+mbpp | 77.0 | 41.0 |
| Iso-C, qwq+mbpp | 70.5 | 40.5 |
| TA, qwq+mbpp | 72.0 | 42.5 |

## Pre-merge predictors of collapse (12 pairs, 5 collapsed)

| Predictor | AUROC |
|---|---|
| Distance from the base, max ‖τ‖/‖W_base‖ | 1.00 |
| Variance score σ (arXiv 2610.03199) | 1.00 |
| Cancellation ratio ρ | 0.94 |
| Norm ratio | 0.91 |
| Task-vector cosine | 0.86 |
| Coherence ratio | 0.86 |
| Principal angles below chance | 0.63 |
