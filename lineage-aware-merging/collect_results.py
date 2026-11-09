"""
Collect every merge of one or more plans into one table: scores of the merged model, of its
parents and of Instruct, the change against the parents, and the pre-merge predictors.

    python collect_results.py plans/phase1.json plans/phase2.json

Writes results_table.csv and results_table.md (in the working folder) and prints the table.

Columns
  gsm8k, mbpp            merged model's accuracy (%)
  parents_avg            mean of the two parents' (gsm8k + mbpp) / 2
  delta_vs_parents       merged (gsm8k + mbpp) / 2 minus parents_avg, in points
  collapsed              True if the merged model scores below half of the weaker parent's
                         (gsm8k + mbpp) / 2 (the definition used in this project)
  angle_below_chance ... predictors of the pair (from predictors/<a>_<b>__ref-<reference>)
"""

import csv
import json
import sys
from pathlib import Path

import run_plan

PREDICTOR_FIELDS = [
    ("angle_below_chance_mean", lambda s: s.get("angle_below_chance_mean")),
    ("frac_cells_5deg_below", lambda s: s.get("frac_cells_5deg_below")),
    ("cos_tau", lambda s: s.get("cos_tau_global")),
    ("norm_ratio", lambda s: s.get("norm_ratio_global")),
    ("sigma_mlp", lambda s: s.get("variance_mlp", {}).get("sigma_global")),
    ("rho_mlp", lambda s: s.get("variance_mlp", {}).get("rho")),
    ("rel_norm_a", lambda s: s.get("rel_norm_a_mean")),
    ("rel_norm_b", lambda s: s.get("rel_norm_b_mean")),
]


def load_summary(results_dir, name):
    p = Path(results_dir) / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(summary, task):
    if not summary or task not in summary:
        return None
    return 100.0 * summary[task]["accuracy"]


def mean_score(summary, tasks):
    vals = [score(summary, t) for t in tasks]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def collect(plan_paths):
    rows = []
    for plan_path in plan_paths:
        plan = json.loads(Path(plan_path).read_text())
        aliases = plan["models"]
        tasks = plan.get("tasks", ["gsm8k", "mbpp"])
        results_dir = plan.get("results_dir", "results")
        predictors_dir = plan.get("predictors_dir", "predictors")
        for step in plan["steps"]:
            if step["type"] != "merge":
                continue
            name = run_plan.step_name(step, aliases)
            merged = load_summary(results_dir, name)
            if merged is None:
                continue
            parents = [load_summary(results_dir, m) for m in step["models"]]
            parent_means = [mean_score(p, tasks) for p in parents]
            merged_mean = mean_score(merged, tasks)
            row = {"merge": name, "method": step["method"], "reference": step["reference"],
                   "models": "+".join(step["models"])}
            for t in tasks:
                row[t] = score(merged, t)
            if all(v is not None for v in parent_means) and merged_mean is not None:
                row["parents_avg"] = sum(parent_means) / len(parent_means)
                row["delta_vs_parents"] = merged_mean - row["parents_avg"]
                row["collapsed"] = merged_mean < 0.5 * min(parent_means)
            instruct = load_summary(results_dir, "instruct")
            if instruct is not None and merged_mean is not None:
                row["delta_vs_instruct"] = merged_mean - mean_score(instruct, tasks)
            if len(step["models"]) == 2:
                pair = f"{step['models'][0]}_{step['models'][1]}__ref-{step['reference']}"
                p = Path(predictors_dir) / pair / "summary.json"
                if p.exists():
                    s = json.loads(p.read_text())
                    for field, get in PREDICTOR_FIELDS:
                        row[field] = get(s)
            rows.append(row)
    return rows


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1 and v != 0 else f"{v:.2f}"
    return "" if v is None else str(v)


def write_outputs(rows, out_csv="results_table.csv", out_md="results_table.md"):
    if not rows:
        print("No finished merges found.")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(fmt(r.get(f)) for f in fields) + " |" for r in rows]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out_csv}, {out_md}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python collect_results.py plans/phase1.json [plans/phase2.json ...]")
    write_outputs(collect(sys.argv[1:]))


if __name__ == "__main__":
    main()
