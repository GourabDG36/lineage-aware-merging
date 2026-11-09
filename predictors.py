"""
Pre-merge predictors for a pair of fine-tuned models, computed from the weights only.

    python predictors.py --reference Qwen/Qwen2.5-1.5B-Instruct \
        --models Thomas-Chou/Qwen2.5-1.5B-Open-R1-GRPO jahyungu/Qwen2.5-1.5B-Instruct_mbpp \
        --name r1math_mbpp_ref-instruct

For every attention/MLP matrix ("cell" = block x weight type; 196 for Qwen2.5-1.5B):
  * subspace geometry (ours): mean principal angle between the top-k left singular
    subspaces of the two task vectors, and how many degrees it lies below the chance mean
    of two random k-dim subspaces of the same output dimension;
  * the interference score of Merge Collapse (arXiv 2610.03199): per-coordinate variance of
    norm-scaled task vectors, with the cancellation ratio rho and coherence ratio, over the
    MLP projections (the paper's default scope) and over all cells;
  * simple baselines: cosine between the task vectors, ratio of their norms;
  * reference validity: ||tau_k|| / ||W_ref|| and cos(W_k, W_ref) per model (Phase K checks).

Writes predictors/<name>/summary.json (one row of predictors for the pair) and
predictors/<name>/cells.csv (every cell). Runs on the CPU (GPU not needed).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import mergecore as mc


def cell_keys(ref):
    cells = {}
    for key in ref.keys():
        c = mc.cell_of(key)
        if c is not None:
            cells[key] = c
    return dict(sorted(cells.items(), key=lambda kv: (kv[1][0], mc.CELL_TYPES.index(kv[1][1]))))


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


class SubspaceCache:
    """Top-k left singular subspaces of one model's task vectors, saved to disk.

    A model's subspaces depend only on (reference, model, k), so they are computed once
    and reused for every pair the model appears in. cache_dir=None disables saving.
    """

    def __init__(self, cache_dir, ref_id, model_id, k):
        self.path = (Path(cache_dir) / f"{_safe(ref_id)}__{_safe(model_id)}__k{k}.npz") if cache_dir else None
        self.stored = dict(np.load(self.path)) if self.path and self.path.exists() else {}
        self.new = {}

    def get(self, key, tau, k):
        if key in self.stored:
            return self.stored[key]
        u = mc.top_left_subspace(tau, k)
        self.new[key] = u
        return u

    def save(self):
        if self.path and self.new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, **{**self.stored, **self.new})


def compute(ref, wa_model, wb_model, k=64, chance_pairs=50, log_every=28, cache_dir=None):
    cells = cell_keys(ref)
    if not cells:
        raise ValueError("no attention/MLP matrices found in the reference")
    scopes = {"mlp": [key for key, (_, t) in cells.items() if t in mc.MLP_TYPES],
              "all": list(cells)}
    cache_a = SubspaceCache(cache_dir, ref.id, wa_model.id, k)
    cache_b = SubspaceCache(cache_dir, ref.id, wb_model.id, k)

    # Pass 1: global squared norms of each task vector over each scope (for the s_k scaling).
    print("Pass 1/2: task-vector norms", flush=True)
    sq = {key: None for key in cells}
    for key in cells:
        w = ref.get(key)
        ta, tb = wa_model.get(key) - w, wb_model.get(key) - w
        sq[key] = (float(np.sum(ta.astype(np.float64) ** 2)), float(np.sum(tb.astype(np.float64) ** 2)))
    scales = {s: mc.norm_scales([sum(sq[k_][0] for k_ in keys), sum(sq[k_][1] for k_ in keys)])
              for s, keys in scopes.items()}

    # Pass 2: everything else, one cell at a time.
    print("Pass 2/2: angles, variance terms, cosines", flush=True)
    chance_cache, rows, var_rows = {}, [], {s: [] for s in scopes}
    t0 = time.time()
    for i, (key, (block, wtype)) in enumerate(cells.items(), 1):
        w = ref.get(key)
        wa, wb = wa_model.get(key), wb_model.get(key)
        ta, tb = wa - w, wb - w
        m = w.shape[0]
        if (m, k) not in chance_cache:
            chance_cache[(m, k)] = mc.chance_mean_angle(m, min(k, min(w.shape)), n_pairs=chance_pairs)
        angles = mc.principal_angles_deg(cache_a.get(key, ta, k), cache_b.get(key, tb, k))
        stats = mc.pair_stats(w, wa, wb)
        wcos_a = float(np.sum(wa.astype(np.float64) * w) / (np.linalg.norm(wa) * np.linalg.norm(w)))
        wcos_b = float(np.sum(wb.astype(np.float64) * w) / (np.linalg.norm(wb) * np.linalg.norm(w)))
        row = {"block": block, "type": wtype, "m": m, "n": w.shape[1],
               "mean_angle": float(angles.mean()), "min_angle": float(angles.min()),
               "chance": chance_cache[(m, k)],
               "below_chance": chance_cache[(m, k)] - float(angles.mean()),
               "cos_tau": stats["cos_tau"], "norm_ratio": stats["norm_ratio"],
               "rel_a": stats["rel_a"], "rel_b": stats["rel_b"],
               "wcos_a": wcos_a, "wcos_b": wcos_b,
               "dot": stats["dot"], "sq_a": stats["sq_a"], "sq_b": stats["sq_b"]}
        for s, keys in scopes.items():
            if key in keys:
                vt = mc.variance_terms([ta, tb], scales[s])
                var_rows[s].append(vt)
                if s == "all":
                    row["sigma_l"] = float(np.sqrt(vt["I"] / vt["n"]))
        rows.append(row)
        if log_every and (i % log_every == 0 or i == len(cells)):
            print(f"  {i}/{len(cells)} cells  {time.time() - t0:.0f}s", flush=True)
    cache_a.save()
    cache_b.save()
    return rows, summarise(rows, var_rows)


def summarise(rows, var_rows):
    below = np.array([r["below_chance"] for r in rows])
    attn = np.array([r["below_chance"] for r in rows if r["type"] in mc.ATTN_TYPES])
    mlp = np.array([r["below_chance"] for r in rows if r["type"] in mc.MLP_TYPES])
    dot = sum(r["dot"] for r in rows)
    sqa, sqb = sum(r["sq_a"] for r in rows), sum(r["sq_b"] for r in rows)
    return {
        "angle_below_chance_mean": float(below.mean()),
        "angle_below_chance_attn": float(attn.mean()) if attn.size else None,
        "angle_below_chance_mlp": float(mlp.mean()) if mlp.size else None,
        "angle_below_chance_max": float(below.max()),
        "frac_cells_5deg_below": float(np.mean(below > 5.0)),
        "variance_mlp": mc.summarise_variance(var_rows["mlp"]),
        "variance_all": mc.summarise_variance(var_rows["all"]),
        "cos_tau_global": float(dot / np.sqrt(sqa * sqb)) if sqa > 0 and sqb > 0 else None,
        "cos_tau_cell_mean": float(np.nanmean([r["cos_tau"] for r in rows])),
        "norm_ratio_global": float(max(sqa, sqb) ** 0.5 / min(sqa, sqb) ** 0.5),
        "rel_norm_a_mean": float(np.mean([r["rel_a"] for r in rows])),
        "rel_norm_b_mean": float(np.mean([r["rel_b"] for r in rows])),
        "weight_cos_a_mean": float(np.mean([r["wcos_a"] for r in rows])),
        "weight_cos_b_mean": float(np.mean([r["wcos_b"] for r in rows])),
        "n_cells": len(rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reference", required=True)
    p.add_argument("--models", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--name", required=True, help="Output folder name under --out-dir")
    p.add_argument("--k", type=int, default=64, help="Subspace dimension (top-k singular vectors)")
    p.add_argument("--out-dir", default="predictors")
    p.add_argument("--cache-dir", default="predictors/_subspace_cache",
                   help="Where each model's top-k subspaces are stored for reuse ('' to disable)")
    args = p.parse_args()

    import hf_io
    ref = hf_io.open_model(args.reference)
    a, b = hf_io.open_model(args.models[0]), hf_io.open_model(args.models[1])
    rows, summary = compute(ref, a, b, k=args.k, cache_dir=args.cache_dir or None)
    summary = {"reference": args.reference, "model_a": args.models[0], "model_b": args.models[1],
               "k": args.k, **summary}
    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "cells.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
