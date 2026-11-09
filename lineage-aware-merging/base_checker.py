"""
Base Checker: catch a wrong `base_model` in a mergekit config before you merge.

    python base_checker.py --config my_merge.yml
    python base_checker.py --config my_merge.yml --candidates Qwen/Qwen2.5-1.5B-Instruct --out fixed.yml
    python base_checker.py --config my_merge.yml --cache lineage/grams.npz --aliases plans/lineage.json

What it does, using the model weights only (no data, no GPU):
  1. Builds the family tree of every model involved: the models in the config, the
     declared base_model, and any --candidates (or, with --discover, the base_model
     listed on each model card; card metadata only proposes candidates, the decision
     is made from the weights).
  2. Finds the closest common ancestor of the models being merged (lineage.py).
  3. Compares it with the declared base_model:
       OK          the declared base is the closest common ancestor;
       WRONG_BASE  the models share a closer ancestor, the merge method adds task
                   vectors, and the weights sum to more than 1, so that ancestor's own
                   update is counted several times (measured on Qwen2.5-1.5B: 3-model
                   task arithmetic 68.0/48.0 -> 6.5/0.0 on GSM8K/MBPP);
       WARNING     a closer ancestor exists but the method is less sensitive
                   (TIES-style trimming, or weights normalised to sum to 1);
       TOO_FAR     even the best ancestor is more than --gate away from a model:
                   merging is unsafe from any base (keep the models separate);
       INFO        the method does not use the base for task vectors (linear, slerp).
  4. With --out, writes a copy of the config with base_model replaced by the closest
     common ancestor.

The evidence printed for a wrong base is the cosine between the models' task vectors
measured from the declared base (near 1 = they share the ancestor's update) and from
the closest common ancestor (near 0 = independent), plus each model's distance.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

import lineage as L

# How each mergekit method uses the base. "sum": task vectors are added with the given
# weights, so a shared ancestor's update is counted sum(weights) times. "trim": TIES-style
# trimming and sign election, which removes much of a repeated update (measured: TIES
# changed by -5.75..+5.75 points with the wrong base vs -10..-55 for task arithmetic).
# "none": the base is not used to form task vectors.
METHODS = {
    "task_arithmetic": "sum", "dare_linear": "sum", "breadcrumbs": "sum", "della_linear": "sum",
    "ties": "trim", "dare_ties": "trim", "breadcrumbs_ties": "trim", "della": "trim",
    "linear": "none", "slerp": "none", "nuslerp": "none", "passthrough": "none",
    "model_stock": "other", "nearswap": "other", "sce": "trim", "karcher": "none",
}

DEFAULT_GATE = 0.10      # max relative distance from the base (Phase M rule; strength sweep, 9 Nov)
DEFAULT_COS = 0.30       # mean task-vector cosine above which a shared update is flagged


# ------------------------------------------------------------------ config

def _scalar(value, notes, what):
    """mergekit allows a number or a gradient list; use the mean of a list."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        notes.append(f"{what} is a gradient {value}; its mean is used")
        return float(np.mean(value))
    notes.append(f"{what} = {value!r} could not be read; 1.0 assumed")
    return 1.0


def parse_config(cfg):
    """mergekit config dict -> {method, base, models: [(id, weight)], normalize, notes}."""
    notes = []
    method = str(cfg.get("merge_method", "")).strip()
    if not method:
        raise ValueError("config has no merge_method")
    base = cfg.get("base_model")
    if isinstance(base, dict):
        base = base.get("model")
    entries = cfg.get("models") or []
    if not entries and cfg.get("slices"):
        raise ValueError("configs with 'slices' are not supported; use a 'models' list")
    glob = cfg.get("parameters") or {}
    models = []
    for e in entries:
        mid = e.get("model") if isinstance(e, dict) else e
        if isinstance(mid, dict):
            mid = mid.get("model")
        params = (e.get("parameters") or {}) if isinstance(e, dict) else {}
        w = _scalar(params.get("weight", glob.get("weight")), notes, f"weight of {mid}")
        if w is None:
            w = 1.0
            if METHODS.get(method) == "sum":
                notes.append(f"no weight for {mid}; mergekit's default 1.0 assumed")
        if base is not None and mid == base:
            continue                      # mergekit lists the base among models at times
        models.append((str(mid), w))
    if not models:
        raise ValueError("config lists no models to merge")
    normalize = glob.get("normalize")
    if normalize is None:
        normalize = (method in ("linear",))
    return {"method": method, "kind": METHODS.get(method, "other"), "base": base,
            "models": models, "normalize": bool(normalize), "notes": notes}


def fixed_config(cfg, new_base):
    out = copy.deepcopy(cfg)
    if isinstance(out.get("base_model"), dict):
        out["base_model"]["model"] = new_base
    else:
        out["base_model"] = new_base
    return out


# ------------------------------------------------------------------ candidates and weights

def discover_bases(model_ids, depth=3):
    """Follow `base_model` on Hugging Face model cards (candidates only)."""
    from huggingface_hub import model_info
    found, frontier = [], list(model_ids)
    for _ in range(depth):
        nxt = []
        for mid in frontier:
            try:
                card = model_info(mid).card_data
                bm = getattr(card, "base_model", None) if card is not None else None
            except Exception:
                bm = None
            for b in ([bm] if isinstance(bm, str) else (bm or [])):
                if b not in found and b not in model_ids:
                    found.append(b)
                    nxt.append(b)
        frontier = nxt
    return found


def grams_from_cache(cache, aliases, pool):
    """Sub-matrix of a saved grams.npz covering every model in pool, or None."""
    data = dict(np.load(cache, allow_pickle=False))
    names = [str(n) for n in data["names"]]
    to_repo = {a: r for a, r in aliases.items()} if aliases else {}
    repo_names = [to_repo.get(n, n) for n in names]
    if not all(p in repo_names for p in pool):
        return None
    idx = [repo_names.index(p) for p in pool]
    return {"names": np.array(pool), "keys": data["keys"], "skipped": data["skipped"],
            "G": data["G"][:, idx][:, :, idx], "kurt": data["kurt"][:, idx]}


def grams_from_weights(pool):
    import hf_io
    models = {}
    for mid in pool:
        print(f"Reading {mid}", flush=True)
        models[mid] = hf_io.open_model(mid)
    return L.gram_pass(models)


# ------------------------------------------------------------------ the check

def check(data, parsed, gate=DEFAULT_GATE, cos_thr=DEFAULT_COS):
    names = [str(n) for n in data["names"]]
    idx = {n: i for i, n in enumerate(names)}
    G = data["G"]
    members = [m for m, _ in parsed["models"]]
    mi = [idx[m] for m in members]
    D = L.distance_matrix(G)
    edges = L.minimum_spanning_tree(D)
    best, cost, _, cands = L.choose_reference(G, edges, mi, gate)
    declared = parsed["base"]
    di = idx.get(declared)

    def pair_cos(r):
        vals = [L.tau_cos(G, a, b, r) for x, a in enumerate(mi) for b in mi[x + 1:]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    weights = [w for _, w in parsed["models"]]
    counted = 1.0 if parsed["normalize"] else float(sum(weights))
    rep = {
        "method": parsed["method"], "method_kind": parsed["kind"],
        "declared_base": declared, "closest_common_ancestor": names[best],
        "candidates_on_tree_path": [names[c] for c in cands],
        "tree_links": [[names[a], names[b], round(float(D[a, b]), 4)] for a, b in edges],
        "distance_from_ancestor": {m: round(L.rel_from(G, i, best), 4) for m, i in zip(members, mi)},
        "task_vector_cos_from_ancestor": pair_cos(best),
        "weights_sum": round(float(sum(weights)), 4), "normalize": parsed["normalize"],
        "ancestor_update_counted_times": round(counted, 3),
        "gate": gate, "notes": list(parsed["notes"]),
    }
    if di is not None:
        rep["distance_from_declared_base"] = {m: round(L.rel_from(G, i, di), 4) for m, i in zip(members, mi)}
        rep["task_vector_cos_from_declared_base"] = pair_cos(di)
        rep["declared_to_ancestor_distance"] = round(L.rel_from(G, best, di), 4) if best != di else 0.0

    if cost > gate:
        far = [m for m, i in zip(members, mi) if L.rel_from(G, i, best) > gate]
        rep["verdict"] = "TOO_FAR"
        rep["message"] = (f"No common ancestor is close enough: {', '.join(far)} "
                          f"{'is' if len(far) == 1 else 'are'} more than {gate:.0%} away from the best "
                          f"base ({names[best]}). Merging is unsafe from any base; keep these models separate.")
        return rep
    if declared is None:
        rep["verdict"] = "WARNING"
        rep["message"] = f"No base_model given; the closest common ancestor is {names[best]}."
        return rep
    if di == best:
        rep["verdict"] = "OK"
        rep["message"] = f"base_model {declared} is the closest common ancestor of the merged models."
        return rep
    if parsed["kind"] == "none":
        rep["verdict"] = "INFO"
        rep["message"] = (f"{parsed['method']} does not use the base to form task vectors, so the base "
                          f"choice does not cause double counting. (Closest ancestor: {names[best]}.)")
        return rep

    shared = rep.get("task_vector_cos_from_declared_base")
    shared_txt = (f" Measured from {declared}, the task vectors have cosine {shared:.2f} "
                  f"(from {names[best]}: {rep['task_vector_cos_from_ancestor']:.2f})." if shared is not None and
                  rep["task_vector_cos_from_ancestor"] is not None else "")
    if parsed["kind"] == "sum" and counted > 1.05 and (shared is None or shared > cos_thr):
        rep["verdict"] = "WRONG_BASE"
        rep["message"] = (f"The merged models share a closer ancestor, {names[best]}. With base_model "
                          f"{declared}, every task vector contains that ancestor's own update, and "
                          f"{parsed['method']} adds them with total weight {counted:g}, so the update is "
                          f"counted {counted:g} times instead of once.{shared_txt} "
                          f"Use base_model: {names[best]}.")
    else:
        why = ("weights are normalised to sum to 1" if counted <= 1.05 and parsed["kind"] == "sum"
               else f"{parsed['method']} trims and sign-votes task vectors, which removes much of a repeated update"
               if parsed["kind"] == "trim" else f"the effect on {parsed['method']} was not measured")
        rep["verdict"] = "WARNING"
        rep["message"] = (f"A closer common ancestor exists: {names[best]} (declared: {declared}). The risk is "
                          f"lower because {why}, but {names[best]} is still the better base.{shared_txt}")
    return rep


ICON = {"OK": "[ OK ]", "INFO": "[INFO]", "WARNING": "[WARN]", "WRONG_BASE": "[FAIL]", "TOO_FAR": "[STOP]"}


def render(rep):
    lines = [f"{ICON.get(rep['verdict'], '')} {rep['verdict']}: {rep['message']}", "",
             f"merge_method: {rep['method']}   declared base_model: {rep['declared_base']}",
             (f"best available base (still too far): {rep['closest_common_ancestor']}"
              if rep["verdict"] == "TOO_FAR" else f"closest common ancestor: {rep['closest_common_ancestor']}"), "",
             "Family tree (links between closest relatives, with distance):"]
    lines += [f"  {a}  --  {b}   ({d:.4f})" for a, b, d in rep["tree_links"]]
    lines += ["", "Distance of each merged model (relative weight change):",
              f"  {'model':55s} {'from ancestor':>14s} {'from declared':>14s}"]
    for m, d in rep["distance_from_ancestor"].items():
        dd = rep.get("distance_from_declared_base", {}).get(m)
        lines.append(f"  {m:55s} {d:14.4f} {('' if dd is None else f'{dd:.4f}'):>14s}")
    if rep.get("task_vector_cos_from_ancestor") is not None:
        lines.append(f"\nTask-vector cosine between the merged models: from ancestor "
                     f"{rep['task_vector_cos_from_ancestor']:.3f}"
                     + (f", from declared base {rep['task_vector_cos_from_declared_base']:.3f}"
                        if rep.get("task_vector_cos_from_declared_base") is not None else "")
                     + "  (near 1 = shared update counted again; near 0 = independent)")
    lines.append(f"Weights sum to {rep['weights_sum']:g} (normalize: {rep['normalize']}); a shared "
                 f"ancestor's update would be counted {rep['ancestor_update_counted_times']:g} times.")
    for n in rep["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="mergekit YAML config")
    p.add_argument("--candidates", nargs="*", default=[], help="Extra possible ancestors (repo ids or folders)")
    p.add_argument("--discover", action="store_true", help="Add base_model entries from model cards as candidates")
    p.add_argument("--cache", default=None, help="Saved grams.npz from lineage.py (skips reading weights)")
    p.add_argument("--aliases", default=None, help="Plan JSON whose 'models' maps cache names to repo ids")
    p.add_argument("--save-cache", default=None, help="Save the computed grams here")
    p.add_argument("--gate", type=float, default=DEFAULT_GATE)
    p.add_argument("--cos", type=float, default=DEFAULT_COS)
    p.add_argument("--out", default=None, help="Write a corrected config here")
    p.add_argument("--json", default=None, help="Write the full report as JSON here")
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    parsed = parse_config(cfg)
    pool = [m for m, _ in parsed["models"]]
    for extra in ([parsed["base"]] if parsed["base"] else []) + list(args.candidates):
        if extra not in pool:
            pool.append(extra)
    if args.discover:
        for b in discover_bases(pool):
            print(f"candidate from model cards: {b}")
            pool.append(b)

    data = None
    if args.cache:
        aliases = json.loads(Path(args.aliases).read_text())["models"] if args.aliases else None
        data = grams_from_cache(args.cache, aliases, pool)
        if data is None:
            print("cache does not cover every model; reading weights instead", flush=True)
    if data is None:
        data = grams_from_weights(pool)
        if args.save_cache:
            np.savez(args.save_cache, **data)

    rep = check(data, parsed, gate=args.gate, cos_thr=args.cos)
    print(render(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2))
    if args.out and rep["verdict"] in ("WRONG_BASE", "WARNING") and rep["closest_common_ancestor"] != parsed["base"]:
        Path(args.out).write_text(yaml.safe_dump(fixed_config(cfg, rep["closest_common_ancestor"]), sort_keys=False))
        print(f"\nCorrected config written to {args.out} (base_model: {rep['closest_common_ancestor']})")
    return {"OK": 0, "INFO": 0, "WARNING": 1, "WRONG_BASE": 2, "TOO_FAR": 3}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
