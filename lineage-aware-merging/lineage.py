"""
Lineage-aware merging, step 1: infer the family tree of a pool of checkpoints from their
weights alone, and choose the merge reference for any set of models from that tree.

    python lineage.py plans/lineage.json                 # reads weights (CPU, ~20-30 min)
    python lineage.py plans/lineage.json --from-grams    # redo the analysis only (seconds)

No data and no training. One pass over the weights computes, for every attention/MLP matrix
("cell", 196 for Qwen2.5-1.5B), the Gram matrix G[i, j] = <W_i, W_j> between all N models
and each model's weight kurtosis. Everything below is derived from those numbers, which are
saved to lineage/grams.npz, so the analysis can be re-run without the weights.

1. Distance.  d(i, j) = mean over cells of ||W_i - W_j|| / sqrt((||W_i||^2 + ||W_j||^2) / 2).
2. Direction. Weight-kurtosis rule of MoTHer (Horwitz et al., ICLR 2025): fine-tuning
              lowers kurtosis, so a parent has higher kurtosis than its fine-tuned child;
              models listed under "generalization" (continued pre-training, e.g.
              Qwen2.5-Coder-1.5B) are the reverse.
3. Tree.      As in MoTHer: minimum directed spanning tree (Chu-Liu/Edmonds) with edge
              cost d(p, c) + lam * [direction disagrees], lam = 0.3 x mean distance, over
              every possible root. A pre-trained checkpoint ("generalization") may only
              descend from another pre-trained checkpoint.
4. Refinement. A far model (e.g. a continued-pretraining checkpoint) is almost equally
              close to a parent and to that parent's small fine-tunes, so the tree can hang
              it on the wrong member of a tight family. When several candidate parents are
              within --tie (default 5%) of the closest one, the most ancestral is chosen.
5. Reference. For models S to merge, the candidates are the tree nodes on the paths
              joining S (their common ancestry). The chosen reference r minimises
              max_k ||W_k - W_r|| / ||W_r|| (mean over cells; the Phase K distance).
6. Gate.      If that distance exceeds --threshold (default 0.10, the Phase M accept rule),
              no reference makes the task vectors small: the merge is flagged "route"
              (keep the models as separate experts) instead of "merge".

Outputs in lineage/: grams.npz, distances.csv, tree.json, references.csv, report.md.
"""

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

import mergecore as mc


# ------------------------------------------------------------------ one pass over the weights

def common_cells(models):
    """Cell keys present with the same shape in every model (sorted block, type)."""
    names = list(models)
    first = models[names[0]]
    keys, skipped = [], []
    for k in first.keys():
        if mc.cell_of(k) is None:
            continue
        shapes = {n: (models[n].shape(k) if k in set(models[n].keys()) else None) for n in names}
        if len(set(shapes.values())) == 1:
            keys.append(k)
        else:
            skipped.append((k, shapes))
    keys.sort(key=lambda k: (mc.cell_of(k)[0], mc.CELL_TYPES.index(mc.cell_of(k)[1])))
    return keys, skipped


def excess_kurtosis(x):
    x = x.ravel()
    mu = x.mean()
    c = x - mu
    var = np.mean(c * c)
    return float(np.mean(c ** 4) / var ** 2 - 3.0) if var > 0 else 0.0


def gram_pass(models, log_every=28):
    """models: {name: object with keys()/shape()/get()}. Returns dict of arrays."""
    names = list(models)
    keys, skipped = common_cells(models)
    if not keys:
        raise SystemExit("no attention/MLP matrices are shared by all models")
    N, C = len(names), len(keys)
    G = np.zeros((C, N, N))
    kurt = np.zeros((C, N))
    t0 = time.time()
    for c, key in enumerate(keys):
        X = np.stack([models[n].get(key).astype(np.float64).ravel() for n in names])
        G[c] = X @ X.T
        kurt[c] = [excess_kurtosis(X[i]) for i in range(N)]
        del X
        if log_every and ((c + 1) % log_every == 0 or c + 1 == C):
            print(f"  {c + 1}/{C} matrices  {time.time() - t0:.0f}s", flush=True)
    return {"names": np.array(names), "keys": np.array(keys), "G": G, "kurt": kurt,
            "skipped": np.array([k for k, _ in skipped])}


# ------------------------------------------------------------------ quantities from the Grams

def sq_dist(G, i, j):
    """Per-cell ||W_i - W_j||^2 (clipped at 0 against rounding)."""
    return np.maximum(G[:, i, i] + G[:, j, j] - 2 * G[:, i, j], 0.0)


def distance_matrix(G):
    """Symmetric relative distance, mean over cells."""
    N = G.shape[1]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            scale = np.sqrt((G[:, i, i] + G[:, j, j]) / 2)
            D[i, j] = D[j, i] = float(np.mean(np.sqrt(sq_dist(G, i, j)) / scale))
    return D


def rel_from(G, k, r):
    """||W_k - W_r|| / ||W_r||, mean over cells: size of k's task vector measured from r."""
    if k == r:
        return 0.0
    return float(np.mean(np.sqrt(sq_dist(G, k, r)) / np.sqrt(G[:, r, r])))


def tau_cos(G, a, b, r):
    """cos(W_a - W_r, W_b - W_r), mean over cells (the double-counting signal)."""
    if r in (a, b):
        return float("nan")
    dot = G[:, a, b] - G[:, a, r] - G[:, b, r] + G[:, r, r]
    na, nb = np.sqrt(sq_dist(G, a, r)), np.sqrt(sq_dist(G, b, r))
    ok = (na > 0) & (nb > 0)
    return float(np.mean(dot[ok] / (na[ok] * nb[ok]))) if ok.any() else float("nan")


# ------------------------------------------------------------------ tree

def minimum_spanning_tree(D):
    """Prim's algorithm. Returns sorted list of undirected edges (i, j), i < j."""
    N = len(D)
    in_tree, edges = {0}, []
    while len(in_tree) < N:
        best = None
        for i in in_tree:
            for j in range(N):
                if j not in in_tree and (best is None or D[i, j] < D[best[0], best[1]]):
                    best = (i, j)
        edges.append(tuple(sorted(best)))
        in_tree.add(best[1])
    return sorted(edges)


def adjacency(edges, N):
    adj = {i: set() for i in range(N)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def orient(edges, N, root):
    """Parent of every node when the tree is hung from root (root's parent is None)."""
    adj, parent, seen, q = adjacency(edges, N), {root: None}, {root}, deque([root])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in seen:
                parent[v] = u
                seen.add(v)
                q.append(v)
    return parent


def edge_agrees(p, c, kscore, generalization):
    """MoTHer rule: fine-tuning lowers kurtosis; continued pre-training raises it."""
    return kscore[c] > kscore[p] if c in generalization else kscore[c] < kscore[p]


FORBIDDEN = 1e6


def mother_cost(D, kscore, generalization, c=0.3):
    """MoTHer edge cost for p -> child: distance + lam if the kurtosis direction disagrees,
    lam = c * mean off-diagonal distance (c = 0.3 as in the paper)."""
    N = len(D)
    lam = c * D[~np.eye(N, dtype=bool)].mean()
    M = np.array(D, dtype=float)
    for p in range(N):
        for ch in range(N):
            if p != ch and not edge_agrees(p, ch, kscore, generalization):
                M[p, ch] += lam
            if p != ch and ch in generalization and p not in generalization:
                M[p, ch] += FORBIDDEN    # a pre-trained checkpoint never descends from a fine-tune
    np.fill_diagonal(M, np.inf)
    return M


def _find_cycle(parent):
    for start in parent:
        seen, v = [], start
        while v in parent and v not in seen:
            seen.append(v)
            v = parent[v]
        if v in seen:
            return seen[seen.index(v):]
    return None


def edmonds(W, root):
    """Chu-Liu/Edmonds minimum spanning arborescence. W[p][c] = cost of edge p -> c.
    Returns {child: parent} for every node except root."""
    n = len(W)
    parent = {v: min((u for u in range(n) if u != v), key=lambda u: W[u][v])
              for v in range(n) if v != root}
    cycle = _find_cycle(parent)
    if cycle is None:
        return parent
    cyc = set(cycle)
    others = [v for v in range(n) if v not in cyc]
    idx = {v: i for i, v in enumerate(others)}
    c = len(others)
    W2 = np.full((c + 1, c + 1), np.inf)
    enter, leave = {}, {}
    for u in range(n):
        for v in range(n):
            if u == v or not np.isfinite(W[u][v]):
                continue
            if u not in cyc and v not in cyc:
                W2[idx[u], idx[v]] = W[u][v]
            elif v in cyc and u not in cyc:
                w = W[u][v] - W[parent[v]][v]
                if w < W2[idx[u], c]:
                    W2[idx[u], c], enter[idx[u]] = w, (u, v)
            elif u in cyc and v not in cyc:
                if W[u][v] < W2[c, idx[v]]:
                    W2[c, idx[v]], leave[idx[v]] = W[u][v], (u, v)
    sub = edmonds(W2, idx[root])
    result = {}
    for v_new, u_new in sub.items():
        if v_new == c:
            u, v = enter[u_new]
            result[v] = u
            for x in cyc:
                if x != v:
                    result[x] = parent[x]
        elif u_new == c:
            u, v = leave[v_new]
            result[v] = u
        else:
            result[others[v_new]] = others[u_new]
    return result


def directed_tree(D, kscore, generalization):
    """Minimum-cost arborescence over every possible root. Returns (parent incl. root: None,
    root, number of edges agreeing with the kurtosis rule)."""
    M = mother_cost(D, kscore, generalization)
    best = None
    for r in range(len(D)):
        par = edmonds(M, r)
        total = sum(M[p][ch] for ch, p in par.items())
        if best is None or total < best[0] - 1e-12:
            best = (total, r, par)
    _, root, par = best
    par = {**par, root: None}
    agree = sum(1 for ch, p in par.items() if p is not None and edge_agrees(p, ch, kscore, generalization))
    return par, root, agree


def depth_of(parent):
    depth = {}
    for v in parent:
        d, u = 0, v
        while parent[u] is not None:
            u, d = parent[u], d + 1
        depth[v] = d
    return depth


def descendants(parent, v):
    kids = {c for c, p in parent.items() if p == v}
    out = set(kids)
    for k in kids:
        out |= descendants(parent, k)
    return out


def refine(parent, D, tie):
    """Re-hang each node on the most ancestral candidate within (1 + tie) of its closest
    admissible parent. Repeated until nothing changes (at most N rounds)."""
    parent = dict(parent)
    N = len(D)
    for _ in range(N):
        changed = False
        depth = depth_of(parent)
        for v in sorted(parent, key=lambda x: depth[x]):
            if parent[v] is None:
                continue
            banned = descendants(parent, v) | {v}
            cands = [u for u in range(N) if u not in banned]
            best = min(D[v, u] for u in cands)
            near = [u for u in cands if D[v, u] <= (1 + tie) * best]
            pick = min(near, key=lambda u: (depth[u], D[v, u]))
            if pick != parent[v]:
                parent[v], changed = pick, True
                depth = depth_of(parent)
        if not changed:
            break
    return parent


def tree_edges(parent):
    return sorted(tuple(sorted((c, p))) for c, p in parent.items() if p is not None)


# ------------------------------------------------------------------ reference choice

def steiner_nodes(edges, N, members):
    """Nodes on the tree paths joining all members."""
    adj = adjacency(edges, N)
    members = list(members)

    def path(a, b):
        prev, q = {a: None}, deque([a])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        out, u = [], b
        while u is not None:
            out.append(u)
            u = prev[u]
        return out

    nodes = set(members)
    for m in members[1:]:
        nodes |= set(path(members[0], m))
    return sorted(nodes)


def choose_reference(G, edges, members, threshold):
    N = G.shape[1]
    cands = steiner_nodes(edges, N, members)
    cost = {r: max(rel_from(G, m, r) for m in members) for r in cands}
    best = min(cands, key=lambda r: (cost[r], r))
    return best, cost[best], ("merge" if cost[best] <= threshold else "route"), cands


# ------------------------------------------------------------------ analysis + report

def analyse(data, cfg, out, tie=0.05, threshold=0.10):
    names = [str(n) for n in data["names"]]
    idx = {n: i for i, n in enumerate(names)}
    G, kurt = data["G"], data["kurt"]
    N = len(names)
    generalization = {idx[n] for n in cfg.get("generalization", []) if n in idx}
    truth = {idx[c]: idx[p] for c, p in cfg.get("true_parent", {}).items() if c in idx and p in idx}

    D = distance_matrix(G)
    kscore = kurt.sum(axis=0)
    raw, root, agree = directed_tree(D, kscore, generalization)
    refined = refine(raw, D, tie)
    edges = tree_edges(refined)

    def score(parent):
        if not truth:
            return None
        hits = sum(parent.get(c) == p for c, p in truth.items())
        und = sum(tuple(sorted((c, p))) in set(tree_edges(parent)) for c, p in truth.items())
        return {"parents_correct": hits, "edges_correct": und, "of": len(truth)}

    out.mkdir(parents=True, exist_ok=True)
    with (out / "distances.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [f"{D[i, j]:.6f}" for j in range(N)])

    tree = {"names": names, "root": names[root], "kurtosis_score": dict(zip(names, map(float, kscore))),
            "kurtosis_edges_agreeing": agree, "tie": tie,
            "tree_parent": {names[c]: (names[p] if p is not None else None) for c, p in raw.items()},
            "parent": {names[c]: (names[p] if p is not None else None) for c, p in refined.items()},
            "true_parent": {names[c]: names[p] for c, p in truth.items()},
            "score_tree": score(raw), "score_refined": score(refined)}
    (out / "tree.json").write_text(json.dumps(tree, indent=2))

    rows = []
    base = idx.get(cfg.get("conventional_reference", "base"))
    for members in cfg.get("merge_sets", []):
        if not all(m in idx for m in members):
            continue
        mi = [idx[m] for m in members]
        r, cost, verdict, cands = choose_reference(G, edges, mi, threshold)
        row = {"models": "+".join(members), "chosen_reference": names[r],
               "max_distance_from_chosen": round(cost, 6), "verdict": verdict,
               "candidates": " ".join(names[c] for c in cands)}
        if base is not None:
            row["max_distance_from_base"] = round(max(rel_from(G, m, base) for m in mi), 6)
        if len(mi) == 2:
            row["tau_cos_at_chosen"] = round(tau_cos(G, mi[0], mi[1], r), 4)
            if base is not None:
                row["tau_cos_at_base"] = round(tau_cos(G, mi[0], mi[1], base), 4)
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / "references.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    lines = ["# Lineage report", "", f"Models: {N}. Matrices used: {len(data['keys'])}"
             f" (skipped, not shared by all: {len(data['skipped'])}).", "",
             "## Inferred parents", "", "| Model | Parent (tree) | Parent (refined) | True parent |",
             "|---|---|---|---|"]
    for i, n in enumerate(names):
        mp = names[raw[i]] if raw[i] is not None else "(root)"
        rp = names[refined[i]] if refined[i] is not None else "(root)"
        tp = names[truth[i]] if i in truth else ("-" if truth else "")
        lines.append(f"| {n} | {mp} | {rp} | {tp} |")
    for label, s in (("tree", score(raw)), ("refined", score(refined))):
        if s:
            lines.append(f"\n{label}: {s['parents_correct']}/{s['of']} parents correct, "
                         f"{s['edges_correct']}/{s['of']} tree edges correct (ignoring direction).")
    lines += ["", f"Root: {names[root]} (kurtosis rule agrees on {agree}/{N - 1} tree edges).", "",
              "## Distance matrix (mean relative distance over matrices)", "",
              "| | " + " | ".join(names) + " |", "|---" * (N + 1) + "|"]
    for i, n in enumerate(names):
        lines.append(f"| {n} | " + " | ".join(f"{D[i, j]:.4f}" for j in range(N)) + " |")
    if rows:
        lines += ["", f"## Merge references (threshold {threshold})", "",
                  "| Models | Chosen reference | Max distance (chosen) | Max distance (base) | "
                  "tau cos (chosen) | tau cos (base) | Verdict |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['models']} | {r['chosen_reference']} | {r['max_distance_from_chosen']:.4f} | "
                         f"{r.get('max_distance_from_base', float('nan')):.4f} | "
                         f"{r.get('tau_cos_at_chosen', '')} | {r.get('tau_cos_at_base', '')} | {r['verdict']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return tree, rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("plan", help="JSON with 'models' (alias -> repo) and optional 'lineage' settings")
    p.add_argument("--out", default="lineage")
    p.add_argument("--from-grams", action="store_true", help="Skip the weights; reuse <out>/grams.npz")
    p.add_argument("--tie", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    cfg = plan.get("lineage", {})
    tie = args.tie if args.tie is not None else cfg.get("tie", 0.05)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.10)
    out = Path(args.out)
    grams = out / "grams.npz"
    if args.from_grams:
        data = dict(np.load(grams, allow_pickle=False))
    else:
        import hf_io
        pool = cfg.get("pool", list(plan["models"]))
        models = {}
        for alias in pool:
            print(f"Opening {alias}: {plan['models'][alias]}", flush=True)
            models[alias] = hf_io.open_model(plan["models"][alias])
        data = gram_pass(models)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(grams, **data)
        print(f"Saved {grams}", flush=True)
    analyse(data, cfg, out, tie=tie, threshold=threshold)


if __name__ == "__main__":
    main()
