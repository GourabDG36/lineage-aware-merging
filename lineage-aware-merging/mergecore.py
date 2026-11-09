"""
Merge operators and pre-merge predictors, in plain NumPy (float32/float64).

Everything here works on one weight tensor at a time, so the same code runs on a
laptop, on Kaggle, or in tests with small random arrays. Reading and writing model
files is done elsewhere (hf_io.py).

Notation: W_ref is the reference weight, W_k the k-th fine-tuned weight,
tau_k = W_k - W_ref its task vector, K the number of models being merged.
"""

import numpy as np

# Weight types whose left singular subspaces are compared (one "cell" = block x type).
ATTN_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TYPES = ("gate_proj", "up_proj", "down_proj")
CELL_TYPES = ATTN_TYPES + MLP_TYPES


def cell_of(key):
    """('layers.12.mlp.up_proj.weight') -> (12, 'up_proj'); None for non-cell tensors."""
    if not key.endswith(".weight"):
        return None
    parts = key.split(".")
    if "layers" not in parts:
        return None
    try:
        block = int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None
    wtype = parts[-2]
    return (block, wtype) if wtype in CELL_TYPES else None


# ------------------------------------------------------------------ merge operators

def task_arithmetic(ref, deltas, lam):
    """W = W_ref + lam * sum_k tau_k  (lam = 1/K gives the plain average of the models)."""
    return ref + lam * np.sum(deltas, axis=0)


def _trim_top(delta, density):
    """Keep the `density` fraction of entries with the largest magnitude, zero the rest."""
    flat = np.abs(delta).ravel()
    keep = int(np.ceil(density * flat.size))
    if keep >= flat.size:
        return delta.copy()
    if keep <= 0:
        return np.zeros_like(delta)
    threshold = np.partition(flat, flat.size - keep)[flat.size - keep]
    return np.where(np.abs(delta) >= threshold, delta, 0.0).astype(delta.dtype)


def ties(ref, deltas, density, lam):
    """TIES-Merging (Yadav et al., 2023): trim, elect sign, disjoint mean.

    1. trim each task vector to its top-`density` entries by magnitude;
    2. elect a sign per entry from the sum of the trimmed vectors;
    3. average only the trimmed entries whose sign agrees with the elected sign;
    4. W = W_ref + lam * merged.
    """
    trimmed = np.stack([_trim_top(d, density) for d in deltas])
    elected = np.sign(trimmed.sum(axis=0))
    agree = (np.sign(trimmed) == elected) & (trimmed != 0)
    count = agree.sum(axis=0)
    total = np.where(agree, trimmed, 0.0).sum(axis=0)
    merged = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return ref + lam * merged


def dare(ref, deltas, drop, lam, rng):
    """DARE (Yu et al., 2024): drop each task-vector entry with probability `drop`,
    rescale survivors by 1/(1-drop), then Task Arithmetic."""
    if not 0.0 <= drop < 1.0:
        raise ValueError("drop must be in [0, 1)")
    kept = []
    for d in deltas:
        mask = rng.random(d.shape) >= drop
        kept.append(np.where(mask, d / (1.0 - drop), 0.0).astype(d.dtype))
    return task_arithmetic(ref, np.stack(kept), lam)


def iso_c(ref, deltas, alpha, rel_tol=1e-6):
    """Iso-C (Marczak et al., ICML 2025) for one weight matrix.

    Sum the task matrices, take the SVD of the sum, replace every singular value by
    their mean (an isotropic spectrum), and add alpha times the result:
        D = sum_k tau_k = U S V^T,   D_iso = mean(S) * U V^T,   W = W_ref + alpha * D_iso.
    Singular values below rel_tol * max are treated as zero and dropped (otherwise a
    rank-deficient sum, e.g. merged LoRAs, would gain arbitrary directions).
    """
    d = np.sum(deltas, axis=0).astype(np.float64)
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return ref.copy()
    keep = s > rel_tol * s[0]
    u, s, vt = u[:, keep], s[keep], vt[keep]
    return (ref + alpha * s.mean() * (u @ vt)).astype(ref.dtype)


def merge_tensor(method, ref, tensors, *, lam, density=0.2, drop=0.9, rng=None,
                 alpha=1.0, is_matrix=True):
    """Merge one tensor. `tensors` are the fine-tuned weights (same shape as ref).

    Iso-C is defined for weight matrices; other tensors (embeddings, norms, biases) are
    merged with Task Arithmetic at `lam` (the plain average by default), as in the paper.
    """
    deltas = np.stack([t - ref for t in tensors])
    if method == "ta":
        return task_arithmetic(ref, deltas, lam)
    if method == "ties":
        return ties(ref, deltas, density, lam)
    if method == "dare":
        return dare(ref, deltas, drop, lam, rng if rng is not None else np.random.default_rng(0))
    if method == "iso_c":
        if is_matrix and ref.ndim == 2:
            return iso_c(ref, deltas, alpha)
        return task_arithmetic(ref, deltas, lam)
    raise ValueError(f"unknown method {method!r}")


# ------------------------------------------------------------------ subspace geometry

def top_left_subspace(tau, k, method="auto"):
    """Orthonormal basis (m x k) of the top-k left singular vectors of tau.

    "svd":  full economy SVD (exact, slow for large matrices).
    "gram": eigenvectors of the small Gram matrix in float64 (much faster when both
            sides are large). For m <= n, U = top eigenvectors of tau tau^T; for m > n,
            V from tau^T tau and U = tau V / sigma, re-orthonormalised.
    "auto": gram when the smaller side exceeds 512, else svd.
    """
    m, n = tau.shape
    k = min(k, m, n)
    if method == "auto":
        method = "gram" if min(m, n) > 512 else "svd"
    if method == "svd":
        u, _, _ = np.linalg.svd(tau.astype(np.float32), full_matrices=False)
        return u[:, :k]
    t = tau.astype(np.float64)
    if m <= n:
        vals, vecs = np.linalg.eigh(t @ t.T)
        return vecs[:, ::-1][:, :k].astype(np.float32)
    vals, vecs = np.linalg.eigh(t.T @ t)
    v = vecs[:, ::-1][:, :k]
    u = t @ v
    q, _ = np.linalg.qr(u)            # same span as the top-k left singular vectors
    return q.astype(np.float32)


def principal_angles_deg(ua, ub):
    """All principal angles (degrees) between span(ua) and span(ub): arccos sigma_i(ua^T ub)."""
    s = np.linalg.svd(ua.astype(np.float64).T @ ub.astype(np.float64), compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))


def chance_mean_angle(m, k, n_pairs=50, seed=0):
    """Mean principal angle between two random k-dim subspaces of R^m (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_pairs):
        qa, _ = np.linalg.qr(rng.standard_normal((m, k)))
        qb, _ = np.linalg.qr(rng.standard_normal((m, k)))
        means.append(principal_angles_deg(qa, qb).mean())
    return float(np.mean(means))


# ------------------------------------------------------------------ variance score (Merge Collapse, 2026)

def variance_terms(deltas, scales):
    """Per-coordinate sums used by the interference score of arXiv 2610.03199.

    With scaled task vectors x_k = s_k tau_k (s_k = min_j ||tau_j|| / ||tau_k||, norms over
    all merged coordinates), per coordinate i:
        P(i) = mean_k x_k(i)^2,  S(i) = (mean_k x_k(i))^2,  I(i) = P(i) - S(i) = Var_k x_k(i)
        Pos(i) = sum_k max(x_k(i), 0),  Neg(i) = sum_k max(-x_k(i), 0)
        C(i) = min( (2/K) min(Pos, Neg)^2 , I(i) )     (cancellation power)
    Returns the sums of P, S, I and C over the tensor and the number of coordinates n.
    """
    x = np.stack([s * d for s, d in zip(scales, deltas)]).astype(np.float64)
    K = x.shape[0]
    P = np.mean(x ** 2, axis=0)
    S = np.mean(x, axis=0) ** 2
    I = np.maximum(P - S, 0.0)
    pos = np.sum(np.maximum(x, 0.0), axis=0)
    neg = np.sum(np.maximum(-x, 0.0), axis=0)
    C = np.minimum((2.0 / K) * np.minimum(pos, neg) ** 2, I)
    return {"P": float(P.sum()), "S": float(S.sum()), "I": float(I.sum()),
            "C": float(C.sum()), "n": int(I.size)}


def norm_scales(sq_norms):
    """s_k = min_j ||tau_j|| / ||tau_k|| from the squared global norms of the task vectors."""
    norms = np.sqrt(np.asarray(sq_norms, dtype=np.float64))
    if np.any(norms == 0):
        raise ValueError("a task vector is exactly zero over the merged coordinates")
    return norms.min() / norms


def summarise_variance(rows, eps=1e-10, eps_g=1e-12):
    """Aggregate variance_terms rows (one per matrix) into the paper's quantities.

    sigma_l = sqrt(mean I) per matrix; reported as the coordinate-weighted global
    sigma = sqrt(sum I / sum n), the mean of per-matrix sigma_l, and the max sigma_l.
    rho = sum C / sum I (cancellation ratio); CR = sum S / sum P (coherence ratio).
    """
    if not rows:
        return {}
    I = np.array([r["I"] for r in rows])
    n = np.array([r["n"] for r in rows])
    sigma_l = np.sqrt(I / n)
    return {
        "sigma_global": float(np.sqrt(I.sum() / n.sum())),
        "sigma_mean": float(sigma_l.mean()),
        "sigma_max": float(sigma_l.max()),
        "rho": float(sum(r["C"] for r in rows) / (I.sum() + eps_g)),
        "coherence_ratio": float(sum(r["S"] for r in rows) / (sum(r["P"] for r in rows) + eps)),
        "n_matrices": len(rows),
    }


# ------------------------------------------------------------------ simple baselines

def pair_stats(ref, wa, wb):
    """Cosine and norm statistics for one matrix and two fine-tuned models."""
    ta, tb = (wa - ref).astype(np.float64), (wb - ref).astype(np.float64)
    na, nb, nr = np.linalg.norm(ta), np.linalg.norm(tb), np.linalg.norm(ref.astype(np.float64))
    dot = float(np.sum(ta * tb))
    return {
        "cos_tau": dot / (na * nb) if na > 0 and nb > 0 else float("nan"),
        "norm_ratio": float(max(na, nb) / min(na, nb)) if min(na, nb) > 0 else float("inf"),
        "rel_a": float(na / nr) if nr > 0 else float("nan"),
        "rel_b": float(nb / nr) if nr > 0 else float("nan"),
        "sq_a": float(na ** 2), "sq_b": float(nb ** 2), "dot": dot,
    }
