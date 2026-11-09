"""Checks for mergecore, merge.run_merge and predictors.compute on small fake models (no torch needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mergecore as mc
import merge
import predictors


class FakeModel:
    def __init__(self, name, tensors):
        self.id, self.t = name, tensors

    def keys(self):
        return list(self.t)

    def shape(self, key):
        return self.t[key].shape

    def get(self, key):
        return self.t[key].astype(np.float32)


SHAPES = {"q_proj": (24, 16), "k_proj": (8, 16), "v_proj": (8, 16), "o_proj": (16, 24),
          "gate_proj": (40, 16), "up_proj": (40, 16), "down_proj": (16, 40)}


def fake_family(seed=0, n_blocks=2, delta_scale=0.01):
    rng = np.random.default_rng(seed)
    ref = {"model.embed_tokens.weight": rng.standard_normal((50, 16)),
           "model.norm.weight": np.ones(16)}
    for b in range(n_blocks):
        for t, shp in SHAPES.items():
            part = "self_attn" if t in mc.ATTN_TYPES else "mlp"
            ref[f"model.layers.{b}.{part}.{t}.weight"] = rng.standard_normal(shp)
    def variant(scale, s):
        r = np.random.default_rng(s)
        return {k: v + scale * r.standard_normal(v.shape) for k, v in ref.items()}
    return ref, variant(delta_scale, seed + 1), variant(delta_scale, seed + 2)


# ---------------------------------------------------------------- merge operators

def test_ta_average():
    ref = np.zeros((2, 2)); a = np.array([[1, 2], [3, 4.]]); b = np.array([[3, 0], [1, 0.]])
    out = mc.merge_tensor("ta", ref, [a, b], lam=0.5)
    assert np.allclose(out, (a + b) / 2)
    out1 = mc.merge_tensor("ta", ref, [a, b], lam=1.0)
    assert np.allclose(out1, a + b)


def test_ties_by_hand():
    ref = np.zeros(4)
    a = np.array([4.0, -1.0, 2.0, 0.5])
    b = np.array([-3.0, -2.0, 1.0, 0.1])
    # density 0.5 keeps the top 2 of 4 entries of each: a -> [4,0,2,0], b -> [-3,-2,0,0]
    # sums: [1,-2,2,0] -> elected signs [+,-,+,0]
    # entry0: only a agrees (+4) -> 4; entry1: only b (-2) -> -2; entry2: only a (2) -> 2; entry3: 0
    out = mc.ties(ref, np.stack([a, b]), density=0.5, lam=1.0)
    assert np.allclose(out, [4.0, -2.0, 2.0, 0.0]), out


def test_ties_full_density_agreeing_is_mean():
    ref = np.ones(3); a = ref + np.array([1.0, 2.0, 3.0]); b = ref + np.array([3.0, 2.0, 1.0])
    out = mc.merge_tensor("ties", ref, [a, b], lam=1.0, density=1.0)
    assert np.allclose(out, ref + np.array([2.0, 2.0, 2.0]))


def test_dare():
    ref = np.zeros((50, 50)); rng = np.random.default_rng(1)
    a, b = rng.standard_normal((50, 50)), rng.standard_normal((50, 50))
    assert np.allclose(mc.merge_tensor("dare", ref, [a, b], lam=0.5, drop=0.0), (a + b) / 2)
    # unbiased: the average over many seeds approaches Task Arithmetic
    outs = [mc.merge_tensor("dare", ref, [a, b], lam=0.5, drop=0.5, rng=np.random.default_rng(s))
            for s in range(400)]
    assert np.abs(np.mean(outs, axis=0) - (a + b) / 2).mean() < 0.05
    one = outs[0]
    assert 0.2 < np.mean(one == 0) < 0.35          # about drop^2 = 25% of entries lose both


def test_rebasing_identity():
    """Averaging (lam = 1/K) does not depend on the reference; summing (lam = 1) does."""
    base, (inst, a_d, b_d) = np.zeros(5), (np.full(5, 1.0), np.arange(5.0), -np.arange(5.0) * 0.5)
    a, b = inst + a_d, inst + b_d
    avg_base = mc.merge_tensor("ta", base, [a, b], lam=0.5)
    avg_inst = mc.merge_tensor("ta", inst, [a, b], lam=0.5)
    assert np.allclose(avg_base, avg_inst)
    sum_base = mc.merge_tensor("ta", base, [a, b], lam=1.0)
    sum_inst = mc.merge_tensor("ta", inst, [a, b], lam=1.0)
    assert np.allclose(sum_inst, inst + a_d + b_d)
    assert np.allclose(sum_base, 2 * inst + a_d + b_d)   # counts Instruct's own change twice


# ---------------------------------------------------------------- geometry

def test_principal_angles():
    e = np.eye(3)
    assert np.allclose(mc.principal_angles_deg(e[:, :1], e[:, :1]), 0.0, atol=1e-6)
    assert np.allclose(mc.principal_angles_deg(e[:, :1], e[:, 1:2]), 90.0)
    v = np.array([[np.cos(np.radians(60))], [np.sin(np.radians(60))], [0.0]])
    assert np.allclose(mc.principal_angles_deg(e[:, :1], v), 60.0)
    # basis rotation inside the subspace does not change the angles
    rng = np.random.default_rng(0)
    qa, _ = np.linalg.qr(rng.standard_normal((30, 4)))
    qb, _ = np.linalg.qr(rng.standard_normal((30, 4)))
    rot, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    assert np.allclose(np.sort(mc.principal_angles_deg(qa, qb)), np.sort(mc.principal_angles_deg(qa @ rot, qb)))


def test_top_left_subspace_recovers_low_rank():
    rng = np.random.default_rng(3)
    u, _ = np.linalg.qr(rng.standard_normal((40, 3)))
    tau = u @ np.diag([5.0, 3.0, 2.0]) @ rng.standard_normal((3, 20))
    est = mc.top_left_subspace(tau, 3)
    assert np.allclose(mc.principal_angles_deg(est, u), 0.0, atol=1e-2)


def test_gram_matches_svd():
    rng = np.random.default_rng(5)
    for shp in [(600, 300), (300, 600), (520, 530)]:
        u, _ = np.linalg.qr(rng.standard_normal((shp[0], 40)))
        v, _ = np.linalg.qr(rng.standard_normal((shp[1], 40)))
        tau = (u * np.linspace(5, 0.5, 40)) @ v.T + 0.01 * rng.standard_normal(shp)
        g = mc.top_left_subspace(tau, 16, "gram")
        s = mc.top_left_subspace(tau, 16, "svd")
        assert g.shape == s.shape == (shp[0], 16)
        assert np.allclose(g.T @ g, np.eye(16), atol=1e-4)
        assert mc.principal_angles_deg(g, s).max() < 0.05


def test_subspace_cache_reuse(tmp=None):
    import tempfile
    ref, a, b = fake_family(n_blocks=1)
    R, A, B = FakeModel("org/ref", ref), FakeModel("org/a", a), FakeModel("org/b", b)
    with tempfile.TemporaryDirectory() as d:
        rows1, s1 = predictors.compute(R, A, B, k=4, chance_pairs=3, log_every=0, cache_dir=d)
        files = sorted(Path(d).glob("*.npz"))
        assert len(files) == 2 and "org_a" in files[0].name
        # second run must read the cache: make SVD unavailable to prove it is not recomputed
        original = mc.top_left_subspace
        mc.top_left_subspace = lambda *a_, **k_: (_ for _ in ()).throw(AssertionError("recomputed"))
        try:
            rows2, s2 = predictors.compute(R, A, B, k=4, chance_pairs=3, log_every=0, cache_dir=d)
        finally:
            mc.top_left_subspace = original
        assert np.isclose(s1["angle_below_chance_mean"], s2["angle_below_chance_mean"])


def test_chance_means_match_report():
    assert abs(mc.chance_mean_angle(256, 64, n_pairs=20) - 63.03) < 0.5
    assert abs(mc.chance_mean_angle(1536, 64, n_pairs=10) - 79.94) < 0.5


# ---------------------------------------------------------------- variance score

def test_variance_terms_two_models():
    a = np.array([2.0, 1.0, -1.0]); b = np.array([2.0, -3.0, 0.0])
    vt = mc.variance_terms([a, b], [1.0, 1.0])
    # I(i) = (a-b)^2/4 = [0, 4, 0.25]
    assert np.isclose(vt["I"], 4.25)
    # C: coordinate 1 has opposite signs, min(Pos,Neg)=1 -> (2/2)*1 = 1 <= I=4 -> 1
    #    coordinate 2: Pos=0 -> 0 ; coordinate 0: same sign -> 0
    assert np.isclose(vt["C"], 1.0)
    assert np.isclose(vt["P"], (8 + 10 + 1) / 2) and np.isclose(vt["S"], 4 + 1 + 0.25)
    assert vt["n"] == 3


def test_norm_scales_and_summary():
    s = mc.norm_scales([4.0, 1.0])          # norms 2 and 1 -> s = [0.5, 1]
    assert np.allclose(s, [0.5, 1.0])
    summ = mc.summarise_variance([{"P": 2, "S": 1, "I": 1, "C": 0.5, "n": 4},
                                  {"P": 6, "S": 3, "I": 3, "C": 0.0, "n": 4}])
    assert np.isclose(summ["sigma_global"], np.sqrt(4 / 8))
    assert np.isclose(summ["rho"], 0.5 / 4, atol=1e-9)
    assert np.isclose(summ["coherence_ratio"], 4 / 8, atol=1e-9)


# ---------------------------------------------------------------- end to end on fake models

def test_run_merge_and_compat():
    ref, a, b = fake_family()
    R, A, B = FakeModel("ref", ref), FakeModel("a", a), FakeModel("b", b)
    assert merge.check_compatible(R, [A, B]) == []
    out, lam = merge.run_merge(R, [A, B], "ta", log_every=0)
    assert lam == 0.5
    for k in ref:
        assert np.allclose(out[k], (a[k] + b[k]) / 2, atol=1e-5)
    bad = dict(b); bad["model.norm.weight"] = np.ones(17); del bad["model.embed_tokens.weight"]
    probs = merge.check_compatible(R, [A, FakeModel("bad", bad)])
    assert any("missing model.embed_tokens.weight" in p for p in probs)
    assert any("shape" in p for p in probs)


def test_predictors_identical_and_independent():
    ref, a, b = fake_family(n_blocks=2)
    R, A = FakeModel("ref", ref), FakeModel("a", a)
    rows, summ = predictors.compute(R, A, A, k=4, chance_pairs=5, log_every=0)
    assert len(rows) == 14 and summ["n_cells"] == 14
    assert all(r["mean_angle"] < 1e-2 for r in rows)                  # same subspace
    assert np.isclose(summ["cos_tau_global"], 1.0)
    assert summ["variance_all"]["sigma_global"] < 1e-9                # no disagreement
    rows2, summ2 = predictors.compute(R, A, FakeModel("b", b), k=4, chance_pairs=5, log_every=0)
    assert abs(summ2["cos_tau_global"]) < 0.2                          # independent noise
    assert summ2["variance_all"]["sigma_global"] > 0
    assert abs(summ2["angle_below_chance_mean"]) < 15                  # near chance
    assert 0.005 < summ2["rel_norm_a_mean"] < 0.02                     # delta_scale = 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
