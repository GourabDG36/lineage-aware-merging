"""Checks for lineage.py on synthetic model families with a known tree (no torch needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lineage as L
import mergecore as mc


class FakeModel:
    def __init__(self, name, tensors):
        self.id, self.t = name, tensors

    def keys(self):
        return list(self.t)

    def shape(self, key):
        return self.t[key].shape

    def get(self, key):
        return self.t[key].astype(np.float32)


SHAPES = {"q_proj": (48, 32), "k_proj": (16, 32), "v_proj": (16, 32), "o_proj": (32, 48),
          "gate_proj": (80, 32), "up_proj": (80, 32), "down_proj": (32, 80)}


def keys(n_blocks=3):
    out = []
    for b in range(n_blocks):
        for t, shp in SHAPES.items():
            part = "self_attn" if t in mc.ATTN_TYPES else "mlp"
            out.append((f"model.layers.{b}.{part}.{t}.weight", shp))
    return out


def realistic_family(seed=0):
    """Mimics the Qwen pool: base -> instruct (1.2%) -> four small fine-tunes; base -> two
    far continued-pretraining models (coder_base, math_base), each with a small instruct
    child. The base is heavy-tailed (high kurtosis); fine-tuning adds Gaussian noise (lowers
    kurtosis); continued pre-training adds heavier-tailed noise (raises it). Instruct's
    update and the far models' updates both point partly against the base weights (as
    measured in Phase K), so far models sit slightly closer to instruct than to the base."""
    rng = np.random.default_rng(seed)
    fam = {n: {} for n in ["base", "instruct", "qwq", "mbpp", "business", "legal",
                           "coder_base", "coder", "math_base", "math"]}
    for k, shp in keys():
        B = rng.standard_t(4, shp)
        s = np.linalg.norm(B)
        unit = lambda: (lambda x: x / np.linalg.norm(x))(rng.standard_normal(shp))
        bdir = B / s

        def finetune(W, size):
            """Small update that also trims outlier weights (lowers kurtosis, as in MoTHer)."""
            tail = W ** 3 / np.linalg.norm(W ** 3)
            return W + np.linalg.norm(W) * size * (0.8 * unit() - 0.5 * tail)

        I = finetune(B, 0.012) - s * 0.0012 * bdir
        fam["base"][k], fam["instruct"][k] = B, I
        for n in ("qwq", "mbpp", "business", "legal"):
            fam[n][k] = finetune(I, 0.015)
        for far, child in (("coder_base", "coder"), ("math_base", "math")):
            heavy = rng.standard_t(3, shp)
            W = 0.45 * B + s * 0.8 * heavy / np.linalg.norm(heavy)
            fam[far][k] = W
            fam[child][k] = finetune(W, 0.03)
    truth = {"instruct": "base", "qwq": "instruct", "mbpp": "instruct", "business": "instruct",
             "legal": "instruct", "coder_base": "base", "coder": "coder_base",
             "math_base": "base", "math": "math_base"}
    return {n: FakeModel(n, t) for n, t in fam.items()}, truth


CFG_SETS = [["qwq", "mbpp"], ["qwq", "business"], ["business", "legal"], ["qwq", "mbpp", "business"],
            ["coder", "instruct"], ["coder", "math"], ["math", "mbpp"]]


def test_gram_quantities_match_direct_computation():
    models, _ = realistic_family(1)
    data = L.gram_pass(models, log_every=0)
    names = list(data["names"])
    G = data["G"]
    a, b, r = names.index("qwq"), names.index("mbpp"), names.index("base")
    ks = list(data["keys"])
    direct_rel, direct_cos, direct_d = [], [], []
    for k in ks:
        Wa, Wb, Wr = (models[n].get(k).astype(np.float64) for n in ("qwq", "mbpp", "base"))
        direct_rel.append(np.linalg.norm(Wa - Wr) / np.linalg.norm(Wr))
        ta, tb = (Wa - Wr).ravel(), (Wb - Wr).ravel()
        direct_cos.append(ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb))
        direct_d.append(np.linalg.norm(Wa - Wb) / np.sqrt((np.sum(Wa ** 2) + np.sum(Wb ** 2)) / 2))
    assert np.isclose(L.rel_from(G, a, r), np.mean(direct_rel), rtol=1e-4)
    assert np.isclose(L.tau_cos(G, a, b, r), np.mean(direct_cos), rtol=1e-3)
    assert np.isclose(L.distance_matrix(G)[a, b], np.mean(direct_d), rtol=1e-4)
    assert len(ks) == 21 and len(data["skipped"]) == 0


def test_recovers_realistic_tree_and_needs_refinement():
    models, truth = realistic_family(0)
    data = L.gram_pass(models, log_every=0)
    cfg = {"true_parent": truth, "generalization": ["base", "coder_base", "math_base"], "merge_sets": CFG_SETS}
    tree, rows = L.analyse(data, cfg, Path(__import__("tempfile").mkdtemp()))
    assert tree["root"] == "base"
    assert tree["parent"] == {**truth, "base": None}, tree["parent"]
    assert tree["score_refined"]["parents_correct"] == len(truth)
    by = {r["models"]: r for r in rows}
    for s in ("qwq+mbpp", "qwq+business", "business+legal", "qwq+mbpp+business"):
        assert by[s]["chosen_reference"] == "instruct" and by[s]["verdict"] == "merge"
    for s in ("coder+instruct", "coder+math", "math+mbpp"):
        assert by[s]["verdict"] == "route"
    # double counting: siblings measured from the base share instruct's update
    assert by["qwq+mbpp"]["tau_cos_at_base"] > 0.3 > abs(by["qwq+mbpp"]["tau_cos_at_chosen"])


def test_member_can_be_reference():
    models, _ = realistic_family(2)
    data = L.gram_pass({n: models[n] for n in ("base", "instruct", "qwq")}, log_every=0)
    G = data["G"]
    names = list(data["names"])
    edges = L.minimum_spanning_tree(L.distance_matrix(G))
    r, cost, verdict, _ = L.choose_reference(G, edges, [names.index("qwq"), names.index("instruct")], 0.1)
    assert names[r] == "instruct" and verdict == "merge" and cost < 0.03


def test_steiner_and_orient_on_hand_tree():
    edges = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)]
    assert L.steiner_nodes(edges, 6, [2, 3]) == [1, 2, 3]
    assert L.steiner_nodes(edges, 6, [2, 5]) == [0, 1, 2, 4, 5]
    par = L.orient(edges, 6, 0)
    assert par == {0: None, 1: 0, 4: 0, 2: 1, 3: 1, 5: 4}
    assert L.descendants(par, 1) == {2, 3}
    assert L.depth_of(par)[5] == 2


def test_cli_end_to_end_from_grams(tmp_path):
    models, truth = realistic_family(3)
    data = L.gram_pass(models, log_every=0)
    out = tmp_path / "lineage"
    out.mkdir()
    np.savez(out / "grams.npz", **data)
    plan = {"models": {n: n for n in models},
            "lineage": {"true_parent": truth, "generalization": ["base", "coder_base", "math_base"],
                        "merge_sets": CFG_SETS}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    here = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(here / "lineage.py"), str(tmp_path / "plan.json"),
                        "--out", str(out), "--from-grams"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tree = json.loads((out / "tree.json").read_text())
    assert tree["score_refined"]["parents_correct"] == 9
    assert (out / "references.csv").exists() and (out / "report.md").exists()


def test_refine_prefers_most_ancestral_near_tie():
    # 0 = base, 1 = instruct (child of 0), 2 = far model hung on instruct, but base is
    # within 1% of instruct's distance -> it should be re-hung on the base
    D = np.array([[0, 0.012, 1.000], [0.012, 0, 0.999], [1.000, 0.999, 0]])
    assert L.refine({0: None, 1: 0, 2: 1}, D, tie=0.05) == {0: None, 1: 0, 2: 0}
    assert L.refine({0: None, 1: 0, 2: 1}, D, tie=0.0) == {0: None, 1: 0, 2: 1}


def test_runs_without_generalization_flags(tmp_path):
    models, truth = realistic_family(4)
    data = L.gram_pass({n: models[n] for n in ("base", "instruct", "qwq", "mbpp")}, log_every=0)
    tree, rows = L.analyse(data, {"merge_sets": [["qwq", "mbpp"]]}, tmp_path)
    assert rows[0]["chosen_reference"] == "instruct" and rows[0]["verdict"] == "merge"
    assert set(tree["parent"]) == {"base", "instruct", "qwq", "mbpp"}


def test_edmonds_breaks_cycles():
    # cheapest incoming edges form a cycle 1 <-> 2; root 0
    W = np.array([[np.inf, 5.0, 6.0], [np.inf, np.inf, 1.0], [np.inf, 1.0, np.inf]])
    par = L.edmonds(W, 0)
    assert par in ({1: 0, 2: 1}, {2: 0, 1: 2}) and par == {1: 0, 2: 1}


def test_lineage_plan_is_consistent():
    plan = json.loads((Path(__file__).resolve().parents[1] / "plans" / "lineage.json").read_text())
    names, cfg = set(plan["models"]), plan["lineage"]
    assert set(cfg["pool"]) <= names and set(cfg["generalization"]) <= names
    assert all(c in names and p in names for c, p in cfg["true_parent"].items())
    assert all(set(s) <= names for s in cfg["merge_sets"])
    assert cfg["conventional_reference"] in names
