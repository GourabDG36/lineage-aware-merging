"""Checks for base_checker.py on the synthetic Qwen-like family of test_lineage (no torch needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_checker as bc
import lineage as L
from test_lineage import realistic_family

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grams():
    models, _ = realistic_family(0)
    return L.gram_pass(models, log_every=0)


def sub(data, pool):
    names = [str(n) for n in data["names"]]
    idx = [names.index(p) for p in pool]
    return {"names": np.array(pool), "keys": data["keys"], "skipped": data["skipped"],
            "G": data["G"][:, idx][:, :, idx], "kurt": data["kurt"][:, idx]}


def cfg(method, base, models, weights=None, **glob):
    c = {"merge_method": method, "base_model": base,
         "models": [{"model": m, "parameters": ({"weight": w} if w is not None else {})}
                    for m, w in zip(models, weights or [1.0] * len(models))]}
    if glob:
        c["parameters"] = glob
    return c


def run(data, c, extra=()):
    parsed = bc.parse_config(c)
    pool = [m for m, _ in parsed["models"]] + [parsed["base"]] + list(extra)
    pool = list(dict.fromkeys(pool))
    return bc.check(sub(data, pool), parsed)


def test_wrong_base_caught_for_task_arithmetic_sum(grams):
    rep = run(grams, cfg("task_arithmetic", "base", ["qwq", "mbpp", "business"]), extra=["instruct"])
    assert rep["verdict"] == "WRONG_BASE"
    assert rep["closest_common_ancestor"] == "instruct"
    assert rep["ancestor_update_counted_times"] == 3
    assert rep["task_vector_cos_from_declared_base"] > 0.3 > abs(rep["task_vector_cos_from_ancestor"])
    assert "instruct" in rep["message"] and "3 times" in rep["message"]


def test_correct_base_is_ok(grams):
    rep = run(grams, cfg("task_arithmetic", "instruct", ["qwq", "mbpp"]), extra=["base"])
    assert rep["verdict"] == "OK" and rep["closest_common_ancestor"] == "instruct"


def test_ties_wrong_base_is_only_a_warning(grams):
    rep = run(grams, cfg("ties", "base", ["qwq", "mbpp"], [1.0, 1.0], density=0.2), extra=["instruct"])
    assert rep["verdict"] == "WARNING" and "trims" in rep["message"]


def test_normalised_weights_are_only_a_warning(grams):
    rep = run(grams, cfg("task_arithmetic", "base", ["qwq", "mbpp"], [0.5, 0.5]), extra=["instruct"])
    assert rep["ancestor_update_counted_times"] == 1.0
    assert rep["verdict"] == "WARNING" and "normalised" in rep["message"]
    rep = run(grams, cfg("task_arithmetic", "base", ["qwq", "mbpp"], [1.0, 1.0], normalize=True), extra=["instruct"])
    assert rep["verdict"] == "WARNING"


def test_linear_does_not_use_base(grams):
    rep = run(grams, cfg("linear", "base", ["qwq", "mbpp"], [0.5, 0.5]), extra=["instruct"])
    assert rep["verdict"] == "INFO"


def test_distant_models_are_refused(grams):
    for models in (["coder", "instruct"], ["coder", "math"], ["math", "mbpp"]):
        rep = run(grams, cfg("task_arithmetic", "base", models), extra=["coder_base", "math_base"])
        assert rep["verdict"] == "TOO_FAR", models
        assert "unsafe" in rep["message"]


def test_without_the_ancestor_in_the_pool_no_false_alarm(grams):
    # if instruct is not available, base is the best base the tool can see
    rep = run(grams, cfg("task_arithmetic", "base", ["qwq", "mbpp"]))
    assert rep["closest_common_ancestor"] in ("base", "qwq", "mbpp")
    assert rep["verdict"] in ("OK", "WARNING", "WRONG_BASE")


def test_parse_config_details():
    c = {"merge_method": "task_arithmetic", "base_model": "b",
         "models": [{"model": "b"}, {"model": "x", "parameters": {"weight": [0.2, 0.4]}}, {"model": "y"}]}
    p = bc.parse_config(c)
    assert [m for m, _ in p["models"]] == ["x", "y"]          # base listed among models is dropped
    assert p["models"][0][1] == pytest.approx(0.3)
    assert any("gradient" in n for n in p["notes"]) and any("default 1.0" in n for n in p["notes"])
    with pytest.raises(ValueError):
        bc.parse_config({"merge_method": "slerp", "slices": [{}]})
    assert bc.fixed_config({"base_model": {"model": "a"}}, "z")["base_model"]["model"] == "z"


def test_cli_with_cache_writes_fixed_config(tmp_path, grams):
    names = [str(n) for n in grams["names"]]
    aliases = {n: f"org/{n}" for n in names}                   # cache uses aliases, config uses repo ids
    np.savez(tmp_path / "grams.npz", **grams)
    (tmp_path / "plan.json").write_text(json.dumps({"models": aliases}))
    c = cfg("task_arithmetic", "org/base", ["org/qwq", "org/mbpp", "org/business"])
    (tmp_path / "merge.yml").write_text(yaml.safe_dump(c, sort_keys=False))
    r = subprocess.run([sys.executable, str(HERE / "base_checker.py"), "--config", str(tmp_path / "merge.yml"),
                        "--candidates", "org/instruct", "--cache", str(tmp_path / "grams.npz"),
                        "--aliases", str(tmp_path / "plan.json"), "--out", str(tmp_path / "fixed.yml"),
                        "--json", str(tmp_path / "rep.json")], capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr                 # 2 = WRONG_BASE
    assert "[FAIL] WRONG_BASE" in r.stdout
    fixed = yaml.safe_load((tmp_path / "fixed.yml").read_text())
    assert fixed["base_model"] == "org/instruct" and fixed["models"] == c["models"]
    assert json.loads((tmp_path / "rep.json").read_text())["verdict"] == "WRONG_BASE"
