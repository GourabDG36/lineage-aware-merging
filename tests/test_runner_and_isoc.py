"""Checks for Iso-C and run_plan.py (no GPU, no downloads; scripts are simulated)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mergecore as mc
import run_plan


# ---------------------------------------------------------------- Iso-C

def test_iso_c_flattens_spectrum():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((30, 20))
    a, b = ref + rng.standard_normal((30, 20)), ref + rng.standard_normal((30, 20))
    out = mc.merge_tensor("iso_c", ref, [a, b], lam=0.5, alpha=1.0)
    s = np.linalg.svd(out - ref, compute_uv=False)
    s_sum = np.linalg.svd((a - ref) + (b - ref), compute_uv=False)
    assert np.allclose(s, s_sum.mean(), rtol=1e-6)          # all singular values equal their mean
    # same singular subspaces as the summed task matrix
    u_iso = np.linalg.svd(out - ref)[0][:, :20]
    u_sum = np.linalg.svd((a - ref) + (b - ref))[0][:, :20]
    assert mc.principal_angles_deg(u_iso, u_sum).max() < 1e-3


def test_iso_c_rank_deficient_and_non_matrix():
    rng = np.random.default_rng(1)
    ref = np.zeros((10, 8))
    u = rng.standard_normal((10, 1)); v = rng.standard_normal((1, 8))
    a, b = 3 * u @ v, 1 * u @ v                               # sum has rank 1
    out = mc.merge_tensor("iso_c", ref, [a, b], lam=0.5, alpha=1.0)
    assert np.allclose(out, a + b)                            # rank 1: mean of {s} is s itself
    vec_ref = np.ones(5); va, vb = vec_ref + 1, vec_ref + 3
    assert np.allclose(mc.merge_tensor("iso_c", vec_ref, [va, vb], lam=0.5, is_matrix=False), vec_ref + 2)


# ---------------------------------------------------------------- run_plan

PLAN = {
    "limit": 200, "tasks": ["gsm8k", "mbpp"],
    "models": {"base": "org/base", "instruct": "org/instruct", "good": "org/good", "bad": "org/bad"},
    "steps": [
        {"type": "eval", "model": "instruct"},
        {"type": "eval", "model": "good"},
        {"type": "eval", "model": "bad"},
        {"type": "predict", "reference": "instruct", "models": ["good", "bad"]},
        {"type": "merge", "reference": "instruct", "models": ["good", "instruct"], "method": "ties", "lam": 1.0},
        {"type": "merge", "reference": "base", "models": ["good", "bad"], "method": "ta"},
    ],
}


def write_summary(results, name, limit, tasks, acc=0.5):
    d = Path(results) / name
    d.mkdir(parents=True, exist_ok=True)
    s = {"limit": limit, **{t: {"n": limit, "accuracy": acc} for t in tasks}}
    (d / "summary.json").write_text(json.dumps(s))


def make_runner(tmp):
    plan_path = Path(tmp) / "plan.json"
    plan_path.write_text(json.dumps(PLAN))
    return run_plan.Runner(plan_path)


def test_step_names():
    a = PLAN["models"]
    assert run_plan.step_name(PLAN["steps"][0], a) == "instruct"
    assert run_plan.step_name(PLAN["steps"][3], a) == "good_bad__ref-instruct"
    assert run_plan.step_name(PLAN["steps"][4], a) == "ties-lam1__good_instruct__ref-instruct"
    assert run_plan.step_name(PLAN["steps"][5], a) == "ta__good_bad__ref-base"


def test_unknown_model_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        bad = dict(PLAN, steps=[{"type": "eval", "model": "nope"}])
        p = Path(tmp) / "plan.json"; p.write_text(json.dumps(bad))
        try:
            run_plan.Runner(p)
            raise AssertionError("should have refused")
        except SystemExit as e:
            assert "nope" in str(e)


def test_full_run_with_simulated_scripts():
    """Simulate every script: 'bad' scores 0 on the sanity check and must be skipped everywhere."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            runner = make_runner(tmp)
            calls = []

            def fake_run(name, args):
                calls.append(args)
                script, argv = args[0], [str(a) for a in args[1:]]
                get = lambda flag: argv[argv.index(flag) + 1]
                if script == "eval_model.py":
                    out_name, limit = get("--name"), int(get("--limit"))
                    tasks = argv[argv.index("--tasks") + 1:argv.index("--results-dir")]
                    acc = 0.0 if "org/bad" in get("--model") else 0.6
                    write_summary("results", out_name, limit, tasks, acc)
                elif script == "predictors.py":
                    d = Path("predictors") / get("--name"); d.mkdir(parents=True)
                    (d / "summary.json").write_text("{}")
                elif script == "merge.py":
                    d = Path(get("--out")); d.mkdir(parents=True)
                    (d / "model.safetensors").write_text("x")
                return 0

            runner.run = fake_run
            runner.run_all()
            st = runner.status
            assert st["peek:instruct"]["state"] == "OK"
            assert st["peek:bad"]["state"] == "BROKEN"
            assert st["instruct"]["state"] == "DONE" and st["good"]["state"] == "DONE"
            assert st["bad"]["state"] == "SKIPPED"
            assert st["good_bad__ref-instruct"]["state"] == "SKIPPED"
            assert st["ties-lam1__good_instruct__ref-instruct"]["state"] == "DONE"
            assert st["ta__good_bad__ref-base"]["state"] == "SKIPPED"
            # the merged weights were deleted after evaluation, results kept
            assert not Path("merged/ties-lam1__good_instruct__ref-instruct").exists()
            assert Path("results/ties-lam1__good_instruct__ref-instruct/summary.json").exists()
            # merge command carried the step's options
            merge_cmd = [c for c in calls if c[0] == "merge.py"][0]
            assert "--lam" in merge_cmd and "ties" in merge_cmd
            assert Path("backup.zip").exists() and Path("status.json").exists()

            # second run: nothing re-runs
            calls.clear()
            runner2 = make_runner(tmp)
            runner2.run = fake_run
            runner2.run_all()
            assert calls == [], calls
        finally:
            os.chdir(cwd)


def test_broken_reference_does_not_block_merges():
    """A model used only as reference must not block a merge even if its own check failed."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            plan = dict(PLAN, steps=[
                {"type": "eval", "model": "base"},
                {"type": "merge", "reference": "base", "models": ["good", "instruct"], "method": "ta"}])
            p = Path(tmp) / "plan.json"; p.write_text(json.dumps(plan))
            r = run_plan.Runner(p)

            def fake_run(name, args):
                argv = [str(a) for a in args[1:]]
                get = lambda flag: argv[argv.index(flag) + 1]
                if args[0] == "eval_model.py":
                    tasks = argv[argv.index("--tasks") + 1:argv.index("--results-dir")]
                    acc = 0.0 if get("--model") == "org/base" else 0.6
                    write_summary("results", get("--name"), int(get("--limit")), tasks, acc)
                elif args[0] == "merge.py":
                    d = Path(get("--out")); d.mkdir(parents=True); (d / "model.safetensors").write_text("x")
                return 0

            r.run = fake_run
            r.run_all()
            assert r.status["peek:base"]["state"] == "BROKEN"
            assert r.status["base"]["state"] == "SKIPPED"
            assert r.status["ta__good_instruct__ref-base"]["state"] == "DONE"
        finally:
            os.chdir(cwd)


def test_dry_run_runs_nothing():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            plan_path = Path(tmp) / "plan.json"; plan_path.write_text(json.dumps(PLAN))
            r = run_plan.Runner(plan_path, dry_run=True)
            r.run_all()
            assert not Path("status.json").exists() and not Path("results").exists()
        finally:
            os.chdir(cwd)


def test_all_plans_are_valid():
    plans = sorted((Path(__file__).resolve().parents[1] / "plans").glob("*.json"))
    assert len(plans) >= 3
    for plan in plans:
        if "steps" not in json.loads(plan.read_text()):
            continue                                       # lineage.json: read by lineage.py
        r = run_plan.Runner(plan, dry_run=True)          # rejects unknown model names
        names = [run_plan.step_name(s, r.aliases) for s in r.plan["steps"]]
        assert len(names) == len(set(names)), f"duplicate step names in {plan.name}"
        assert not any(s["type"] == "eval" and s["model"] == "base" for s in r.plan["steps"]), \
            f"{plan.name} evaluates the base model, which is not chat-tuned"


def test_collect_results():
    import collect_results
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            write_summary("results", "instruct", 200, ["gsm8k", "mbpp"], 0.5)
            write_summary("results", "good", 200, ["gsm8k", "mbpp"], 0.6)
            write_summary("results", "bad", 200, ["gsm8k", "mbpp"], 0.4)
            write_summary("results", "ta__good_bad__ref-base", 200, ["gsm8k", "mbpp"], 0.1)
            d = Path("predictors/good_bad__ref-base"); d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({
                "angle_below_chance_mean": 12.0, "cos_tau_global": 0.4,
                "variance_mlp": {"sigma_global": 0.01, "rho": 0.25}}))
            p = Path(tmp) / "plan.json"; p.write_text(json.dumps(PLAN))
            rows = collect_results.collect([p])
            assert len(rows) == 1                        # only the finished merge
            r = rows[0]
            assert r["merge"] == "ta__good_bad__ref-base"
            assert abs(r["parents_avg"] - 50.0) < 1e-9 and abs(r["delta_vs_parents"] + 40.0) < 1e-9
            assert r["collapsed"] is True                 # 10 < 0.5 * 40
            assert r["angle_below_chance_mean"] == 12.0 and r["sigma_mlp"] == 0.01
            collect_results.write_outputs(rows)
            assert Path("results_table.csv").exists() and Path("results_table.md").exists()
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
