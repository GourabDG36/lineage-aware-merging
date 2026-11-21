"""Checks for audit_hub.py with a fake Hugging Face (no network)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_hub as A

BASE, INST = "qwen/qwen2.5-1.5b", "qwen/qwen2.5-1.5b-instruct"


def ft(*parents, rel="finetune", params=1.5e9, downloads=10):
    return {"tags": [f"base_model:{rel}:{p}" for p in parents], "base_model": None,
            "base_model_relation": None, "downloads": downloads, "likes": 0, "params": params}


META = {
    BASE: {"tags": [], "base_model": None, "base_model_relation": None},
    INST: ft("Qwen/Qwen2.5-1.5B"),
    "a/qwq": ft(INST), "b/mbpp": ft(INST), "c/biz": ft(INST),
    "unsloth/qwen2.5-1.5b": ft(BASE),                                   # re-upload of the base
    "qwen/coder-base": ft(BASE), "d/coder": ft("qwen/coder-base"),
    "e/mergedab": ft("a/qwq", "b/mbpp", rel="merge"),
    "other/unrelated": {"tags": []}, "other/unrelated2": {"tags": []},
    "g/cardonly": {"tags": [], "base_model": "Qwen/Qwen2.5-1.5B-Instruct"},  # only card base_model
    "f/nometa": None,
}


def recipe(method, base, models, weights=None, normalize=None):
    lines = [f"merge_method: {method}"] + ([f"base_model: {base}"] if base else []) + ["models:"]
    for i, m in enumerate(models):
        w = (weights or [1.0] * len(models))[i]
        lines += [f"  - model: {m}", f"    parameters: {{weight: {w}}}"]
    if normalize is not None:
        lines += ["parameters:", f"  normalize: {str(normalize).lower()}"]
    return "\n".join(lines) + "\n"


RECIPES = {
    "m/dc-high": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["a/qwq", "b/mbpp", "c/biz"]),
    "m/ok": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B-Instruct", ["a/qwq", "b/mbpp"]),
    "m/ties": recipe("ties", "Qwen/Qwen2.5-1.5B", ["a/qwq", "b/mbpp"]),
    "m/norm": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["a/qwq", "b/mbpp"], normalize=True),
    "m/half": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["a/qwq", "b/mbpp"], [0.5, 0.5]),
    "m/far": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["a/qwq", "d/coder"]),
    "m/mirror": recipe("task_arithmetic", "unsloth/Qwen2.5-1.5B", ["a/qwq", "b/mbpp"]),
    "m/mirror-ok": recipe("task_arithmetic", "unsloth/Qwen2.5-1.5B", ["a/qwq", "d/coder"]),
    "m/unrelated": recipe("task_arithmetic", "other/unrelated", ["a/qwq", "b/mbpp"]),
    "m/merged-ingredient": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["e/mergedab", "c/biz"]),
    "m/cardonly": recipe("dare_ties", "Qwen/Qwen2.5-1.5B", ["g/cardonly", "a/qwq"]),
    "m/unknown": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B-Instruct", ["a/qwq", "f/nometa"]),
    "m/slerp": "slices:\n  - sources:\n      - model: a/qwq\n        layer_range: [0, 28]\n      - model: b/mbpp\n        layer_range: [0, 28]\nmerge_method: slerp\nbase_model: a/qwq\n",
    "m/single": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["a/qwq"]),
    "m/refs": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B-Instruct", ["a/qwq@main", "b/mbpp+x/some-lora"]),
    "m/multistage": "- merge_method: linear\n  models: [{model: a/qwq}]\n- merge_method: ties\n  base_model: Qwen/Qwen2.5-1.5B\n  models:\n    - model: a/qwq\n    - model: c/biz\n",
    "m/no-common": recipe("task_arithmetic", "other/unrelated", ["a/qwq", "other/unrelated2"]),
    "m/readme": None, "m/noconfig": None,
}
README = {"m/readme": "# My merge\n\n```yaml\nmerge_method: ties\nbase_model: Qwen/Qwen2.5-1.5B\nmodels:\n  - model: b/mbpp\n  - model: c/biz\n```\n"}


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.cache = {"meta": {}, "recipe": {}}

    def meta(self, repo):
        self.calls += 1
        if repo.startswith("m/"):
            return {"tags": ["mergekit"], "downloads": 100, "likes": 1, "params": 1.5e9}
        return META.get(repo)

    def recipe_text(self, repo):
        if RECIPES.get(repo) is not None:
            return RECIPES[repo]
        return A.recipe_from_readme(README.get(repo, ""))

    def save(self):
        pass


def run(repo):
    c = FakeClient()
    return A.audit_one(repo, c, A.Ancestry(c))


def test_verdicts_cover_every_case():
    expect = {
        "m/dc-high": ("DOUBLE_COUNT", "high"), "m/ok": ("OK", None), "m/ties": ("DOUBLE_COUNT", "lower"),
        "m/norm": ("DOUBLE_COUNT", "lower"), "m/half": ("DOUBLE_COUNT", "lower"), "m/far": ("OK", None),
        "m/mirror": ("DOUBLE_COUNT", "high"), "m/mirror-ok": ("OK", None),
        "m/unrelated": ("BASE_NOT_ANCESTOR", None), "m/merged-ingredient": ("DOUBLE_COUNT", "high"),
        "m/cardonly": ("DOUBLE_COUNT", "lower"), "m/unknown": ("UNKNOWN", None),
        "m/slerp": ("NOT_TASK_VECTOR", None), "m/single": ("SINGLE_INGREDIENT", None),
        "m/refs": ("OK", None), "m/multistage": ("DOUBLE_COUNT", "lower"),
        "m/no-common": ("INCOMPLETE_LINEAGE", None), "m/readme": ("DOUBLE_COUNT", "lower"),
        "m/noconfig": ("NO_CONFIG", None),
    }
    for repo, (verdict, risk) in expect.items():
        row = run(repo)
        assert row["verdict"] == verdict, (repo, row)
        assert row.get("risk") == risk, (repo, row)


def test_details_of_the_main_mistake():
    row = run("m/dc-high")
    assert row["closest_common_ancestor"] == INST and row["declared_base"] == BASE
    assert row["shared_update_counted"] == 3 and row["cca_distance"] == 1 and row["base_distance"] == 2
    assert run("m/merged-ingredient")["closest_common_ancestor"] == INST   # a merge descends from what its parents share
    assert "re-upload of qwen/qwen2.5-1.5b" in run("m/mirror")["notes"]
    refs = run("m/refs")
    assert refs["ingredients"] == "a/qwq b/mbpp" and "LoRA" in refs["notes"] and "pinned revision" in refs["notes"]
    assert "no metadata: f/nometa" in run("m/unknown")["notes"]


def test_clean_ref_and_parsing():
    assert A.clean_ref("Org/Model") == ("org/model", "")
    assert A.clean_ref("./local/model")[0] is None
    assert A.clean_ref({"model": "a/b@rev"}) == ("a/b", "pinned revision")
    r = A.parse_recipe(RECIPES["m/slerp"])
    assert r["format"] == "slices" and [i for i, _ in r["ingredients"]] == ["b/mbpp"]   # base dropped
    assert A.parse_recipe("not: a merge") is None and A.parse_recipe(":: bad yaml ::\n -") is None
    g = A.parse_recipe("merge_method: task_arithmetic\nbase_model: x/y\nmodels:\n - model: a/b\n   parameters: {weight: [0.2, 0.4]}\n")
    assert abs(g["ingredients"][0][1] - 0.3) < 1e-9


def test_parents_from_meta():
    assert A.parents_from_meta({"tags": ["base_model:finetune:Qwen/X", "base_model:quantized:Y/Z"]}) == ("finetune", ["qwen/x"])
    assert A.parents_from_meta({"tags": ["base_model:merge:a/b", "base_model:merge:c/d"]}) == ("merge", ["a/b", "c/d"])
    assert A.parents_from_meta({"tags": [], "base_model": ["A/B", "C/D"]}) == ("merge", ["a/b", "c/d"])
    assert A.parents_from_meta({"tags": []}) == (None, [])
    assert A.parents_from_meta(None) == (None, [])


def test_summary_files(tmp_path):
    rows = [run(r) for r in RECIPES]
    v = A.summarise(rows, tmp_path, "downloads", 99)
    assert v["DOUBLE_COUNT"] == 9 and v["OK"] == 4 and v["INCOMPLETE_LINEAGE"] == 1
    text = (tmp_path / "summary.md").read_text()
    assert "DOUBLE_COUNT: 9 of 14" in text and "incomplete lineage 1" in text and "high risk" in text
    with (tmp_path / "audit.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(RECIPES)
    with (tmp_path / "verify_candidates.csv").open() as f:
        cands = list(csv.DictReader(f))
    assert len(cands) == 9 and cands[0]["risk"] == "high"


def test_lineage_loops():
    RECIPES.update({
        "m/loop-inst": recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["Qwen/Qwen2.5-1.5B-Instruct", "a/qwq"]),
        "m/loop-merge": recipe("ties", "Qwen/Qwen2.5-1.5B-Instruct", ["e/mergedab", "a/qwq", "c/biz"]),
        "m/loop-slerp": "merge_method: slerp\nbase_model: a/qwq\nslices:\n  - sources:\n      - model: e/mergedab\n      - model: b/mbpp\n",
    })
    try:
        r = run("m/loop-inst")
        assert r["n_loops"] == 1 and r["lineage_loops"] == f"{INST} > a/qwq"
        assert r["verdict"] == "DOUBLE_COUNT" and r["closest_common_ancestor"] == INST   # ingredient is the CCA
        r = run("m/loop-merge")          # a/qwq is a parent of the merged ingredient e/mergedab
        assert r["n_loops"] == 1 and r["lineage_loops"] == "a/qwq > e/mergedab" and r["verdict"] == "OK"
        r = run("m/loop-slerp")          # loops are checked for every method, not only task vectors
        assert r["verdict"] == "NOT_TASK_VECTOR" and r["lineage_loops"] == "b/mbpp > e/mergedab"
        assert run("m/dc-high")["n_loops"] == 0 and run("m/dc-high")["loops_checked"] is True
        assert run("m/unknown")["loops_checked"] is False
        rows = [run(k) for k in ("m/loop-inst", "m/loop-merge", "m/loop-slerp", "m/dc-high", "m/ok")]
        import tempfile
        out = Path(tempfile.mkdtemp())
        A.summarise(rows, out, "downloads", 5)
        text = (out / "summary.md").read_text()
        assert "With a lineage loop: 3 (60.0%)" in text and "Both mistakes (DOUBLE_COUNT base and a loop): 1" in text
    finally:
        for k in ("m/loop-inst", "m/loop-merge", "m/loop-slerp"):
            RECIPES.pop(k, None)


def test_more_recipe_formats():
    multi = ("name: stage1\nmerge_method: ties\nbase_model: x/base\nmodels:\n  - model: a/one\n  - model: a/two\n"
             "---\nname: final\nmerge_method: task_arithmetic\nbase_model: x/base\nmodels:\n  - model: stage1\n  - model: a/three\n")
    r = A.parse_recipe(multi)                       # several YAML documents: the final stage counts
    assert r["method"] == "task_arithmetic" and [i for i, _ in r["ingredients"]] == ["a/three"]
    crlf = "merge_method: ties\r\nbase_model: x/base\r\nmodels:\r\n  - model: a/one\r\n  - model: a/two\r\n"
    assert [i for i, _ in A.parse_recipe(crlf)["ingredients"]] == ["a/one", "a/two"]
    tilde = "Intro\n\n~~~yml\nmerge_method: dare_ties\nbase_model: x/base\nmodels:\n  - model: a/one\n  - model: a/two\n~~~\n"
    assert A.parse_recipe(A.recipe_from_readme(tilde))["method"] == "dare_ties"
    lang = "```YAML title=config\nmerge_method: ties\nbase_model: x/base\nmodels:\n  - model: a/one\n  - model: a/two\n```"
    assert A.parse_recipe(A.recipe_from_readme(lang))["method"] == "ties"
    bare = ("# Card\n\nThe following configuration was used:\n\nmodels:\n  - model: a/one\n    parameters:\n"
            "      weight: 0.5\n  - model: a/two\nmerge_method: task_arithmetic\nbase_model: x/base\ndtype: bfloat16\n\nThanks!\n")
    r = A.parse_recipe(A.recipe_from_readme(bare))  # no code fence at all
    assert r["method"] == "task_arithmetic" and len(r["ingredients"]) == 2
    assert A.recipe_from_readme("# nothing here\n") is None
    assert A.load_merge_doc("a: [1, 2\n") is None and A.load_merge_doc(None) is None


def test_quantized_copies_followed_to_the_original_merge():
    extra = {
        "mradermacher/dc-high-gguf": {"tags": ["base_model:quantized:m/dc-high"]},
        "bartowski/dc-high-gguf": {"tags": [], "base_model": "m/dc-high"},          # name says GGUF, no relation tag
        "someone/dc-high-exl2-4bpw": {"tags": [], "base_model": "mradermacher/dc-high-gguf"},  # copy of a copy
        "x/real-finetune": {"tags": [], "base_model": "m/dc-high"},                  # a fine-tune, not a copy
    }
    META.update(extra)
    try:
        c = FakeClient()
        c.meta = lambda r: {"tags": ["mergekit"], "downloads": 1} if r.startswith("m/") else META.get(r)
        assert A.resolve_source("mradermacher/dc-high-gguf", c) == ("m/dc-high", ["mradermacher/dc-high-gguf"])
        assert A.resolve_source("bartowski/dc-high-gguf", c)[0] == "m/dc-high"
        assert A.resolve_source("someone/dc-high-exl2-4bpw", c)[0] == "m/dc-high"
        assert A.resolve_source("x/real-finetune", c)[0] == "x/real-finetune"
        order, copies = A.plan_sources(["mradermacher/dc-high-gguf", "m/ok", "bartowski/dc-high-gguf", "m/dc-high"], c)
        assert order == ["m/dc-high", "m/ok"] and len(copies["m/dc-high"]) == 3
    finally:
        for k in extra:
            META.pop(k, None)


def test_parents_read_from_recipe_when_card_lists_none():
    META["h/merged-nocard"] = {"tags": ["mergekit"]}             # a community merge whose card lists no parent
    RECIPES["h/merged-nocard"] = recipe("ties", "Qwen/Qwen2.5-1.5B-Instruct", ["a/qwq", "b/mbpp"])
    RECIPES["m/via-recipe"] = recipe("task_arithmetic", "Qwen/Qwen2.5-1.5B", ["h/merged-nocard", "c/biz"])
    try:
        c = FakeClient()
        c.meta = lambda r: {"tags": ["mergekit"], "downloads": 1} if r.startswith("m/") else META.get(r)
        anc = A.Ancestry(c)
        a = anc.ancestors("h/merged-nocard")
        assert INST in a and BASE in a and "a/qwq" not in a      # full ancestry = what all its parents share
        assert {"a/qwq", "b/mbpp", INST} <= anc.any_ancestors("h/merged-nocard")
        row = A.audit_one("m/via-recipe", c, anc)
        assert row["verdict"] == "DOUBLE_COUNT" and row["closest_common_ancestor"] == INST
        assert "parents read from recipe: h/merged-nocard" in row["notes"]
    finally:
        META.pop("h/merged-nocard", None)
        RECIPES.pop("h/merged-nocard", None)
        RECIPES.pop("m/via-recipe", None)
