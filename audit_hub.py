"""
Audit public model merges on Hugging Face: how often is `base_model` not the closest common
ancestor of the merged models (the double-counting mistake)?

    python audit_hub.py --limit 1500                 # most-downloaded mergekit models
    python audit_hub.py --limit 500 --sort created_at --out audit_recent
    python audit_hub.py --summarise-only             # rebuild the summary from the saved cache

Stage 1 (this script, CPU, text only, no model weights):
  1. List models tagged `mergekit` on the Hub.
  2. Read each one's merge recipe: `mergekit_config.yml`, or a ```yaml block with
     `merge_method` in its README.
  3. Build each ingredient's ancestry from model-card metadata (the Hub's
     base_model:finetune / adapter / quantized / merge tags, or card base_model), up to
     --depth levels. A merged ingredient descends from what all its parents share.
  4. Closest common ancestor (CCA) of the ingredients = the shared ancestor nearest to all
     of them. Compare it with the declared base_model.

Verdicts (task-vector methods with 2+ ingredients):
  OK                 declared base is the CCA
  DOUBLE_COUNT       the ingredients share a closer ancestor than the declared base, so
                     every task vector carries that ancestor's update. Risk "high" for
                     summing methods with total weight > 1.05 (as in our 6.5/0.0 merge),
                     "lower" for TIES-style or normalised weights.
  BASE_NOT_ANCESTOR  metadata says the declared base is not an ancestor of every ingredient
                     (re-uploads/mirrors of the same base also land here; see notes)
  NO_COMMON_ANCESTOR metadata gives the ingredients no shared ancestor
  UNKNOWN            missing metadata for an ingredient or the base
  LOCAL_PATHS        the recipe names local folders that cannot be matched to Hub models
                     ('./Org/Model', 'E:\\dir\\Org_Model' and 'models--Org--Model' are matched)
  INCOMPLETE_LINEAGE an ingredient's card lists no parent (a true root or missing
                     metadata), so the verdict cannot be judged; excluded from percentages
Quantized copies (GGUF, EXL2, AWQ, ...) inherit the original's `mergekit` tag; they are followed
back to the original merge (card base_model / quantized relation) and each merge is audited once.
Other rows: NOT_TASK_VECTOR (linear, slerp, passthrough ...), SINGLE_INGREDIENT, NO_CONFIG,
UNPARSEABLE.

Lineage loops (any method, 2+ ingredients): one ingredient is a (partial) ancestor of another,
e.g. merging A with a merge that already contains A, so A's contribution enters twice. A
correct base does not fix this. Partial ancestry includes every parent of a merge.

Links that cards leave out are filled in two transparent ways (noted per row): an official
instruct/chat model is linked to its base by name when that base exists (Meta, Mistral, Google, ...),
and a parentless re-upload is linked to the official model of the same name.

Card metadata is self-reported, so this is a screen. Stage 2 checks a sample with the
weight-based Base Checker (base_checker.py), using the CCA found here as the candidate.

Outputs in --out (default audit/): audit.csv (one row per merged model), summary.md,
verify_candidates.csv (small DOUBLE_COUNT merges for weight checks), cache.json (resumable).
Set HF_TOKEN for higher Hub rate limits.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

TASK_VECTOR = {
    "task_arithmetic": "sum", "dare_linear": "sum", "breadcrumbs": "sum", "della_linear": "sum",
    "ties": "trim", "dare_ties": "trim", "breadcrumbs_ties": "trim", "della": "trim", "sce": "trim",
    "model_stock": "other",
}
PARENT_RELATIONS = ("finetune", "adapter", "quantized", "merge")


# ------------------------------------------------------------------ references and recipes

def clean_ref(ref):
    """mergekit model reference -> (lower-case repo id or None, note).
    Handles 'repo@revision', 'repo+lora_repo' (LoRA applied on top) and local paths."""
    if isinstance(ref, dict):
        ref = ref.get("model")
    if not isinstance(ref, str) or not ref.strip():
        return None, "missing model name"
    r = ref.strip()
    note = ""
    if "+" in r:
        r, adapter = r.split("+", 1)
        note = f"LoRA {adapter} applied"
    if "@" in r:
        r = r.split("@", 1)[0]
        note = (note + "; " if note else "") + "pinned revision"
    if r.startswith((".", "/", "~")) or r.count("/") != 1 or " " in r:
        return None, f"local or invalid path {ref!r}"
    return r.lower(), note


def _num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return float(sum(v) / len(v))
    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
        return float(v["value"])
    return None


def load_merge_doc(text):
    """The final merge stage in a recipe text, or None. Handles single documents, lists of
    stages, several YAML documents separated by '---' (multi-stage mergekit recipes), and
    Windows line endings."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n").replace("\t", "  ")
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        docs = []
        for piece in re.split(r"^---[ \t]*$", text, flags=re.M):
            try:
                docs.append(yaml.safe_load(piece))
            except yaml.YAMLError:
                continue
    flat = []
    for d in docs:
        flat.extend(d if isinstance(d, list) else [d])
    stages = [d for d in flat if isinstance(d, dict) and "merge_method" in d]
    return stages[-1] if stages else None


def parse_recipe(text):
    """mergekit YAML text -> dict or None. Supports the `models` and `slices` formats."""
    cfg = load_merge_doc(text)
    if cfg is None:
        return None
    method = str(cfg.get("merge_method", "")).strip().lower()
    glob = cfg.get("parameters") if isinstance(cfg.get("parameters"), dict) else {}
    base_raw = cfg.get("base_model")
    if isinstance(base_raw, dict):
        base_raw = base_raw.get("model")
    base, base_note = clean_ref(base_raw) if base_raw is not None else (None, "")
    if base is None:
        base_note = ""
    raw = []
    if isinstance(cfg.get("models"), list):
        for e in cfg["models"]:
            params = (e.get("parameters") or {}) if isinstance(e, dict) else {}
            raw.append((e.get("model") if isinstance(e, dict) else e, params))
        fmt = "models"
    elif isinstance(cfg.get("slices"), list):
        for sl in cfg["slices"]:
            for src in (sl.get("sources") or []) if isinstance(sl, dict) else []:
                if isinstance(src, dict):
                    raw.append((src.get("model"), src.get("parameters") or {}))
        fmt = "slices"
    else:
        fmt = "none"
    seen, ingredients, notes, unresolved = {}, [], [], []
    for ref, params in raw:
        rid, note = clean_ref(ref)
        w = _num(params.get("weight")) if isinstance(params, dict) else None
        if w is None:
            w = _num(glob.get("weight"))
        w = 1.0 if w is None else w
        if rid is None:
            raw_ref = ref.get("model") if isinstance(ref, dict) else ref
            if isinstance(raw_ref, str) and raw_ref.strip() and raw_ref not in [u for u, _ in unresolved]:
                unresolved.append((raw_ref.strip(), w))
            continue
        if note:
            notes.append(note)
        if rid == base:
            continue
        if rid not in seen:
            seen[rid] = len(ingredients)
            ingredients.append([rid, w])
    normalize = glob.get("normalize")
    if normalize is None:
        normalize = method == "linear"
    return {"method": method, "kind": TASK_VECTOR.get(method), "base": base, "base_note": base_note,
            "base_unresolved": base_raw.strip() if base is None and isinstance(base_raw, str) and base_raw.strip() else None,
            "ingredients": ingredients, "unresolved": unresolved, "normalize": bool(normalize),
            "format": fmt, "notes": notes, "slices": fmt == "slices"}


YAMLISH = re.compile(r"^\s*$|^\s*#|^\s*-\s|^\s*[\w.@/+-]+\s*:")


def recipe_from_readme(text):
    """A mergekit recipe inside a README: a fenced block (``` or ~~~, any language tag) that
    contains merge_method, else the run of YAML-looking lines around a 'merge_method:' line."""
    text = (text or "").replace("\r\n", "\n")
    for block in re.findall(r"(?:```|~~~)[^\n]*\n(.*?)(?:```|~~~)", text, flags=re.S):
        if "merge_method" in block and parse_recipe(block) is not None:
            return block
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if re.match(r"^\s*merge_method\s*:", line):
            a = k
            while a > 0 and YAMLISH.match(lines[a - 1]) and not lines[a - 1].lstrip().startswith(("```", "~~~")):
                a -= 1
            b = k
            while b + 1 < len(lines) and YAMLISH.match(lines[b + 1]):
                b += 1
            block = "\n".join(lines[a:b + 1])
            if parse_recipe(block) is not None:
                return block
    return None


# ------------------------------------------------------------------ ancestry from metadata

def parents_from_meta(meta):
    """meta = {'tags': [...], 'base_model': str|list|None, 'base_model_relation': str|None}
    -> (relation, [parent ids]) or (None, [])."""
    if meta is None:
        return None, []
    by_rel = {}
    for t in meta.get("tags") or []:
        m = re.match(r"base_model:(?:(finetune|adapter|quantized|merge):)?(.+)$", str(t))
        if m:
            by_rel.setdefault(m.group(1) or "unspecified", []).append(m.group(2).lower())
    for rel in ("merge", "finetune", "adapter", "quantized"):
        if by_rel.get(rel):
            return rel, sorted(set(by_rel[rel]))
    bm = meta.get("base_model")
    lst = [bm] if isinstance(bm, str) else [b for b in (bm or []) if isinstance(b, str)]
    if by_rel.get("unspecified"):
        lst = by_rel["unspecified"]
    lst = sorted({b.lower() for b in lst})
    if not lst:
        return None, []
    rel = meta.get("base_model_relation") or ("merge" if len(lst) > 1 else "finetune")
    return rel, lst


OFFICIAL_ORGS = ("meta-llama", "mistralai", "google", "qwen", "microsoft", "nvidia", "deepseek-ai",
                 "ibm-granite", "tiiuae", "01-ai", "internlm", "allenai", "huggingfacetb")
MIRROR_ORGS = ("meta-llama", "mistralai", "google", "qwen", "microsoft")


def local_path_candidates(raw):
    """Hugging Face ids a local mergekit path may stand for, most likely first:
    './Org/Model' -> org/model; 'E:\\dir\\Org_Model' -> org/model; '.../models--Org--Model/...' -> org/model."""
    p = raw.strip().replace("\\", "/").strip("/")
    p = re.sub(r"^[A-Za-z]:/", "", p)
    out = []
    m = re.search(r"models--([^/]+?)--([^/]+)", p)
    if m:
        out.append(f"{m.group(1)}/{m.group(2)}")
    parts = [x for x in p.split("/") if x not in ("", ".", "..")]
    if len(parts) >= 2:
        out.append("/".join(parts[-2:]))
    if parts:
        last = parts[-1]
        for sep in ("--", "_"):
            if sep in last:
                a, b = last.split(sep, 1)
                if a and b:
                    out.append(f"{a}/{b}")
    seen, res = set(), []
    for c in out:
        c = c.lower()
        if c.count("/") == 1 and " " not in c and c not in seen:
            seen.add(c)
            res.append(c)
    return res


def resolve_local(raw, client):
    for c in local_path_candidates(raw):
        if client.meta(c) is not None:
            return c
    return None


def official_parent(repo, client):
    """Links model cards often leave out, inferred by name and kept only if the model exists:
    an official instruct/chat model -> its base (e.g. meta-llama/Meta-Llama-3-8B-Instruct ->
    meta-llama/Meta-Llama-3-8B; google/gemma-3-27b-it -> google/gemma-3-27b-pt), and a re-upload
    with no listed parent -> the official model of the same name (NousResearch/Meta-Llama-3-8B ->
    meta-llama/Meta-Llama-3-8B). Returns (parent, note) or (None, None)."""
    org, _, name = repo.partition("/")
    if org in OFFICIAL_ORGS:
        cands = []
        for pat, rep in ((r"-instruct(?=$|-)", ""), (r"-instruct(?=$|-)", "-base"), (r"-it(?=$|-)", "-pt"),
                         (r"-it(?=$|-)", ""), (r"-chat(?=$|-)", "")):
            if re.search(pat, name):
                cands.append(f"{org}/{re.sub(pat, rep, name, count=1)}")
        for c in dict.fromkeys(cands):
            if c != repo and client.meta(c) is not None:
                return c, f"official base inferred: {repo} -> {c}"
        return None, None
    for off in MIRROR_ORGS:
        c = f"{off}/{name}"
        if client.meta(c) is not None:
            return c, f"re-upload of {c}"
    return None, None


class Ancestry:
    """ancestors(repo) -> {ancestor: distance}, repo itself at 0; None if repo metadata is missing."""

    def __init__(self, client, depth=6):
        self.client, self.depth, self.memo = client, depth, {}
        self.parent_memo, self.from_recipe, self.inferred = {}, set(), {}

    def parents(self, repo, meta):
        """(relation, parents) from card metadata; if the card lists none, from the model's own
        mergekit recipe (many community merges list their parents only there)."""
        if repo in self.parent_memo:
            return self.parent_memo[repo]
        rel, parents = parents_from_meta(meta)
        if not parents and hasattr(self.client, "recipe_text"):
            r = parse_recipe(self.client.recipe_text(repo) or "")
            if r is not None:
                found = [i for i, _ in r["ingredients"]] + ([r["base"]] if r["base"] else [])
                found = sorted(set(found) - {repo})
                if found:
                    rel, parents = ("merge" if len(found) > 1 else "finetune"), found
                    self.from_recipe.add(repo)
        if not parents:
            p, note = official_parent(repo, self.client)
            if p:
                rel, parents = "finetune", [p]
                self.inferred[repo] = note
        self.parent_memo[repo] = (rel, parents)
        return rel, parents

    def ancestors(self, repo, _stack=()):
        if repo in self.memo:
            return self.memo[repo]
        if repo in _stack or len(_stack) > self.depth:
            return {repo: 0}
        meta = self.client.meta(repo)
        if meta is None:
            self.memo[repo] = None
            return None
        rel, parents = self.parents(repo, meta)
        out = {repo: 0}
        parent_sets = []
        for p in parents:
            a = self.ancestors(p, _stack + (repo,))
            parent_sets.append(a if a is not None else {p: 0})
        if parent_sets:
            if rel == "merge" and len(parent_sets) > 1:
                shared = set(parent_sets[0]).intersection(*parent_sets[1:])
                for a in shared:
                    out[a] = 1 + max(ps[a] for ps in parent_sets)
            else:
                for ps in parent_sets:
                    for a, d in ps.items():
                        out[a] = min(out.get(a, 99), d + 1)
        self.memo[repo] = out
        return out

    def any_ancestors(self, repo, _stack=()):
        """Every model this repo descends from, even partly (all parents of merges included).
        None if repo metadata is missing."""
        memo = self.__dict__.setdefault("memo_any", {})
        if repo in memo:
            return memo[repo]
        if repo in _stack or len(_stack) > self.depth:
            return set()
        meta = self.client.meta(repo)
        if meta is None:
            memo[repo] = None
            return None
        out = set()
        for p in self.parents(repo, meta)[1]:
            out.add(p)
            out |= self.any_ancestors(p, _stack + (repo,)) or set()
        memo[repo] = out
        return out


def lineage_loops(ingredients, ancestry):
    """Pairs (ancestor, descendant) where one ingredient is a (partial) ancestor of another:
    the ancestor's contribution enters the merge twice. None if any ingredient lacks metadata."""
    anys = [ancestry.any_ancestors(i) for i in ingredients]
    if any(a is None for a in anys):
        return None
    return [(a, d) for d, anc in zip(ingredients, anys) for a in ingredients if a != d and a in anc]


def closest_common_ancestor(anc_list):
    shared = set(anc_list[0]).intersection(*anc_list[1:]) if anc_list else set()
    if not shared:
        return None, None
    best = min(shared, key=lambda a: (max(an[a] for an in anc_list), sum(an[a] for an in anc_list), a))
    return best, max(an[best] for an in anc_list)


# ------------------------------------------------------------------ audit one merge

def audit_one(repo, client, ancestry):
    row = {"repo": repo}
    info = client.meta(repo) or {}
    for k in ("downloads", "likes", "params", "created_at"):
        row[k] = info.get(k)
    text = client.recipe_text(repo)
    if text is None:
        row["verdict"] = "NO_CONFIG"
        return row
    r = parse_recipe(text)
    if r is None:
        row["verdict"] = "UNPARSEABLE"
        return row
    local_notes, left = [], []
    for raw_ref, w in r["unresolved"]:
        c = resolve_local(raw_ref, client)
        if c and c != r["base"] and c not in [i for i, _ in r["ingredients"]]:
            r["ingredients"].append([c, w])
            local_notes.append(f"local path {raw_ref!r} -> {c}")
        elif not c:
            left.append(raw_ref)
    if r["base"] is None and r["base_unresolved"]:
        c = resolve_local(r["base_unresolved"], client)
        if c:
            r["base"] = c
            r["ingredients"] = [x for x in r["ingredients"] if x[0] != c]
            local_notes.append(f"local base path {r['base_unresolved']!r} -> {c}")
        else:
            left.append(r["base_unresolved"])
    r["notes"] += local_notes + ([f"unresolved local paths: {', '.join(repr(x) for x in left[:4])}"] if left else [])
    ing = [i for i, _ in r["ingredients"]]
    wsum = sum(w for _, w in r["ingredients"])
    counted = 1.0 if r["normalize"] else wsum
    row.update({"method": r["method"], "kind": r["kind"] or "none", "declared_base": r["base"] or "",
                "ingredients": " ".join(ing), "n_ingredients": len(ing), "weights_sum": round(wsum, 4),
                "normalize": r["normalize"], "format": r["format"],
                "notes": "; ".join(sorted(set(r["notes"] + ([r["base_note"]] if r["base_note"] else []))))})
    if len(ing) >= 2:
        loops = lineage_loops(ing, ancestry)
        row["loops_checked"] = loops is not None
        row["n_loops"] = len(loops) if loops else 0
        row["lineage_loops"] = "; ".join(f"{a} > {d}" for a, d in loops) if loops else ""
    if r["kind"] is None:
        row["verdict"] = "NOT_TASK_VECTOR"
        return row
    if left:
        row["verdict"] = "LOCAL_PATHS"           # some ingredients or the base are unresolvable local folders
        return row
    if len(ing) < 2:
        row["verdict"] = "SINGLE_INGREDIENT"
        return row
    if not r["base"]:
        row["verdict"] = "UNKNOWN"
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no usable base_model"
        return row
    ancs = [ancestry.ancestors(i) for i in ing]
    base_anc = ancestry.ancestors(r["base"])
    if any(a is None for a in ancs) or base_anc is None:
        missing = [i for i, a in zip(ing, ancs) if a is None] + ([r["base"]] if base_anc is None else [])
        row["verdict"] = "UNKNOWN"
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no metadata: " + " ".join(missing)
        return row
    cca, cca_depth = closest_common_ancestor(ancs)
    row["closest_common_ancestor"] = cca or ""
    row["cca_distance"] = cca_depth
    base = r["base"]
    if not all(base in a for a in ancs):
        # a re-upload of the base under another account (same model name, listed as its parent)
        name = base.split("/", 1)[1]
        mirrors = [a for a in sorted(base_anc, key=lambda a: base_anc[a]) if a != base
                   and a.split("/", 1)[-1] == name and all(a in an for an in ancs)]
        if mirrors:
            base = mirrors[0]
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"declared base is a re-upload of {base}"
    base_in_all = all(base in a for a in ancs)
    row["base_distance"] = max(a[base] for a in ancs) if base_in_all else ""
    inf = sorted(n for a, n in ancestry.inferred.items() if any(a in an for an in ancs) or a == r["base"])
    if inf:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "; ".join(inf[:4])
    via = sorted(a for a in ancestry.from_recipe if any(a in an for an in ancs))
    if via:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "parents read from recipe: " + " ".join(via[:5])
    no_parents = [i for i, a in zip(ing, ancs) if len(a) == 1]
    if no_parents and (cca is None or not base_in_all):
        # an ingredient whose card lists no parent: a true root or missing metadata; cannot judge
        row["verdict"] = "INCOMPLETE_LINEAGE"
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no parent listed: " + " ".join(no_parents)
    elif cca is None:
        row["verdict"] = "NO_COMMON_ANCESTOR"
    elif cca == base:
        row["verdict"] = "OK"
    elif base_in_all:
        row["verdict"] = "DOUBLE_COUNT"
        row["risk"] = "high" if (r["kind"] == "sum" and counted > 1.05) else "lower"
        row["shared_update_counted"] = round(counted, 3)
    else:
        row["verdict"] = "BASE_NOT_ANCESTOR"
    return row


# ------------------------------------------------------------------ Hub client with a resumable cache

class HubClient:
    def __init__(self, cache_path, sleep=0.0):
        import huggingface_hub as hh
        self.hh = hh
        self.api = hh.HfApi()
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {"meta": {}, "recipe": {}}
        self.sleep, self.calls = sleep, 0

    def save(self):
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_path)

    def _retry(self, fn):
        for attempt in range(6):
            try:
                self.calls += 1
                if self.sleep:
                    time.sleep(self.sleep)
                return fn()
            except Exception as e:                       # noqa: BLE001 - Hub errors vary by version
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429 or "429" in str(e)[:200]:
                    wait = 30 * (attempt + 1)
                    print(f"  rate limited; waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("rate limited too many times")

    def list_merges(self, limit, sort):
        try:
            it = self.api.list_models(filter="mergekit", sort=sort, limit=limit)
        except TypeError:
            it = self.api.list_models(filter="mergekit", sort=sort, direction=-1, limit=limit)
        return [m.id for m in it]

    def meta(self, repo):
        if repo in self.cache["meta"]:
            return self.cache["meta"][repo]
        try:
            info = self._retry(lambda: self.api.model_info(repo))
        except Exception:                                # gated, deleted, private, renamed
            self.cache["meta"][repo] = None
            return None
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        get = (lambda k: getattr(card, k, None) if not isinstance(card, dict) else card.get(k)) if card else (lambda k: None)
        st = getattr(info, "safetensors", None)
        params = getattr(st, "total", None) if st is not None and not isinstance(st, dict) else (st or {}).get("total")
        created = getattr(info, "created_at", None)
        m = {"tags": list(getattr(info, "tags", None) or []), "base_model": get("base_model"),
             "base_model_relation": get("base_model_relation"), "downloads": getattr(info, "downloads", None),
             "likes": getattr(info, "likes", None), "params": params,
             "created_at": created.isoformat() if hasattr(created, "isoformat") else created}
        self.cache["meta"][repo] = m
        return m

    def recipe_text(self, repo):
        if repo in self.cache["recipe"]:
            return self.cache["recipe"][repo]
        text = None
        for fname in ("mergekit_config.yml", "mergekit_config.yaml", "merge.yml", "merge.yaml",
                      "config.yaml", "README.md"):
            try:
                path = self._retry(lambda: self.hh.hf_hub_download(repo, fname))
            except Exception:
                continue
            body = Path(path).read_text(errors="replace")
            if fname == "README.md":
                text = recipe_from_readme(body)
            else:
                text = body if parse_recipe(body) is not None else None   # e.g. config.yaml may be unrelated
            if text:
                break
        self.cache["recipe"][repo] = text
        return text


QUANT_NAME = re.compile(r"gguf|exl2|exl3|awq|gptq|mlx|bnb|imatrix|\d(?:\.\d+)?bpw|\d+-?bit|int[48]|fp8|-quant", re.I)


def resolve_source(repo, client, hops=4):
    """Follow quantized copies (GGUF, EXL2, AWQ, ...) back to the model they were made from.
    Returns (source repo, chain of copies)."""
    chain, cur = [], repo
    for _ in range(hops):
        rel, parents = parents_from_meta(client.meta(cur))
        if len(parents) == 1 and (rel == "quantized" or QUANT_NAME.search(cur)):
            chain.append(cur)
            cur = parents[0]
        else:
            break
    return cur, chain


def plan_sources(listed, client):
    """Unique merges to audit, in listing order, with the listed copies that point to each."""
    order, copies = [], {}
    for repo in listed:
        src, _ = resolve_source(repo, client)
        if src not in copies:
            copies[src] = []
            order.append(src)
        copies[src].append(repo)
    return order, copies


# ------------------------------------------------------------------ outputs

FIELDS = ["repo", "verdict", "risk", "method", "kind", "declared_base", "closest_common_ancestor",
          "cca_distance", "base_distance", "n_ingredients", "ingredients", "weights_sum", "normalize",
          "shared_update_counted", "loops_checked", "n_loops", "lineage_loops",
          "format", "downloads", "likes", "params", "created_at", "listed_copies", "listed_as", "notes"]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def summarise(rows, out, sort, limit):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    v = Counter(r["verdict"] for r in rows)
    judged = [r for r in rows if r["verdict"] in ("OK", "DOUBLE_COUNT", "BASE_NOT_ANCESTOR", "NO_COMMON_ANCESTOR")]
    dc = [r for r in rows if r["verdict"] == "DOUBLE_COUNT"]
    high = [r for r in dc if r.get("risk") == "high"]
    tv = [r for r in rows if r.get("kind") not in (None, "none") and r["verdict"] not in ("NO_CONFIG", "UNPARSEABLE")]
    lines = [
        "# Audit of public mergekit merges on Hugging Face", "",
        f"Sample: {len(rows)} distinct merges from the top {limit} models tagged `mergekit` by {sort} "
        "(quantized copies such as GGUF followed back to the original merge). "
        "Ancestry from model-card metadata (self-reported); see notes in audit.csv.", "",
        "| Verdict | Count |", "|---|---|"]
    lines += [f"| {k} | {n} |" for k, n in v.most_common()]
    lines += ["", "## Headline",
              f"- Merges using a task-vector method (TIES, task arithmetic, DARE, ...): {len(tv)}",
              f"- Of these, with 2+ ingredients and complete metadata (judged): {len(judged)}",
              f"- **DOUBLE_COUNT: {len(dc)} of {len(judged)} ({pct(len(dc), len(judged))})**, of which "
              f"high risk (summing method, weights > 1): {len(high)}",
              f"- OK: {v.get('OK', 0)} ({pct(v.get('OK', 0), len(judged))}); base not an ancestor: "
              f"{v.get('BASE_NOT_ANCESTOR', 0)}; no common ancestor: {v.get('NO_COMMON_ANCESTOR', 0)}",
              f"- Not judged: incomplete lineage {v.get('INCOMPLETE_LINEAGE', 0)}, missing metadata {v.get('UNKNOWN', 0)}, "
              f"unresolvable local paths {v.get('LOCAL_PATHS', 0)}; "
              f"no published recipe {v.get('NO_CONFIG', 0)}; quantized copies followed back: "
              f"{sum(int(r.get('listed_copies') or 0) for r in rows)}"]
    checked = [r for r in rows if r.get("loops_checked")]
    looped = [r for r in checked if r.get("n_loops")]
    tv_checked = [r for r in checked if r.get("kind") not in (None, "none")]
    tv_looped = [r for r in tv_checked if r.get("n_loops")]
    lines += ["", "## Ancestor + descendant in the same merge (lineage loops)",
              "One ingredient is a (partial) ancestor of another, so its contribution enters the merge twice; "
              "choosing the right base does not remove this.",
              f"- Merges with 2+ ingredients and complete ingredient metadata: {len(checked)}",
              f"- **With a lineage loop: {len(looped)} ({pct(len(looped), len(checked))})**; "
              f"among task-vector methods: {len(tv_looped)} of {len(tv_checked)} ({pct(len(tv_looped), len(tv_checked))})",
              f"- Both mistakes (DOUBLE_COUNT base and a loop): {sum(1 for r in dc if r.get('n_loops'))}"]
    top_loops = sorted(looped, key=lambda r: -(r.get("downloads") or 0))[:10]
    if top_loops:
        lines += ["", "| Merge | Method | Ancestor > descendant | Downloads |", "|---|---|---|---|"]
        lines += [f"| {r['repo']} | {r.get('method', '')} | {r['lineage_loops']} | {r.get('downloads') or ''} |" for r in top_loops]
    by_method = Counter(r["method"] for r in dc)
    if by_method:
        lines += ["", "DOUBLE_COUNT by method: " + ", ".join(f"{m} {n}" for m, n in by_method.most_common())]
    pair = Counter((r["declared_base"], r["closest_common_ancestor"]) for r in dc)
    if pair:
        lines += ["", "## Most common double-counting patterns (declared base -> closer shared ancestor)", "",
                  "| Declared base | Closer shared ancestor | Merges |", "|---|---|---|"]
        lines += [f"| {b} | {c} | {n} |" for (b, c), n in pair.most_common(15)]
    top = sorted(dc, key=lambda r: -(r.get("downloads") or 0))[:15]
    if top:
        lines += ["", "## Most-downloaded DOUBLE_COUNT merges", "",
                  "| Merge | Method | Declared base | Closer ancestor | Weights sum | Downloads |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['repo']} | {r['method']} | {r['declared_base']} | {r['closest_common_ancestor']} | "
                  f"{r['weights_sum']} | {r.get('downloads') or ''} |" for r in top]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    small = [r for r in dc if r.get("params") and r["params"] <= 3.5e9]
    small.sort(key=lambda r: (r.get("risk") != "high", r["params"], -(r.get("downloads") or 0)))
    with (out / "verify_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(small[:30])
    print("\n".join(lines))
    return v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--sort", default="downloads", help="downloads, likes, created_at, trending_score")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--out", default="audit")
    p.add_argument("--sleep", type=float, default=0.0, help="Pause between Hub calls (seconds)")
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = HubClient(out / "cache.json", sleep=args.sleep)
    order_file = out / "merges.json"
    if order_file.exists():
        merges = json.loads(order_file.read_text())
    else:
        print(f"Listing up to {args.limit} mergekit models by {args.sort} ...", flush=True)
        merges = [m.lower() for m in client.list_merges(args.limit, args.sort)]
        order_file.write_text(json.dumps(merges))
    print(f"{len(merges)} merged models", flush=True)
    ancestry = Ancestry(client, depth=args.depth)
    print("Following quantized copies back to the original merges ...", flush=True)
    sources, copies = plan_sources(merges, client)
    client.save()
    print(f"{len(merges)} listed models -> {len(sources)} distinct merges to audit", flush=True)
    rows, t0 = [], time.time()
    for k, repo in enumerate(sources, 1):
        if args.summarise_only and repo not in client.cache["recipe"]:
            continue
        try:
            row = audit_one(repo, client, ancestry)
        except Exception as e:                           # never lose a long run to one bad repo
            row = {"repo": repo, "verdict": "ERROR", "notes": str(e)[:200]}
        row["listed_copies"] = len([c for c in copies[repo] if c != repo])
        row["listed_as"] = " ".join(copies[repo][:3])
        rows.append(row)
        if k % 25 == 0:
            client.save()
            print(f"  {k}/{len(sources)}  {time.time() - t0:.0f}s  Hub calls {client.calls}", flush=True)
    client.save()
    summarise(rows, out, args.sort, args.limit)


if __name__ == "__main__":
    sys.exit(main())
