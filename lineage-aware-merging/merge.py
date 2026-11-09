"""
Merge fine-tuned models into a reference model with Task Arithmetic, TIES or DARE.

    python merge.py --reference Qwen/Qwen2.5-1.5B-Instruct \
        --models Thomas-Chou/Qwen2.5-1.5B-Open-R1-GRPO jahyungu/Qwen2.5-1.5B-Instruct_mbpp \
        --method ta --out merged/ta_r1math_mbpp

Task vectors are tau_k = W_k - W_reference for EVERY tensor (embeddings, norms and all
attention/MLP matrices). The --reference is what "re-basing" changes: the same pair can be
merged against the released base or against Instruct, and nothing else differs.

Methods (defaults follow the original papers / common mergekit settings):
  ta    W = W_ref + lam * sum_k tau_k                     (--lam, default 1/K = plain average)
  ties  trim each tau_k to its top --density entries, elect signs, disjoint mean, x --lam
        (default density 0.2, lam 1.0)
  dare  drop entries with probability --drop, rescale by 1/(1-drop), then ta (--seed)
        (default drop 0.9, lam 1/K)
  iso_c for every attention/MLP matrix: SVD of sum_k tau_k, all singular values set to
        their mean, W = W_ref + --alpha * result (Marczak et al., ICML 2025; default
        alpha 1.0 - the paper tunes alpha on held-out data). Embeddings, norms and other
        tensors use ta with lam 1/K.

The output folder holds model.safetensors (bfloat16), the config and tokenizer of
--tokenizer-from (default: the reference), and merge_info.json with every setting, so it
can be evaluated directly:  python eval_model.py --model merged/ta_r1math_mbpp
Merging runs on the CPU; on Kaggle turn the GPU off for it so it does not use GPU hours.
"""

import argparse
import json
import sys
import time

import numpy as np

import mergecore as mc

DEFAULT_LAM = {"ta": None, "ties": 1.0, "dare": None, "iso_c": None}   # None -> 1/K


def check_compatible(ref, models):
    """Every reference tensor must exist with the same shape in every model."""
    problems = []
    for m in models:
        mkeys = set(m.keys())
        for k in ref.keys():
            if k not in mkeys:
                problems.append(f"{m.id}: missing {k}")
            elif m.shape(k) != ref.shape(k):
                problems.append(f"{m.id}: {k} has shape {m.shape(k)}, reference has {ref.shape(k)}")
    return problems


def run_merge(ref, models, method, *, lam=None, density=0.2, drop=0.9, seed=0, alpha=1.0,
              to_storage=lambda a: a, log_every=50):
    """Merge tensor by tensor. Returns {key: stored tensor}."""
    K = len(models)
    lam = (1.0 / K) if lam is None else lam
    rng = np.random.default_rng(seed)
    out = {}
    keys = ref.keys()
    t0 = time.time()
    for i, key in enumerate(keys, 1):
        w_ref = ref.get(key)
        merged = mc.merge_tensor(method, w_ref, [m.get(key) for m in models],
                                 lam=lam, density=density, drop=drop, rng=rng, alpha=alpha,
                                 is_matrix=mc.cell_of(key) is not None)
        out[key] = to_storage(merged.astype(np.float32))
        if log_every and (i % log_every == 0 or i == len(keys)):
            print(f"  merged {i}/{len(keys)} tensors  {time.time() - t0:.0f}s", flush=True)
    return out, lam


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reference", required=True, help="Model the task vectors are measured from")
    p.add_argument("--models", nargs="+", required=True, help="Fine-tuned models to merge (2 or more; 1 with ta + --lam)")
    p.add_argument("--method", choices=["ta", "ties", "dare", "iso_c"], required=True)
    p.add_argument("--alpha", type=float, default=1.0, help="Iso-C: scaling of the isotropic update")
    p.add_argument("--lam", type=float, default=None, help="Scaling (default: 1/K for ta/dare, 1.0 for ties)")
    p.add_argument("--density", type=float, default=0.2, help="TIES: fraction of entries kept")
    p.add_argument("--drop", type=float, default=0.9, help="DARE: drop probability")
    p.add_argument("--seed", type=int, default=0, help="DARE: random seed")
    p.add_argument("--tokenizer-from", default=None, help="Copy config/tokenizer from this model (default: reference)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if len(args.models) == 1 and (args.lam is None or args.method != "ta"):
        sys.exit("one model is only allowed with --method ta and an explicit --lam "
                 "(W = W_ref + lam * tau: a step of size lam from the reference towards the model)")

    import hf_io
    print(f"Opening reference {args.reference}", flush=True)
    ref = hf_io.open_model(args.reference)
    models = []
    for mid in args.models:
        print(f"Opening {mid}", flush=True)
        models.append(hf_io.open_model(mid))
    problems = check_compatible(ref, models)
    if problems:
        sys.exit("Models are not compatible with the reference:\n  " + "\n  ".join(problems[:20]))

    lam = args.lam if args.lam is not None else DEFAULT_LAM[args.method]
    print(f"Merging {len(models)} models with {args.method}", flush=True)
    tensors, lam_used = run_merge(ref, models, args.method, lam=lam, density=args.density,
                                  drop=args.drop, seed=args.seed, alpha=args.alpha,
                                  to_storage=hf_io.to_storage)
    side = hf_io.local_dir(args.tokenizer_from) if args.tokenizer_from else ref.dir
    info = {"reference": args.reference, "models": args.models, "method": args.method,
            "lam": lam_used, "density": args.density if args.method == "ties" else None,
            "drop": args.drop if args.method == "dare" else None,
            "seed": args.seed if args.method == "dare" else None,
            "alpha": args.alpha if args.method == "iso_c" else None,
            "tokenizer_from": args.tokenizer_from or args.reference,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    hf_io.save_model(tensors, args.out, side, metadata=info)
    print(f"Saved -> {args.out}\n{json.dumps(info, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
