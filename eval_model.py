"""
Evaluate one causal language model on GSM8K (maths) and MBPP (Python code).

Works the same on Kaggle (T4, float16) and on a local GPU (bfloat16 when supported).

    python eval_model.py --model Qwen/Qwen2.5-1.5B-Instruct --tasks gsm8k mbpp --limit 200
    python eval_model.py --model merged/ta_r1math_mbpp --name ta_r1math_mbpp --device cuda:1

--model is a Hugging Face repo id or a local folder (e.g. a merged model).
Every answer is written to results/<name>/<task>.jsonl as soon as its batch finishes,
so a run that is interrupted can be restarted with the same command and continues
where it stopped. The scores are written to results/<name>/summary.json.

Scoring
  gsm8k : zero-shot, greedy. The final number in the answer (after "####", inside
          \\boxed{}, or else the last number in the text) must equal the reference.
  mbpp  : zero-shot, greedy, pass@1. The first Python code block in the answer is run
          with the task's three assert tests in a separate process (10 s timeout).
          This runs model-written code: do it on Kaggle or in a throwaway environment.

Both tasks use the first --limit examples of the test split, so every model is scored
on exactly the same questions.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GSM8K_PROMPT = (
    "Solve the following math problem step by step. "
    "At the end, write the final answer as a single number on its own line in the form "
    "'#### <number>'.\n\n{question}"
)
MBPP_PROMPT = (
    "You are an expert Python programmer, and here is your task: {text}\n"
    "Your code should pass these tests:\n\n{tests}\n\n"
    "Write the complete function in a single ```python code block."
)
MAX_NEW_TOKENS = {"gsm8k": 1024, "mbpp": 768}
CODE_TIMEOUT_S = 10

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


# ---------------------------------------------------------------- answer extraction

def _to_number(text):
    if text is None:
        return None
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def gsm8k_reference(answer_field):
    """The reference answer of a GSM8K item is the number after '####'."""
    return _to_number(answer_field.split("####")[-1])


def gsm8k_prediction(generation):
    """Final number of a model answer: '####' line, then \\boxed{}, then last number."""
    hashes = re.findall(r"####\s*(" + _NUMBER + ")", generation)
    if hashes:
        return _to_number(hashes[-1])
    boxed = re.findall(r"\\boxed\{\s*\$?(" + _NUMBER + r")\s*\}", generation)
    if boxed:
        return _to_number(boxed[-1])
    numbers = re.findall(_NUMBER, generation)
    return _to_number(numbers[-1]) if numbers else None


def gsm8k_correct(generation, answer_field):
    pred, ref = gsm8k_prediction(generation), gsm8k_reference(answer_field)
    return pred is not None and ref is not None and abs(pred - ref) < 1e-6


def extract_code(generation):
    """First ```python block; else first ``` block; else the whole answer."""
    for pattern in (r"```python\s*\n(.*?)```", r"```[a-zA-Z]*\s*\n(.*?)```"):
        match = re.search(pattern, generation, re.DOTALL)
        if match:
            return match.group(1)
    unterminated = re.search(r"```(?:python)?\s*\n(.*)", generation, re.DOTALL)
    return unterminated.group(1) if unterminated else generation


def run_mbpp_tests(code, setup_code, tests, timeout=CODE_TIMEOUT_S):
    """True if code + all assert tests run without error in a fresh Python process."""
    program = "\n".join([setup_code or "", code, "", *tests, ""])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=tmp, capture_output=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


# ---------------------------------------------------------------- data

def load_task(task, limit):
    from datasets import load_dataset
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": i, "prompt": GSM8K_PROMPT.format(question=row["question"]),
                  "answer": row["answer"]} for i, row in enumerate(ds)]
    elif task == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
        items = [{"id": row["task_id"],
                  "prompt": MBPP_PROMPT.format(text=row["text"], tests="\n".join(row["test_list"])),
                  "tests": list(row["test_list"]), "setup": row.get("test_setup_code", "")}
                 for row in ds]
    else:
        raise ValueError(f"unknown task {task!r}")
    return items[:limit] if limit else items


def score(task, item, generation):
    if task == "gsm8k":
        return gsm8k_correct(generation, item["answer"])
    return run_mbpp_tests(extract_code(generation), item["setup"], item["tests"])


# ---------------------------------------------------------------- model

def pick_dtype(torch, requested):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    # "auto": native bfloat16 only on compute capability 8.0+ (Ampere and newer).
    # torch.cuda.is_bf16_supported() also returns True on a T4 (7.5), where bfloat16
    # is emulated and several times slower, so it is not used here.
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model(model_id, device, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = pick_dtype(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    return tok, model, dtype


def generate_batch(tok, model, prompts, max_new_tokens):
    import torch
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    # an answer is "truncated" if it used every allowed token (generation stopped by the limit)
    lengths = (new_tokens != tok.pad_token_id).sum(dim=1).tolist()
    texts_out = tok.batch_decode(new_tokens, skip_special_tokens=True)
    return [(t, n, n >= max_new_tokens) for t, n in zip(texts_out, lengths)]


# ---------------------------------------------------------------- driver

def read_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def evaluate_task(task, items, tok, model, out_dir, batch_size, max_new_tokens=None):
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS[task]
    path = out_dir / f"{task}.jsonl"
    done = read_done(path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[{task}] {len(items)} items, {len(done)} already done, {len(todo)} to run", flush=True)
    start = time.time()
    with path.open("a", encoding="utf-8") as fh:
        for b in range(0, len(todo), batch_size):
            batch = todo[b:b + batch_size]
            gens = generate_batch(tok, model, [it["prompt"] for it in batch], max_new_tokens)
            for item, (gen, n_tokens, truncated) in zip(batch, gens):
                row = {"id": item["id"], "correct": bool(score(task, item, gen)),
                       "tokens": n_tokens, "truncated": truncated, "generation": gen}
                done[item["id"]] = row
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            n = b + len(batch)
            scored = [done[i["id"]]["correct"] for i in items if i["id"] in done]
            acc = sum(scored) / len(scored)
            print(f"[{task}] {n}/{len(todo)}  running accuracy {acc:.3f}  "
                  f"{time.time() - start:.0f}s", flush=True)
    rows = [done[it["id"]] for it in items]
    tokens = [r.get("tokens") for r in rows if r.get("tokens") is not None]
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "max_new_tokens": max_new_tokens,
            "truncated": sum(bool(r.get("truncated")) for r in rows),
            "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local folder")
    parser.add_argument("--name", default=None, help="Results folder name (default: from --model)")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "mbpp"], choices=["gsm8k", "mbpp"])
    parser.add_argument("--limit", type=int, default=200, help="First N test items per task (0 = all)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"],
                        help="float16 (default) runs natively on T4 and newer GPUs, so every model "
                             "is scored with the same number format on Kaggle and locally")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override the answer length limit (default: gsm8k 1024, mbpp 768)")
    args = parser.parse_args()

    name = args.name or args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.results_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model, dtype = load_model(args.model, args.device, args.dtype)
    print(f"Model {args.model} on {args.device} in {str(dtype).replace('torch.', '')}", flush=True)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    dtype_name = str(dtype).replace("torch.", "")
    if summary.get("dtype") and summary["dtype"] != dtype_name:
        sys.exit(f"{out_dir} already holds answers generated in {summary['dtype']}, but this run "
                 f"uses {dtype_name}. Use a new --name (or delete that folder) so results are not mixed.")
    summary.update({"model": args.model, "dtype": dtype_name, "limit": args.limit})
    for task in args.tasks:
        wanted = args.max_new_tokens or MAX_NEW_TOKENS[task]
        previous = summary.get(task, {}).get("max_new_tokens")
        if (out_dir / f"{task}.jsonl").exists() and previous != wanted:
            sys.exit(f"{out_dir}/{task}.jsonl was generated with a "
                     f"{previous or 'different (older)'} token limit, this run uses {wanted}. "
                     f"Use a new --name (or delete that folder) so results are not mixed.")
        items = load_task(task, args.limit)
        t0 = time.time()
        result = evaluate_task(task, items, tok, model, out_dir, args.batch_size, args.max_new_tokens)
        result["seconds"] = round(time.time() - t0, 1)
        summary[task] = result
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{task}] accuracy {result['accuracy']:.3f} on {result['n']} items; "
              f"{result['truncated']} hit the {result['max_new_tokens']}-token limit", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
