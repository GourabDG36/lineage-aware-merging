"""
Run a whole experiment plan with one command, safely and resumably.

    python run_plan.py plans/phase1.json            # run everything not yet done
    python run_plan.py plans/phase1.json --dry-run  # only print what would run
    python run_plan.py plans/phase1.json --status   # table of done / to-do / failed

The plan (JSON) lists models by short name and a sequence of steps:
  {"type": "eval",    "model": "qwq"}
  {"type": "predict", "reference": "instruct", "models": ["qwq", "mbpp"]}
  {"type": "merge",   "reference": "instruct", "models": ["qwq", "mbpp"], "method": "ties",
                      "lam": 1.0, "density": 0.2, "drop": 0.9, "alpha": 1.0,
                      "tokenizer_from": "instruct", "keep": false}
A merge step merges, evaluates the merged model, then deletes the merged weights
(unless "keep": true), so disk use stays small.

Safety:
  * finished steps are detected from their output files and skipped, so after a crash or a
    Kaggle restart the same command continues where it stopped;
  * before the first evaluation of a downloaded model, a sanity check scores it on 8 GSM8K
    questions; a model scoring 0/8 is marked BROKEN and every step using it is skipped
    (this is what caught the damaged r1_math checkpoint);
  * a failed step is logged and the plan continues with the next step;
  * every step's output goes to logs/<step>.log, and backup.zip (results, predictors,
    logs, status) is refreshed after every step - download it from the Output panel.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PEEK_ITEMS = 8


def fmt_num(x):
    return f"{x:g}"


def step_name(step, aliases):
    t = step["type"]
    if t == "eval":
        return step.get("name", step["model"])
    pair = "_".join(step["models"])
    if t == "predict":
        return step.get("name", f"{pair}__ref-{step['reference']}")
    if t == "merge":
        tag = step["method"]
        for key, short in (("lam", "lam"), ("density", "d"), ("drop", "p"), ("alpha", "a")):
            if key in step:
                tag += f"-{short}{fmt_num(step[key])}"
        return step.get("name", f"{tag}__{pair}__ref-{step['reference']}")
    raise ValueError(f"unknown step type {t!r}")


def eval_done(name, limit, tasks, results_dir):
    path = Path(results_dir) / name / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    return all(summary.get(t, {}).get("n") == (limit or summary.get(t, {}).get("n")) and
               summary.get("limit") == limit for t in tasks)


def models_used(step):
    """Every model alias a step mentions (used to validate the plan)."""
    used = list(step.get("models", []))
    for key in ("model", "reference", "tokenizer_from"):
        if key in step:
            used.append(step[key])
    return used


def models_tested(step):
    """Models whose behaviour a step depends on: the evaluated or merged models. A reference
    or tokenizer source is never asked questions, so it is not subject to the sanity check."""
    return list(step.get("models", [])) + ([step["model"]] if "model" in step else [])


class Runner:
    def __init__(self, plan_path, dry_run=False):
        self.plan_path = Path(plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        self.aliases = self.plan["models"]
        self.limit = self.plan.get("limit", 200)
        self.tasks = self.plan.get("tasks", ["gsm8k", "mbpp"])
        self.results = Path(self.plan.get("results_dir", "results"))
        self.predictors = Path(self.plan.get("predictors_dir", "predictors"))
        self.merged = Path(self.plan.get("merged_dir", "merged"))
        self.logs = Path("logs")
        self.status_path = Path("status.json")
        self.dry_run = dry_run
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        for step in self.plan["steps"]:
            for m in models_used(step):
                if m not in self.aliases:
                    raise SystemExit(f"step {step} uses unknown model {m!r}; add it to 'models'")

    # ---------------------------------------------------------------- helpers

    def repo(self, alias):
        return self.aliases[alias]

    def run(self, name, args):
        """Run a script in a fresh Python process; stream output to the console and a log."""
        cmd = [sys.executable, str(HERE / args[0])] + [str(a) for a in args[1:]]
        print(f"\n$ {' '.join(cmd)}", flush=True)
        if self.dry_run:
            return 0
        self.logs.mkdir(exist_ok=True)
        with (self.logs / f"{name}.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return proc.wait()

    def mark(self, name, state, note=""):
        self.status[name] = {"state": state, "note": note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if not self.dry_run:
            self.status_path.write_text(json.dumps(self.status, indent=2))

    def backup(self):
        if self.dry_run:
            return
        with zipfile.ZipFile("backup.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for folder in (self.results, self.predictors, self.logs):
                if folder.exists():
                    for f in folder.rglob("*"):
                        if f.is_file() and "_subspace_cache" not in f.parts:
                            z.write(f)
            if self.status_path.exists():
                z.write(self.status_path)

    def broken(self, aliases):
        return [a for a in aliases if self.status.get(f"peek:{a}", {}).get("state") == "BROKEN"]

    # ---------------------------------------------------------------- sanity check

    def peek(self, alias):
        key = f"peek:{alias}"
        if self.status.get(key, {}).get("state") in ("OK", "BROKEN"):
            return self.status[key]["state"] == "OK"
        name = f"{alias}__peek"
        rc = self.run(name, ["eval_model.py", "--model", self.repo(alias), "--name", name,
                             "--limit", PEEK_ITEMS, "--tasks", "gsm8k", "--results-dir", self.results])
        if self.dry_run:
            return True
        summary_path = self.results / name / "summary.json"
        if rc != 0 or not summary_path.exists():
            self.mark(key, "FAILED", "sanity check did not run; see log")
            return False
        acc = json.loads(summary_path.read_text())["gsm8k"]["accuracy"]
        if acc == 0:
            self.mark(key, "BROKEN", f"0/{PEEK_ITEMS} on GSM8K - read {name}/gsm8k.jsonl")
            print(f"!! {alias} scored 0/{PEEK_ITEMS}: marked BROKEN, its steps will be skipped")
            return False
        self.mark(key, "OK", f"{acc:.3f} on {PEEK_ITEMS} items")
        return True

    # ---------------------------------------------------------------- steps

    def eval_args(self, model, name):
        return ["eval_model.py", "--model", model, "--name", name, "--limit", self.limit,
                "--tasks", *self.tasks, "--results-dir", self.results]

    def do_eval(self, step, name):
        if not self.peek(step["model"]):
            return "SKIPPED", "model failed the sanity check"
        rc = self.run(name, self.eval_args(self.repo(step["model"]), name))
        return ("DONE", "") if rc == 0 else ("FAILED", f"exit code {rc}")

    def do_predict(self, step, name):
        a, b = step["models"]
        rc = self.run(name, ["predictors.py", "--reference", self.repo(step["reference"]),
                             "--models", self.repo(a), self.repo(b), "--name", name,
                             "--k", step.get("k", 64), "--out-dir", self.predictors])
        return ("DONE", "") if rc == 0 else ("FAILED", f"exit code {rc}")

    def do_merge(self, step, name):
        out = self.merged / name
        if not (out / "model.safetensors").exists():
            args = ["merge.py", "--reference", self.repo(step["reference"]),
                    "--models", *[self.repo(m) for m in step["models"]],
                    "--method", step["method"], "--out", out]
            for key, flag in (("lam", "--lam"), ("density", "--density"), ("drop", "--drop"),
                              ("seed", "--seed"), ("alpha", "--alpha")):
                if key in step:
                    args += [flag, step[key]]
            if "tokenizer_from" in step:
                args += ["--tokenizer-from", self.repo(step["tokenizer_from"])]
            rc = self.run(name, args)
            if rc != 0:
                return "FAILED", f"merge exit code {rc}"
        rc = self.run(name, self.eval_args(out, name))
        if rc != 0:
            return "FAILED", f"evaluation exit code {rc} (merged model kept in {out})"
        if not step.get("keep") and not self.dry_run:
            shutil.rmtree(out, ignore_errors=True)
        return "DONE", ""

    def is_done(self, step, name):
        if step["type"] == "predict":
            return (self.predictors / name / "summary.json").exists()
        return eval_done(name, self.limit, self.tasks, self.results)

    def run_all(self, only=None):
        steps = self.plan["steps"]
        for i, step in enumerate(steps, 1):
            name = step_name(step, self.aliases)
            if only and only not in name:
                continue
            header = f"[{i}/{len(steps)}] {step['type']} {name}"
            if self.is_done(step, name):
                print(f"{header}: already done, skipping")
                if self.status.get(name, {}).get("state") != "DONE":
                    self.mark(name, "DONE", "found existing outputs")
                continue
            bad = self.broken(models_tested(step))
            if bad:
                print(f"{header}: skipped, uses broken model(s) {bad}")
                self.mark(name, "SKIPPED", f"uses broken model(s) {bad}")
                continue
            print(f"\n{'=' * 70}\n{header}\n{'=' * 70}", flush=True)
            t0 = time.time()
            state, note = getattr(self, f"do_{step['type']}")(step, name)
            note = f"{note} ({time.time() - t0:.0f}s)".strip()
            self.mark(name, state, note)
            self.backup()
            print(f"{header}: {state} {note}", flush=True)
        self.print_status()

    def print_status(self):
        print(f"\n{'step':60s} state")
        for step in self.plan["steps"]:
            name = step_name(step, self.aliases)
            state = "DONE" if self.is_done(step, name) else self.status.get(name, {}).get("state", "to do")
            note = self.status.get(name, {}).get("note", "")
            print(f"{name:60s} {state:8s} {note}")
        peeks = {k: v for k, v in self.status.items() if k.startswith("peek:")}
        for k, v in peeks.items():
            print(f"{k:60s} {v['state']:8s} {v['note']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("plan")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    p.add_argument("--status", action="store_true", help="Only print the status table")
    p.add_argument("--only", default=None, help="Run only steps whose name contains this text")
    args = p.parse_args()
    runner = Runner(args.plan, dry_run=args.dry_run)
    if args.status:
        runner.print_status()
    else:
        runner.run_all(only=args.only)


if __name__ == "__main__":
    main()
