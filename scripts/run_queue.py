"""Run a plan of training runs one after another, skipping the finished ones.

A plan file holds one line of src.train arguments per run ('#' starts a
comment). A run counts as finished when results/metrics/<run>.json exists,
so the queue can be stopped and restarted at any point without repeating
work - useful on a laptop that sleeps, and on Colab sessions that time out.
Runs that fail are tried once more at the end of the pass.
Each run's console output goes to logs/<run>.log.

Usage:
    caffeinate -i python scripts/run_queue.py experiments/species40_main.txt
    python scripts/run_queue.py experiments/species40_main.txt --dry-run
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_name(argv):
    opts = {}
    for i, tok in enumerate(argv[:-1]):
        if tok in ("--model", "--data", "--tag"):
            opts[tok] = argv[i + 1]
    return f"{opts['--model']}_{Path(opts.get('--data', 'pilot')).name}{opts.get('--tag', '')}"


def read_plan(path):
    runs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            argv = shlex.split(line)
            runs.append((run_name(argv), argv))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = [r for plan in args.plans for r in read_plan(plan)]
    done = lambda name: (ROOT / "results/metrics" / f"{name}.json").exists()
    todo = [(n, a) for n, a in runs if not done(n)]
    print(f"{len(runs)} runs in plan, {len(runs) - len(todo)} done, {len(todo)} to run", flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    failed = run_all(todo, done, args.dry_run)
    if failed:
        # one more attempt: a run killed from outside (memory pressure, a closed session) usually succeeds
        print(f"retrying {len(failed)} failed run(s) once", flush=True)
        failed = run_all(failed, done, args.dry_run)
    if failed:
        print("still failing:", ", ".join(name for name, _ in failed), flush=True)


def run_all(todo, done, dry_run):
    failed = []
    for k, (name, argv) in enumerate(todo, 1):
        if done(name):     # finished meanwhile, e.g. by another queue running CPU-only runs
            print(f"[{k}/{len(todo)}] {name}: already done, skipped", flush=True)
            continue
        print(f"[{k}/{len(todo)}] {name}: {' '.join(argv)}", flush=True)
        if dry_run:
            continue
        t0 = time.time()
        with open(ROOT / "logs" / f"{name}.log", "w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", "src.train", *argv],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 else f"FAILED (exit {code}, see logs/{name}.log)"
        print(f"    {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if code != 0:
            failed.append((name, argv))
    return failed

if __name__ == "__main__":
    main()
