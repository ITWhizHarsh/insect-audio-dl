"""Run a plan of training runs one after another, skipping the finished ones.

A plan file holds one line of src.train arguments per run ('#' starts a
comment). A run counts as finished when results/metrics/<run>.json exists,
so the queue can be stopped and restarted at any point without repeating
work - useful on a laptop that sleeps, and on Colab sessions that time out.
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
    todo = [(n, a) for n, a in runs if not (ROOT / "results/metrics" / f"{n}.json").exists()]
    print(f"{len(runs)} runs in plan, {len(runs) - len(todo)} done, {len(todo)} to run", flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    for k, (name, argv) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {name}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(ROOT / "logs" / f"{name}.log", "w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", "src.train", *argv],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 else f"FAILED (exit {code}, see logs/{name}.log)"
        print(f"    {status} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
