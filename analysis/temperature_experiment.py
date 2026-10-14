"""Temperature-conditioning experiment: does the recorded air temperature help?

Reads the runs of experiments/orthotemp_temperature.txt, named
<model>_<data>_<condition>_s<seed>, and reports per backbone:

  * test macro-F1 and accuracy for each condition - none (no temperature),
    real (FiLM on the recorded temperature), shuffled (FiLM on temperatures
    permuted across training recordings) - as mean +- sd over seeds, also on
    the recordist-unseen test files
  * paired tests on the same test files, using the seed-averaged
    probabilities of each condition: real vs none and real vs shuffled
    (paired bootstrap of macro-F1 and McNemar's test)

Only "real" beating "shuffled" is evidence that the temperature values carry
species information. "shuffled" has the same extra parameters and inputs, so
a gain of "real" over "none" alone could come from the conditioning head.
And because a handful of recordists supply most temperature labels, a gain
that vanishes on the recordist-unseen files may be recordist identity
rather than physiology (analysis/temperature_audit.py).

Usage:
    python analysis/temperature_experiment.py --data orthotemp
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.eval import stats  # noqa: E402

CONDITIONS = ("none", "real", "shuffled")


def find_runs(data):
    """{(model, condition): [run names]} for result files <model>_<data>_<condition>_s<seed>.json."""
    pattern = re.compile(rf"^([a-z]+)_{re.escape(data)}_({'|'.join(CONDITIONS)})_s\d+$")
    runs = {}
    for path in sorted((C.RESULTS_DIR / "metrics").glob(f"*_{data}_*_s*.json")):
        m = pattern.match(path.stem)
        if m:
            runs.setdefault((m.group(1), m.group(2)), []).append(path.stem)
    return runs


def seed_scores(run):
    r = json.loads((C.RESULTS_DIR / "metrics" / f"{run}.json").read_text())
    unseen = r.get("test_subsets", {}).get("recordist_unseen", {})
    return {"macro_f1": r["test"]["file"]["macro_f1"], "accuracy": r["test"]["file"]["accuracy"],
            "unseen_macro_f1": unseen.get("macro_f1"), "n_unseen": unseen.get("n_files")}


def mean_probs(runs):
    """Seed-averaged test probabilities; every run must cover the same files in the same order."""
    preds = [stats.load_predictions(run) for run in runs]
    for p in preds[1:]:
        assert np.array_equal(p["files"], preds[0]["files"]), "runs scored different test files"
    return preds[0]["labels"], np.mean([p["probs"] for p in preds], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="orthotemp")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    runs = find_runs(args.data)
    if not runs:
        sys.exit(f"no runs named <model>_{args.data}_<none|real|shuffled>_s<seed> in results/metrics")

    rows, tests = [], {}
    for model in sorted({m for m, _ in runs}):
        for cond in CONDITIONS:
            if (model, cond) not in runs:
                continue
            agg = stats.aggregate_seeds([seed_scores(r) for r in runs[model, cond]])
            rows.append({"model": model, "condition": cond, "seeds": agg["macro_f1"]["n"],
                         "test macro-F1 mean": agg["macro_f1"]["mean"], "test macro-F1 sd": agg["macro_f1"]["sd"],
                         "test acc mean": agg["accuracy"]["mean"], "test acc sd": agg["accuracy"]["sd"],
                         "recordist-unseen macro-F1 mean": agg["unseen_macro_f1"]["mean"],
                         "recordist-unseen files": agg["n_unseen"]["mean"]})
        probs = {c: mean_probs(runs[model, c]) for c in CONDITIONS if (model, c) in runs}
        for other in ("none", "shuffled"):
            if "real" in probs and other in probs:
                labels, p_real = probs["real"]
                _, p_other = probs[other]
                n_classes = p_real.shape[1]
                tests[f"{model}: real vs {other}"] = {
                    "paired_bootstrap": stats.paired_bootstrap(labels, p_real, p_other, n_classes, n_boot=args.n_boot),
                    "mcnemar": stats.mcnemar(labels, p_real.argmax(1), p_other.argmax(1)),
                    "seeds": [len(runs[model, "real"]), len(runs[model, other])]}

    table = pd.DataFrame(rows)
    out = C.RESULTS_DIR / "metrics" / f"temperature_{args.data}"
    table.round(4).to_csv(f"{out}.csv", index=False)
    with open(f"{out}.json", "w") as fh:
        json.dump(tests, fh, indent=2, default=float)
    print(table.round(3).to_markdown(index=False))
    for name, t in tests.items():
        pb, mc = t["paired_bootstrap"], t["mcnemar"]
        print(f"{name}: delta macro-F1 {pb['delta_macro_f1']:+.3f} "
              f"[{pb['ci'][0]:+.3f}, {pb['ci'][1]:+.3f}], bootstrap p {pb['p_value']:.3f}, McNemar p {mc['p_value']:.3f}")

    models = list(dict.fromkeys(table.model))
    fig, ax = plt.subplots(figsize=(max(4.5, 2.2 + 1.6 * len(models)), 3.2))
    width = 0.8 / len(CONDITIONS)
    for k, cond in enumerate(CONDITIONS):
        part = table[table.condition == cond].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + (k - 1) * width, part["test macro-F1 mean"], width,
               yerr=part["test macro-F1 sd"].fillna(0), capsize=3, label=cond)
    ax.set_xticks(range(len(models)), [m.upper() for m in models])
    ax.set_ylabel("test macro-F1 (file level)")
    ax.set_title(f"Temperature conditioning ({args.data}), mean ± sd over seeds", fontsize=9)
    ax.legend(title="temperature", fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "figures" / f"temperature_{args.data}.png", dpi=200)


if __name__ == "__main__":
    main()
