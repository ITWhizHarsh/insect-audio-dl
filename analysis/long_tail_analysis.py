"""Does per-class accuracy follow the amount of training data? (long-tail analysis)

On a long-tailed dataset a model can reach a decent macro-F1 by doing well
on the frequent species and failing on the rare ones. This script puts each
species' test F1 next to the number of training recordings it actually had
in a run, so the imbalance strategies can be judged on the classes they are
meant to help:

  * a scatter of per-class F1 against training recordings (log x) per run,
    with the Spearman rank correlation (near 0: F1 does not depend on how
    much data a species had; strongly positive: the head classes win)
  * mean F1 in head / medium / tail bands, the tertiles of species ranked
    by training recordings

When every species had the same number of recordings (the pilot) the
correlation and the bands are undefined and left empty. On the full
species40 set (25-30 recordings each) they are computed but mean little,
since most species tie at 30 and the head / medium split among them is
arbitrary.

Runs trained on a long-tailed subset (config "long_tail" > 1) are counted
on that subset, rebuilt with long_tail_subset and the run's subset seed
("lt_seed", or "seed" if the run has none), the same call src/train.py makes.

Usage:
    python analysis/long_tail_analysis.py --data species40 --name lt10 \
        --runs crnn_species40_lt10_s42 crnn_species40_lt10_effective_s42
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.utils.imbalance import long_tail_subset  # noqa: E402

BANDS = ("head", "medium", "tail")


def train_files(index, cfg):
    """Training recordings per species as the run saw them (after the long-tail subset, if any)."""
    ratio = float(cfg.get("long_tail") or 1)
    if ratio > 1:
        seed = cfg.get("lt_seed", cfg.get("seed", C.SEED))     # train.py draws the subset with --lt-seed
        index = index.iloc[long_tail_subset(index, ratio, seed=seed)]
    return index.groupby("species_name").file_name.nunique()


def load_run(run, index):
    """Per-class scores of one run with its training counts, or None if the scores are missing."""
    metrics_dir = C.RESULTS_DIR / "metrics"
    per_class, result = metrics_dir / f"{run}_per_class.csv", metrics_dir / f"{run}.json"
    if not per_class.exists():
        print(f"skipping {run}: {per_class} not found")
        return None, None
    cfg = {}
    if result.exists():
        cfg = json.loads(result.read_text()).get("config", {})
    else:
        print(f"{run}: {result.name} not found, assuming the full training set was used")
    df = pd.read_csv(per_class)
    df["train_files"] = df.species.map(train_files(index, cfg)).fillna(0).astype(int)
    if (df.train_files == 0).any():
        print(f"{run}: {(df.train_files == 0).sum()} species have no training files in this index - check --data")
    return df, float(cfg.get("long_tail") or 1)


def spearman(x, y):
    """Rank correlation: scipy if installed, else Pearson correlation of average ranks."""
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")                    # undefined when every species has the same value
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float(x.rank().corr(y.rank()))
    return float(spearmanr(x, y)[0])


def bands(df):
    """Tertiles of species by training recordings, most first (ties broken by name)."""
    order = df.sort_values(["train_files", "species"], ascending=[False, True]).index
    return dict(zip(BANDS, (df.loc[part] for part in np.array_split(order, 3))))


def summarise(run, df, ratio):
    row = {"run": run, "long_tail": ratio, "train_files": int(df.train_files.sum()),
           "spearman_rho": spearman(df.train_files, df.f1), "macro_f1": df.f1.mean()}
    parts = bands(df)
    balanced = df.train_files.nunique() < 2            # equal counts: the bands would be an arbitrary split
    for b in BANDS:
        row[f"{b}_f1"] = float("nan") if balanced else parts[b].f1.mean()
    for b in BANDS:
        row[f"{b}_files"] = f"{parts[b].train_files.min()}-{parts[b].train_files.max()}"
    return row


def plot(panels, path):
    cols = min(len(panels), 3)
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), sharey=True, squeeze=False)
    x_max = max(df.train_files.max() for _, df in panels) * 1.3      # same x range in every panel
    for ax, (row, df) in zip(axes.flat, panels):
        ax.scatter(df.train_files, df.f1, s=30, color="#2a6f97", edgecolor="white", linewidth=0.6, zorder=3)
        rho = "n/a (equal counts)" if np.isnan(row["spearman_rho"]) else f"{row['spearman_rho']:.2f}"
        ax.set_title(f"{row['run']}\nSpearman rho = {rho}", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.8, x_max)
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training recordings (log scale)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.85", lw=0.5)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("test F1 (file level)", fontsize=8)
    axes[0, 0].set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    index_path = C.PROCESSED_DIR / args.data / "Train_index.csv"
    if not index_path.exists():
        sys.exit(f"{index_path} not found: build the '{args.data}' features first")
    index = pd.read_csv(index_path)

    panels = []
    for run in args.runs:
        df, ratio = load_run(run, index)
        if df is not None:
            panels.append((summarise(run, df, ratio), df))
    if not panels:
        sys.exit("none of the given runs has a per-class results file")

    table = pd.DataFrame([row for row, _ in panels])
    out = C.RESULTS_DIR
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "metrics" / f"longtail_{args.name}.csv", index=False)
    plot(panels, out / "figures" / f"longtail_{args.name}.png")
    print(table.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
