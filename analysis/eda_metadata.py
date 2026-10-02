"""Exploratory analysis of the InsectSet459 metadata (no audio needed).

Produces the figures and numbers used in the interim report:
  * rank-frequency (long-tail) curve of recordings per species
  * how many species / recordings each candidate working-subset threshold keeps
  * coverage of the ambient-temperature field, by insect group
  * temperature distribution for the species that have enough labelled files

Usage:
    python analysis/eda_metadata.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

FIG = C.RESULTS_DIR / "figures"


def long_tail(df, stats):
    counts = df.species_name.value_counts()
    groups = df.groupby("species_name").group.first().loc[counts.index]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for g, colour in [("Orthoptera", "#2a6f97"), ("Cicadidae", "#c8553d")]:
        mask = (groups == g).values
        ax.bar(np.arange(len(counts))[mask], counts.values[mask], width=1.0, color=colour, label=g)
    for t, ls in [(25, ":"), (40, "--")]:
        ax.axhline(t, color="black", lw=0.8, ls=ls, label=f"{t} recordings")
    ax.set_yscale("log")
    ax.set_xlabel("species rank")
    ax.set_ylabel("recordings (log scale)")
    ax.set_title("InsectSet459: recordings per species (459 species)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "eda_long_tail.png", dpi=200)
    plt.close(fig)

    stats["recordings_per_species"] = {
        "max": int(counts.max()), "median": float(counts.median()), "min": int(counts.min()),
        "gini": float(gini(counts.values)),
        "top10pct_species_share_of_files": float(counts.head(len(counts) // 10).sum() / counts.sum()),
    }
    stats["thresholds"] = {
        str(t): {"species": int((counts >= t).sum()), "files": int(counts[counts >= t].sum())}
        for t in (10, 25, 40, 60, 100)
    }
    stats["species_below_25"] = int((counts < 25).sum())


def species_hours(readme=C.ROOT / "data/meta/Train_Val_Test_README.txt"):
    """Per-species duration from the species table in the dataset README."""
    import re
    hours = {}
    for line in open(readme):
        m = re.match(r"^([A-Z][a-z]+ [a-z\-]+(?: [a-z\-]+)?)\s+\t?(\d+)\s*\t(\d+):(\d+):(\d+)", line)
        if m:
            h, mi, se = map(int, m.groups()[2:])
            hours[m.group(1).replace(" ", "_")] = h + mi / 60 + se / 3600
    return hours


def threshold_table(df, stats):
    """Working-subset candidates: species, files, hours and official split sizes per threshold."""
    counts, hours = df.species_name.value_counts(), species_hours()
    rows = []
    for t in (10, 25, 40, 60, 100):
        sp = counts[counts >= t].index
        v = df[df.species_name.isin(sp)].subset.value_counts()
        rows.append({"threshold": t, "species": int(len(sp)), "files": int(counts[sp].sum()),
                     "hours": round(sum(hours.get(s, 0) for s in sp), 1),
                     "train": int(v.Train), "val": int(v.Validation), "test": int(v.Test)})
    stats["working_subset_table"] = rows
    (C.RESULTS_DIR / "metrics" / "working_subset_thresholds.json").write_text(json.dumps(rows, indent=2))


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def temperature(df, stats):
    t = df.dropna(subset=["temperature"])
    stats["temperature"] = {
        "files_with_temperature": int(len(t)),
        "fraction": float(len(t) / len(df)),
        "by_group": t.group.value_counts().to_dict(),
        "species_with_any": int(t.species_name.nunique()),
        "species_with_>=10": int((t.species_name.value_counts() >= 10).sum()),
        "range_c": [float(t.temperature.min()), float(t.temperature.max())],
        "mean_c": float(t.temperature.mean()),
        "by_split": t.subset.value_counts().to_dict(),
    }
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for g, colour in [("Orthoptera", "#2a6f97"), ("Cicadidae", "#c8553d")]:
        ax[0].hist(t[t.group == g].temperature, bins=30, alpha=0.7, color=colour, label=g)
    ax[0].set(xlabel="ambient temperature (°C)", ylabel="recordings")
    ax[0].legend(fontsize=8)
    top = t.species_name.value_counts().head(8).index
    data = [t[t.species_name == s].temperature.values for s in top]
    ax[1].boxplot(data, vert=False, labels=[s.replace("_", " ") for s in top])
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_xlabel("ambient temperature (°C)")
    fig.suptitle("Ambient temperature metadata", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "eda_temperature.png", dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(C.META_CSV)
    # recordist-entered temperatures outside a plausible range are treated as missing
    df.loc[(df.temperature < -5) | (df.temperature > 50), "temperature"] = np.nan
    stats = {"files": int(len(df)), "species": int(df.species_name.nunique()),
             "by_group": df.groupby("group").species_name.nunique().to_dict(),
             "files_by_group": df.group.value_counts().to_dict(),
             "format": df.file_name.str.extract(r"\.(\w+)$")[0].value_counts().to_dict(),
             "split": df.subset.value_counts().to_dict()}
    long_tail(df, stats)
    threshold_table(df, stats)
    temperature(df, stats)
    (C.RESULTS_DIR / "metrics" / "eda_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
