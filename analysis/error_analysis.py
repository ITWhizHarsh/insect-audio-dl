"""Where do the models go wrong, and do they go wrong on the same recordings?

Reads the file-level prediction files saved by src/train.py
(results/predictions/<run>_<split>.npz) for several runs and reports:

  a. the species pairs each model confuses most, and the pairs every model
     confuses (more likely similar-sounding species or label problems than
     a weakness of one architecture);
  b. accuracy on Orthoptera vs Cicadidae, and how many errors cross the
     group boundary (a cricket called a cicada is a coarse error, a cricket
     called another cricket a fine-grained one);
  c. agreement: files every model gets right / wrong, the oracle accuracy
     (at least one model right, an upper bound for any way of combining
     them) and an ensemble that averages the models' probabilities, scored
     with the same metrics as the single models;
  d. per-species F1 of every model side by side (CSV);
  e. a bar chart of per-species F1 sorted by the mean over the models.

Runs may list files in different orders, so they are aligned on file name
and only files present in every run are compared.

Usage:
    python analysis/error_analysis.py --runs mlp_species40_s42 cnn_species40_s42 \
        crnn_species40_s42 --split test --name species40_s42
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.stats import load_predictions  # noqa: E402

GROUP_COLOURS = {"Orthoptera": "#2a78d6", "Cicadidae": "#eb6834"}
MARKERS = "os^Dvp"


def align(preds):
    """Keep the files present in every run, in the first run's order -> files, labels, {run: probs}."""
    first = next(iter(preds.values()))
    common = set.intersection(*(set(p["files"]) for p in preds.values()))
    files = np.array([f for f in first["files"] if f in common])
    labels, probs = None, {}
    for run, p in preds.items():
        if list(p["classes"]) != list(first["classes"]):
            raise ValueError(f"{run} was trained on a different class list")
        pos = {f: i for i, f in enumerate(p["files"])}
        idx = np.array([pos[f] for f in files])
        probs[run] = p["probs"][idx]
        if labels is None:
            labels = p["labels"][idx]
        elif not np.array_equal(labels, p["labels"][idx]):
            raise ValueError(f"{run} has different labels for the same files")
    return files, labels, probs


def pair_counts(cm):
    """Confusions between species i and j in either direction, upper triangle only."""
    return np.triu(cm + cm.T, 1)


def top_pairs(cm, classes, groups, top):
    pairs = pair_counts(cm)
    i, j = np.nonzero(pairs)
    order = np.argsort(-pairs[i, j], kind="stable")[:top]
    return [{"a": classes[i[k]], "b": classes[j[k]], "count": int(pairs[i[k], j[k]]),
             "a_as_b": int(cm[i[k], j[k]]), "b_as_a": int(cm[j[k], i[k]]),
             "cross_group": bool(groups[i[k]] != groups[j[k]])} for k in order]


def shared_pairs(cms, classes):
    """Pairs confused by every model, most confused first."""
    pairs = {run: pair_counts(cm) for run, cm in cms.items()}
    everyone = np.logical_and.reduce([p > 0 for p in pairs.values()])
    i, j = np.nonzero(everyone)
    rows = [{"a": classes[a], "b": classes[b], **{run: int(p[a, b]) for run, p in pairs.items()}}
            for a, b in zip(i, j)]
    return sorted(rows, key=lambda r: -sum(r[run] for run in pairs))


def group_report(labels, pred, groups):
    true_g, pred_g = groups[labels], groups[pred]
    out = {}
    for g in np.unique(true_g):
        mask = true_g == g
        out[g] = {"n_files": int(mask.sum()), "accuracy": float(np.mean(pred[mask] == labels[mask]))}
    wrong = pred != labels
    out["errors"] = int(wrong.sum())
    out["cross_group_errors"] = int((wrong & (true_g != pred_g)).sum())
    return out


def plot_per_class(table, runs, path, title):
    n = len(table)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(min(max(7, 0.2 * n), 30), 4))
    ax.bar(x, table.mean_f1, width=0.8, color=table["group"].map(GROUP_COLOURS).fillna("#8a8a8a"))
    for k, run in enumerate(runs):
        ax.scatter(x, table[run], s=14, marker=MARKERS[k % len(MARKERS)], facecolors="none",
                   edgecolors="#222222", linewidths=0.8, zorder=3, label=run)
    groups = [Patch(color=c, label=f"{g} (mean of models)") for g, c in GROUP_COLOURS.items()
              if g in set(table["group"])]
    ax.legend(handles=groups + ax.get_legend_handles_labels()[0], fontsize=7, frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set(xlim=(-0.6, n - 0.4), ylim=(0, 1.05), ylabel="F1 per species")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    if n <= 80:
        ax.set_xticks(x, [s.replace("_", " ") for s in table.species], rotation=90, fontsize=6, fontstyle="italic")
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"{n} species, sorted by mean F1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")   # "tight" keeps the legend outside the axes
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--top", type=int, default=10, help="confused pairs listed per model")
    ap.add_argument("--results", type=Path, default=C.RESULTS_DIR)
    args = ap.parse_args()

    preds = {run: load_predictions(run, args.split, args.results) for run in args.runs}
    files, labels, probs = align(preds)
    classes = list(next(iter(preds.values()))["classes"])
    n = len(classes)
    meta = pd.read_csv(C.META_CSV, usecols=["species_name", "group"]).drop_duplicates("species_name")
    group_of = dict(zip(meta.species_name, meta.group))
    groups = np.array([group_of.get(c, "unknown") for c in classes])

    pred = {run: p.argmax(1) for run, p in probs.items()}
    cms = {run: confusion_matrix(labels, pr, labels=range(n)) for run, pr in pred.items()}
    right = np.stack([pred[run] == labels for run in args.runs], axis=1)
    ensemble = np.mean([probs[run] for run in args.runs], axis=0).argmax(1)

    report = {
        "name": args.name, "split": args.split, "runs": args.runs, "n_files": int(len(files)),
        "files_dropped_to_align": {run: int(len(p["files"]) - len(files)) for run, p in preds.items()},
        "scores": {run: metrics.scores(labels, pr, n) for run, pr in pred.items()},
        "ensemble_mean_probs": metrics.scores(labels, ensemble, n),
        "agreement": {"all_right": int(right.all(1).sum()), "all_wrong": int((~right.any(1)).sum()),
                      "oracle_accuracy": float(right.any(1).mean())},
        "groups": {run: group_report(labels, pr, groups) for run, pr in pred.items()},
        "top_confusions": {run: top_pairs(cm, classes, groups, args.top) for run, cm in cms.items()},
        "shared_confusions": shared_pairs(cms, classes),
        "all_wrong_files": [{"file": f, "true": classes[y], **{run: classes[pred[run][k]] for run in args.runs}}
                            for k, (f, y) in enumerate(zip(files, labels)) if not right[k].any()],
    }
    report["groups"]["ensemble_mean_probs"] = group_report(labels, ensemble, groups)

    table = pd.DataFrame({"species": classes, "group": groups,
                          "support": np.bincount(labels, minlength=n)})
    for run in args.runs:
        table[run] = metrics.per_class(labels, pred[run], classes).f1.values
    table["mean_f1"] = table[args.runs].mean(axis=1)
    table = table[table.support > 0].sort_values("mean_f1", kind="stable")   # species absent from the split carry no F1

    out_m, out_f = args.results / "metrics", args.results / "figures"
    out_m.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    metrics.save_json(report, out_m / f"error_{args.name}.json")
    table.round(4).to_csv(out_m / f"error_{args.name}_per_class_f1.csv", index=False)
    plot_per_class(table, args.runs, out_f / f"error_{args.name}_per_class_f1.png",
                   f"Per-species F1 on {args.split} files ({args.name})")

    for run, s in report["scores"].items():
        print(f"{run:32s} acc {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}")
    e, a = report["ensemble_mean_probs"], report["agreement"]
    print(f"{'ensemble (mean probs)':32s} acc {e['accuracy']:.3f}  macro-F1 {e['macro_f1']:.3f}")
    print(f"oracle acc {a['oracle_accuracy']:.3f}; all right {a['all_right']}, all wrong {a['all_wrong']} "
          f"of {len(files)} files; {len(report['shared_confusions'])} pairs confused by every model")


if __name__ == "__main__":
    main()
