"""Common evaluation protocol: the same metrics for every model.

Models predict on 5 s chunks, but labels in InsectSet459 belong to whole
recordings, so the primary scores are file-level: chunk softmax outputs are
averaged per recording and the arg-max is the file prediction. Chunk-level
scores are reported as well.

Macro-F1 is the headline metric because the full dataset is long-tailed;
accuracy alone would be dominated by the most frequent species.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def file_level(probs, index_df):
    """Average chunk probabilities per recording -> (file_names, labels, probs)."""
    df = pd.DataFrame(probs)
    df["file_name"] = index_df.file_name.values
    mean = df.groupby("file_name", sort=False).mean()
    labels = index_df.groupby("file_name", sort=False).label.first().loc[mean.index].values
    return mean.index.values, labels, mean.values


def scores(y_true, y_pred, n_classes):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p,
            "macro_recall": r, "macro_f1": f, "weighted_f1": f_w}


def per_class(y_true, y_pred, classes):
    p, r, f, n = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    return pd.DataFrame({"species": classes, "precision": p, "recall": r, "f1": f, "support": n})


def full_report(chunk_probs, index_df, classes):
    n = len(classes)
    chunk_pred = chunk_probs.argmax(1)
    files, y_file, file_probs = file_level(chunk_probs, index_df)
    y_file_pred = file_probs.argmax(1)
    return {
        "chunk": scores(index_df.label.values, chunk_pred, n),
        "file": scores(y_file, y_file_pred, n),
        "n_chunks": int(len(index_df)), "n_files": int(len(files)),
    }, per_class(y_file, y_file_pred, classes), confusion_matrix(y_file, y_file_pred, labels=range(n))


def plot_confusion(cm, classes, path, title):
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), short, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
