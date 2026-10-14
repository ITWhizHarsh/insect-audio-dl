"""Classical (non-neural) baselines on the MLP's handcrafted features.

The MLP is the "no learned features" model of the comparison, but it is
still a neural network trained with a deep-learning recipe. Logistic
regression, an RBF support-vector machine and a random forest on exactly the
same 146-d chunk features show where the floor of the comparison really is,
and whether the MLP adds anything over standard classifiers.

Same protocol as src/train.py: features standardised with training-split
statistics only, chunk probabilities averaged per recording, file-level
macro-F1 as the primary score (src/eval/metrics.py), plus the test split
divided into recordist-seen and recordist-unseen files (src/data/recordist.py).
The settings below are common defaults; nothing is tuned on the test split.

Usage:
    python analysis/classical_baselines.py --data pilot
    python analysis/classical_baselines.py --data species40 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.recordist import seen_mask  # noqa: E402
from src.eval import metrics  # noqa: E402


def classifiers(seed):
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=2000),
        "svm_rbf": SVC(C=10.0, gamma="scale", probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=500, n_jobs=2, random_state=seed),
    }


def load(data_dir, split):
    return pd.read_csv(data_dir / f"{split}_index.csv"), np.load(data_dir / f"{split}_hand.npy")


def chunk_probs(clf, x, n_classes):
    # predict_proba only has columns for the classes present in training
    probs = np.zeros((len(x), n_classes))
    probs[:, clf.classes_] = clf.predict_proba(x)
    return probs


def recordist_scores(probs, index_df, seen, n_classes):
    """File-level test scores on recordist-seen and -unseen files, computed as in src/train.py.

    Macro averages run over the species present in each subset, so the numbers
    are comparable with the neural models' test_subsets.
    """
    files, labels, file_probs = metrics.file_level(probs, index_df)
    seen = seen.reindex(files).values.astype(bool)
    out = {}
    for name, mask in (("recordist_seen", seen), ("recordist_unseen", ~seen)):
        if mask.any():
            s = metrics.scores(labels[mask], file_probs[mask].argmax(1), n_classes, present_only=True)
            out[name] = {**s, "n_files": int(mask.sum()), "n_species": int(len(np.unique(labels[mask])))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    data_dir = C.PROCESSED_DIR / args.data
    classes = json.loads((data_dir / "classes.json").read_text())
    n = len(classes)
    tr_idx, x_tr = load(data_dir, "Train")
    va_idx, x_va = load(data_dir, "Validation")
    te_idx, x_te = load(data_dir, "Test")

    # normalisation statistics from the training split only, as in src/train.py
    mean, std = x_tr.mean(0, dtype=np.float64), x_tr.std(0, dtype=np.float64) + 1e-6
    x_tr, x_va, x_te = [(x - mean) / std for x in (x_tr, x_va, x_te)]
    seen = seen_mask(te_idx, tr_idx)

    results, rows = {}, []
    for name, clf in classifiers(args.seed).items():
        t0 = time.time()
        clf.fit(x_tr, tr_idx.label.values)
        fit_seconds = round(time.time() - t0, 2)
        te_probs = chunk_probs(clf, x_te, n)
        va, _, _ = metrics.full_report(chunk_probs(clf, x_va, n), va_idx, classes)
        te, _, _ = metrics.full_report(te_probs, te_idx, classes)
        sub = recordist_scores(te_probs, te_idx, seen, n)
        results[name] = {"params": clf.get_params(), "fit_seconds": fit_seconds,
                         "validation": va, "test": te, "test_subsets": sub}
        rows.append({"model": name, "fit (s)": fit_seconds,
                     "val macro-F1": va["file"]["macro_f1"],
                     "test acc (file)": te["file"]["accuracy"],
                     "test macro-F1 (file)": te["file"]["macro_f1"],
                     "test macro-F1 (chunk)": te["chunk"]["macro_f1"],
                     "test acc seen": sub.get("recordist_seen", {}).get("accuracy"),
                     "test acc unseen": sub.get("recordist_unseen", {}).get("accuracy")})
        print(f"{name}: fitted in {fit_seconds}s", flush=True)

    out = {"data": args.data, "seed": args.seed, "n_features": int(x_tr.shape[1]),
           "n_train_chunks": int(len(x_tr)), "models": results}
    path = C.RESULTS_DIR / "metrics" / f"classical_{args.data}.json"
    metrics.save_json(out, path)
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))
    print("saved", path)


if __name__ == "__main__":
    main()
