"""Are the differences between the models real? Significance for the comparison table.

The comparison table (src/eval/compare.py) gives each model's mean +- sd over
training seeds. This script asks the complementary question: with only ~400
test recordings, which gaps between models would survive a different sample
of test files? For each model it averages the file-level probabilities of its
seeds (a seed ensemble, so one unlucky initialisation does not decide the
test) and then

  * a bootstrap 95% CI of accuracy and macro-F1 over test files
  * for every pair of models, on the same test files: a paired bootstrap of
    the macro-F1 difference and McNemar's test on the files only one of the
    two gets right
  * the same paired tests on the recordist-unseen test files only, where the
    official split's recordist overlap cannot flatter either model

Usage:
    python analysis/model_comparison.py --data species40 \\
        --models mlp_species40 cnn_species40 crnn_species40 ast_species40_lite
"""
import argparse
import itertools
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
from src.data.recordist import seen_mask  # noqa: E402
from src.eval import stats  # noqa: E402

NAMES = {"mlp": "MLP", "cnn": "CNN", "crnn": "CRNN", "ast": "AST", "effnet": "EfficientNetV2-S"}


def seeds_of(prefix):
    """Run names <prefix>_s<seed> with a result file, e.g. cnn_species40 -> cnn_species40_s42 ..."""
    pattern = re.compile(rf"^{re.escape(prefix)}_s\d+$")
    return sorted(p.stem for p in (C.RESULTS_DIR / "metrics").glob(f"{prefix}_s*.json") if pattern.match(p.stem))


def label(prefix, data):
    model, _, variant = prefix.partition(f"_{data}")
    name = NAMES.get(model, model)
    return f"{name} ({variant.strip('_')})" if variant else name


def ensemble(runs):
    preds = [stats.load_predictions(r) for r in runs]
    for p in preds[1:]:
        assert np.array_equal(p["files"], preds[0]["files"]), "runs scored different test files"
    return preds[0]["files"], preds[0]["labels"], np.mean([p["probs"] for p in preds], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="species40")
    ap.add_argument("--models", nargs="+", required=True, help="run prefixes without the _s<seed> tag")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    data_dir = C.PROCESSED_DIR / args.data
    unseen = ~seen_mask(pd.read_csv(data_dir / "Test_index.csv"), pd.read_csv(data_dir / "Train_index.csv"))

    models, rows = {}, []
    for prefix in args.models:
        runs = seeds_of(prefix)
        if not runs:
            print(f"skipping {prefix}: no runs")
            continue
        files, labels, probs = ensemble(runs)
        mask = unseen.reindex(files).fillna(False).values.astype(bool)
        name = label(prefix, args.data)
        models[name] = (labels, probs, mask)
        n_classes = probs.shape[1]
        ci = stats.bootstrap_ci(labels, probs, n_classes, n_boot=args.n_boot)
        per_seed = stats.aggregate_seeds([
            {"f1": json.loads((C.RESULTS_DIR / "metrics" / f"{r}.json").read_text())["test"]["file"]["macro_f1"]}
            for r in runs])["f1"]
        f1_unseen = stats.macro_f1(labels[mask], probs[mask].argmax(1), n_classes)
        rows.append({"model": name, "seeds": len(runs),
                     "macro-F1 per seed (mean)": per_seed["mean"], "sd": per_seed["sd"],
                     "ensemble macro-F1": ci["macro_f1"][0], "CI low": ci["macro_f1"][1], "CI high": ci["macro_f1"][2],
                     "ensemble accuracy": ci["accuracy"][0], "unseen macro-F1": f1_unseen})

    pairs = []
    for (a, (y, pa, mask)), (b, (_, pb, _)) in itertools.combinations(models.items(), 2):
        n_classes = pa.shape[1]
        for subset, m in (("all", np.ones_like(mask)), ("recordist-unseen", mask)):
            pb_res = stats.paired_bootstrap(y[m], pa[m], pb[m], n_classes, n_boot=args.n_boot)
            mc = stats.mcnemar(y[m], pa[m].argmax(1), pb[m].argmax(1))
            pairs.append({"model A": a, "model B": b, "test files": subset, "n": int(m.sum()),
                          "macro-F1 A - B": pb_res["delta_macro_f1"], "CI low": pb_res["ci"][0], "CI high": pb_res["ci"][1],
                          "bootstrap p": pb_res["p_value"], "only A right": mc["b"], "only B right": mc["c"],
                          "McNemar p": mc["p_value"]})

    table, pair_table = pd.DataFrame(rows), pd.DataFrame(pairs)
    out = C.RESULTS_DIR / "metrics" / f"model_comparison_{args.data}"
    table.round(4).to_csv(f"{out}.csv", index=False)
    pair_table.round(4).to_csv(f"{out}_pairs.csv", index=False)
    print(table.round(3).to_markdown(index=False))
    print()
    print(pair_table.round(3).to_markdown(index=False))

    order = table.sort_values("ensemble macro-F1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(order) + 1.2))
    y = np.arange(len(order))
    ax.barh(y - 0.18, order["macro-F1 per seed (mean)"], 0.34, xerr=order["sd"].fillna(0), capsize=3,
            label="single run, mean ± sd over seeds", color="#4c78a8")
    ax.barh(y + 0.18, order["ensemble macro-F1"], 0.34,
            xerr=[order["ensemble macro-F1"] - order["CI low"], order["CI high"] - order["ensemble macro-F1"]],
            capsize=3, label="seed ensemble, 95% bootstrap CI", color="#f58518")
    ax.scatter(order["unseen macro-F1"], y + 0.18, marker="|", s=160, color="black", zorder=3,
               label="seed ensemble on recordist-unseen files")
    ax.set_yticks(y, order["model"])
    ax.set_xlabel("test macro-F1 (file level)")
    ax.set_xlim(0.4, 0.95)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Model comparison, {args.data}", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "figures" / f"model_comparison_{args.data}.png", dpi=200)


if __name__ == "__main__":
    main()
