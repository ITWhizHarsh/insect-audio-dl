"""Collect the result JSONs of one dataset into a single comparison table.

Each configuration is trained with several seeds, and the seed is the last
part of the run name (cnn_species40_s42 ... cnn_species40_s46). One row of
the table is one model + variant, with the mean and sample sd over its
seeds. That sd is the variation due to initialisation and data order; the
test-set uncertainty of a single run comes from src/eval/stats.py. Older
result files (the 12-species pilot) have no seed tag and form a group of
one; columns they never recorded (recordist-unseen test score) stay blank.

The CSV keeps means and sds as separate numeric columns so the report can
plot from it; the printed / markdown table shows "mean ± sd".

Usage:
    python -m src.eval.compare --data species40 --markdown
    python -m src.eval.compare --data pilot
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.eval.stats import aggregate_seeds

ORDER = {"mlp": 0, "cnn": 1, "effnet": 2, "crnn": 3, "ast": 4}
NAMES = {"mlp": "MLP (handcrafted)", "cnn": "CNN (log-mel)", "effnet": "EfficientNetV2-S (ImageNet)",
         "crnn": "CRNN (CNN + BiLSTM)", "ast": "AST (AudioSet)"}


def run_metrics(r):
    """The numbers one run contributes to the table; None where an older run did not record them."""
    test, val = r["test"]["file"], r["validation"]["file"]
    unseen = r.get("test_subsets", {}).get("recordist_unseen", {})
    return {"params": r["params"] / 1e6, "test_f1": test["macro_f1"], "test_acc": test["accuracy"],
            "val_f1": val["macro_f1"], "chunk_f1": r["test"]["chunk"]["macro_f1"],
            "unseen_f1": unseen.get("macro_f1"), "train_s": r["train_seconds"]}


def load_groups(metrics_dir, data):
    """{(model, run name without seed): [run_metrics, ...]} for the runs on exactly this dataset."""
    groups = {}
    for path in sorted(metrics_dir.glob(f"*_{data}*.json")):
        r = json.loads(path.read_text())
        if r.get("data") != data or not {"model", "test", "validation"} <= r.keys():   # other datasets, non-run JSONs
            continue
        group = re.sub(r"_s\d+$", "", r.get("run", path.stem))
        groups.setdefault((r["model"], group), []).append(run_metrics(r))
    return groups


def build_table(groups, data):
    rows = []
    for (model, group), runs in groups.items():
        prefix = f"{model}_{data}"
        variant = group[len(prefix):].strip("_") if group.startswith(prefix) else group
        a = aggregate_seeds(runs)
        rows.append({
            "model": NAMES.get(model, model), "variant": variant or "-", "seeds": len(runs),
            "params (M)": a["params"]["mean"],
            "test macro-F1 mean": a["test_f1"]["mean"], "test macro-F1 sd": a["test_f1"]["sd"],
            "test acc mean": a["test_acc"]["mean"], "test acc sd": a["test_acc"]["sd"],
            "val macro-F1 mean": a["val_f1"]["mean"], "test chunk macro-F1 mean": a["chunk_f1"]["mean"],
            "recordist-unseen macro-F1 mean": a["unseen_f1"]["mean"], "train time (s) mean": a["train_s"]["mean"],
            "_order": ORDER.get(model, 9),
        })
    return pd.DataFrame(rows).sort_values(["_order", "variant"]).drop(columns="_order")


def pretty(table):
    def pm(m, s):
        return f"{m:.3f}" if np.isnan(s) else f"{m:.3f} ± {s:.3f}"

    def num(v):
        return "" if np.isnan(v) else f"{v:.3f}"

    t = table
    return pd.DataFrame({
        "model": t["model"], "variant": t["variant"], "seeds": t["seeds"],
        "params (M)": t["params (M)"].map("{:.2f}".format),
        "test macro-F1": [pm(m, s) for m, s in zip(t["test macro-F1 mean"], t["test macro-F1 sd"])],
        "test acc": [pm(m, s) for m, s in zip(t["test acc mean"], t["test acc sd"])],
        "val macro-F1": t["val macro-F1 mean"].map(num),
        "test chunk macro-F1": t["test chunk macro-F1 mean"].map(num),
        "recordist-unseen macro-F1": t["recordist-unseen macro-F1 mean"].map(num),
        "train time (s)": t["train time (s) mean"].map("{:.0f}".format),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--out", type=Path, help="CSV path (default results/metrics/comparison_<data>.csv)")
    ap.add_argument("--markdown", action="store_true", help="also write results/comparison_<data>.md")
    ap.add_argument("--results", type=Path, default=C.RESULTS_DIR)
    args = ap.parse_args()

    groups = load_groups(args.results / "metrics", args.data)
    if not groups:
        raise SystemExit(f"no result JSONs for data '{args.data}' in {args.results / 'metrics'}")
    table = build_table(groups, args.data)
    out = args.out or args.results / "metrics" / f"comparison_{args.data}.csv"
    table.round(4).to_csv(out, index=False)
    text = pretty(table).to_markdown(index=False)
    print(text)
    print(f"wrote {out}")
    if args.markdown:
        md = args.results / f"comparison_{args.data}.md"
        md.write_text(f"# Model comparison: {args.data}\n\n"
                      "File-level scores on the official test split; mean ± sample sd over seeds.\n\n"
                      f"{text}\n")
        print(f"wrote {md}")


if __name__ == "__main__":
    main()
