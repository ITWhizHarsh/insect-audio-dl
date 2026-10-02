"""Collect every results/metrics/<model>_<data>.json into one comparison table.

Usage:
    python -m src.eval.compare --data pilot
"""
import argparse
import json

import pandas as pd

from src import config as C

ORDER = {"mlp": 0, "cnn": 1, "crnn": 2, "ast": 3}
NAMES = {"mlp": "MLP (handcrafted)", "cnn": "CNN (log-mel)", "crnn": "CRNN (CNN + BiLSTM)", "ast": "AST"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    args = ap.parse_args()

    rows = []
    for path in sorted((C.RESULTS_DIR / "metrics").glob(f"*_{args.data}.json")):
        r = json.loads(path.read_text())
        if "model" not in r:
            continue
        t, v = r["test"], r["validation"]
        rows.append({
            "model": NAMES.get(r["model"], r["model"]), "params (M)": round(r["params"] / 1e6, 2),
            "best epoch": r["best_epoch"], "train time (s)": r["train_seconds"],
            "val macro-F1": v["file"]["macro_f1"],
            "test acc (file)": t["file"]["accuracy"], "test macro-F1 (file)": t["file"]["macro_f1"],
            "test macro-P": t["file"]["macro_precision"], "test macro-R": t["file"]["macro_recall"],
            "test acc (chunk)": t["chunk"]["accuracy"], "test macro-F1 (chunk)": t["chunk"]["macro_f1"],
            "_order": ORDER.get(r["model"], 9),
        })
    table = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
    out = C.RESULTS_DIR / "metrics" / f"comparison_{args.data}.csv"
    table.round(4).to_csv(out, index=False)
    print(table.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
