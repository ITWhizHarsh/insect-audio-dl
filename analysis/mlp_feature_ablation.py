"""Which handcrafted feature groups carry the MLP's signal?

The 146-d vector of src/features/handcrafted.py is five groups, each stored
as [mean over time, std over time] of its frame-level descriptors:

    mfcc      0:40     20 MFCCs
    delta     40:80    first temporal derivative of the MFCCs
    delta2    80:120   second derivative
    spectral  120:132  centroid, bandwidth, roll-off, flatness, ZCR, RMS
    contrast  132:146  7-band spectral contrast

The MLP is retrained on all features, on each group alone, and on all
features minus each group, with several seeds. A group that is strong alone
but costs little when removed carries information the other groups also
have; a group whose removal hurts carries something only it has.

Protocol of the MLP runs in src/train.py: train-only standardisation, AdamW
(lr 1e-3, weight decay 1e-4), batch 64, one-cycle schedule (10% warm-up),
150 epochs, the epoch with the best validation file-level macro-F1 kept and
then scored on test. The loop is copied here (CPU, a few lines) so the
ablation does not change when the shared training script does.

Usage:
    python analysis/mlp_feature_ablation.py --data species40     # 11 configs x 3 seeds, ~30 min on CPU
    python analysis/mlp_feature_ablation.py --data pilot --epochs 5 --seeds 42 --name ablation_smoke
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.handcrafted import FEATURE_DIM  # noqa: E402
from src.models.mlp import MLP  # noqa: E402

GROUPS = {"mfcc": (0, 40), "delta": (40, 80), "delta2": (80, 120),
          "spectral": (120, 132), "contrast": (132, 146)}
COLOURS = {"all": "#2a78d6", "only": "#eb6834", "without": "#1baf7a"}


def configurations():
    """(name, kind, feature columns) for all features, each group alone, and all minus each group."""
    every = np.arange(FEATURE_DIM)
    out = [("all features", "all", every)]
    out += [(f"only {g}", "only", np.arange(a, b)) for g, (a, b) in GROUPS.items()]
    out += [(f"without {g}", "without", np.setdiff1d(every, np.arange(a, b))) for g, (a, b) in GROUPS.items()]
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(data_dir, split):
    return pd.read_csv(data_dir / f"{split}_index.csv"), np.load(data_dir / f"{split}_hand.npy")


@torch.no_grad()
def predict(net, x, batch=64):
    net.eval()
    return torch.cat([torch.softmax(net(x[i:i + batch]), 1) for i in range(0, len(x), batch)]).numpy()


def train_once(splits, classes, cols, seed, epochs, lr=1e-3, wd=1e-4, batch=64):
    """Train one MLP on the given feature columns; returns (best epoch, validation report, test report)."""
    (tr_idx, x_tr), (va_idx, x_va), (te_idx, x_te) = splits
    n_classes = len(classes)
    seed_everything(seed)
    x_tr = x_tr[:, cols]
    # statistics in float64, applied in float32, as in src/train.py
    mean = torch.tensor(x_tr.mean(0, dtype=np.float64), dtype=torch.float32)
    std = torch.tensor(x_tr.std(0, dtype=np.float64) + 1e-6, dtype=torch.float32)
    x_va = (torch.from_numpy(x_va[:, cols]) - mean) / std
    x_te = (torch.from_numpy(x_te[:, cols]) - mean) / std
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(tr_idx.label.values)),
                        batch_size=batch, shuffle=True, drop_last=True)

    net = MLP(len(cols), n_classes)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.1)
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        net.train()
        for xb, yb in loader:
            # same loss expression as src/train.py with --weights none
            target = F.one_hot(yb, n_classes).float()
            loss = -(target * F.log_softmax(net((xb - mean) / std), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        _, y_f, p_f = metrics.file_level(predict(net, x_va), va_idx)
        f1 = metrics.scores(y_f, p_f.argmax(1), n_classes)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch, copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    va, _, _ = metrics.full_report(predict(net, x_va), va_idx, classes)
    te, _, _ = metrics.full_report(predict(net, x_te), te_idx, classes)
    return best_epoch, va, te


def summarise(runs):
    """Mean and sd over seeds per configuration, in the order the configurations were run."""
    cols = ["val_macro_f1", "test_macro_f1", "test_accuracy"]
    s = runs.groupby("config", sort=False).agg(
        kind=("kind", "first"), n_features=("n_features", "first"), n_seeds=("seed", "size"),
        **{f"{c}_{stat}": (c, stat) for c in cols for stat in ("mean", "std")})
    return s.reset_index()


def plot(summary, path, title):
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(s) + 1.2))
    ax.barh(s.config, s.test_macro_f1_mean, xerr=s.test_macro_f1_std.fillna(0), height=0.6,
            color=[COLOURS[k] for k in s.kind], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"})
    ax.axvline(summary.test_macro_f1_mean.iloc[0], color="#52514e", lw=1, ls="--")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLOURS.items()],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("test macro-F1 (file level), mean $\\pm$ sd over seeds")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--name", default=None, help="output name (default mlp_ablation_<data>)")
    args = ap.parse_args()
    name = args.name or f"mlp_ablation_{args.data}"

    data_dir = C.PROCESSED_DIR / args.data
    classes = json.loads((data_dir / "classes.json").read_text())
    splits = [load(data_dir, s) for s in C.SPLITS]
    assert splits[0][1].shape[1] == FEATURE_DIM

    rows, t0 = [], time.time()
    for config, kind, cols in configurations():
        for seed in args.seeds:
            best_epoch, va, te = train_once(splits, classes, cols, seed, args.epochs)
            rows.append({"config": config, "kind": kind, "n_features": len(cols), "seed": seed,
                         "best_epoch": best_epoch, "val_macro_f1": va["file"]["macro_f1"],
                         "test_macro_f1": te["file"]["macro_f1"], "test_accuracy": te["file"]["accuracy"],
                         "test_chunk_macro_f1": te["chunk"]["macro_f1"]})
            print(f"{config:18s} seed {seed}  val F1 {va['file']['macro_f1']:.3f}  "
                  f"test F1 {te['file']['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    runs = pd.DataFrame(rows)
    summary = summarise(runs)
    out = C.RESULTS_DIR / "metrics"
    runs.round(4).to_csv(out / f"{name}.csv", index=False)
    summary.round(4).to_csv(out / f"{name}_summary.csv", index=False)
    plot(summary, C.RESULTS_DIR / "figures" / f"{name}.png",
         f"MLP feature-group ablation, {args.data}, seeds {' '.join(map(str, args.seeds))}")
    print(summary.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
