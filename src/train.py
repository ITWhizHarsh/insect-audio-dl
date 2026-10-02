"""Shared training / evaluation loop used by every model.

Same splits, same seed, same model-selection rule (best validation
file-level macro-F1), same test metrics. Only the model and its input
representation change between runs.

Usage:
    python -m src.train --model mlp  --data pilot --epochs 150 --batch 64
    python -m src.train --model cnn  --data pilot --epochs 40  --augment
    python -m src.train --model crnn --data pilot --epochs 40  --augment
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import config as C
from src.eval import metrics
from src.features.augment import augment_batch
from src.models.cnn import LogMelCNN
from src.models.crnn import CRNN
from src.models.mlp import MLP
from src.utils.imbalance import class_weights

INPUT = {"mlp": "hand", "cnn": "logmel", "crnn": "logmel"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data_dir, split, kind):
    idx = pd.read_csv(data_dir / f"{split}_index.csv")
    x = np.load(data_dir / f"{split}_{kind}.npy", mmap_mode="r")
    return idx, np.asarray(x)


def build(model, n_classes, in_dim):
    if model == "mlp":
        return MLP(in_dim, n_classes)
    if model == "cnn":
        return LogMelCNN(n_classes)
    if model == "crnn":
        return CRNN(n_classes)
    raise ValueError(model)


@torch.no_grad()
def predict(net, x, kind, norm, device, batch=64):
    net.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = prepare(torch.from_numpy(np.asarray(x[i:i + batch])), kind, norm, device)
        out.append(torch.softmax(net(xb), 1).cpu())
    return torch.cat(out).numpy()


def prepare(xb, kind, norm, device):
    xb = (xb.float().to(device) - norm[0]) / norm[1]
    return xb.unsqueeze(1) if kind == "logmel" else xb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(INPUT), required=True)
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--weights", default="none", choices=["none", "inverse", "effective"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    kind = INPUT[args.model]
    data_dir = C.PROCESSED_DIR / args.data
    classes = json.loads((data_dir / "classes.json").read_text())
    n_classes = len(classes)
    device = torch.device("mps" if torch.backends.mps.is_available() and kind == "logmel" else "cpu")

    tr_idx, x_tr = load_split(data_dir, "Train", kind)
    va_idx, x_va = load_split(data_dir, "Validation", kind)
    te_idx, x_te = load_split(data_dir, "Test", kind)

    # normalisation statistics from the training split only
    if kind == "hand":
        norm = (torch.tensor(x_tr.mean(0), device=device), torch.tensor(x_tr.std(0) + 1e-6, device=device))
    else:
        m, s = float(x_tr.astype(np.float32).mean()), float(x_tr.astype(np.float32).std())
        norm = (m, s)

    y_tr = tr_idx.label.values
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=args.batch, shuffle=True, drop_last=True)

    net = build(args.model, n_classes, x_tr.shape[1] if kind == "hand" else None).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    cw = class_weights(y_tr, n_classes, args.weights).to(device)

    name = f"{args.model}_{args.data}{args.tag}"
    ckpt = C.ROOT / "outputs" / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    history, best_f1, t0 = [], -1.0, time.time()
    print(f"{name}: {n_params / 1e6:.2f}M params, {len(y_tr)} train chunks, device={device}", flush=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        for xb, yb in loader:
            xb = prepare(xb, kind, norm, device)
            target = F.one_hot(yb.to(device), n_classes).float()
            if args.augment and kind == "logmel":
                xb, target = augment_batch(xb, target)
            logits = net(xb)
            loss = -(target * F.log_softmax(logits, 1) * cw).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        va_probs = predict(net, x_va, kind, norm, device)
        _, y_f, p_f = metrics.file_level(va_probs, va_idx)
        va = metrics.scores(y_f, p_f.argmax(1), n_classes)
        history.append({"epoch": epoch, "train_loss": running / len(loader),
                        "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"]})
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save(net.state_dict(), ckpt)
        print(f"epoch {epoch:3d}  loss {running / len(loader):.3f}  val acc {va['accuracy']:.3f}  "
              f"val macro-F1 {va['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # test with the checkpoint chosen on validation
    net.load_state_dict(torch.load(ckpt, map_location=device))
    report, per_cls, cm = metrics.full_report(predict(net, x_te, kind, norm, device), te_idx, classes)
    va_report, _, _ = metrics.full_report(predict(net, x_va, kind, norm, device), va_idx, classes)
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    result = {"model": args.model, "data": args.data, "params": n_params, "config": vars(args),
              "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
              "validation": va_report, "test": report}

    out = C.RESULTS_DIR
    metrics.save_json(result, out / "metrics" / f"{name}.json")
    per_cls.to_csv(out / "metrics" / f"{name}_per_class.csv", index=False)
    pd.DataFrame(history).to_csv(out / "metrics" / f"{name}_history.csv", index=False)
    metrics.plot_confusion(cm, classes, out / "figures" / f"confusion_{name}.png",
                           f"{args.model.upper()} - test confusion matrix (file level)")
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss)
    ax[0].set(xlabel="epoch", ylabel="train loss")
    ax[1].plot(h.epoch, h.val_acc, label="val accuracy")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].set(xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / f"curves_{name}.png", dpi=200)
    print(json.dumps(result["test"], indent=2, default=float))


if __name__ == "__main__":
    main()
