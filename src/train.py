"""Shared training / evaluation loop used by every model.

Same splits, same seed, same model-selection rule (best validation
file-level macro-F1), same test metrics. Only the model and its input
representation change between runs, plus the options of the controlled
experiments (augmentation ablation, long-tail imbalance, temperature).

Small input caches are loaded into RAM; larger ones (the 141-species log-mel
cache is several GB) are read through memory maps so they never have to fit
in RAM at once.

Every run writes, under results/:
    metrics/<run>.json               test + validation scores, config, recordist split
    metrics/<run>_per_class.csv      per-species precision / recall / F1 (test, file level)
    metrics/<run>_history.csv        loss and validation scores per epoch
    predictions/<run>_<split>.npz    file-level probabilities (for statistics and ensembles)
    figures/{confusion,curves}_<run>.png

Usage:
    python -m src.train --model mlp    --data species40 --epochs 150 --batch 64 --wd 1e-4
    python -m src.train --model cnn    --data species40 --epochs 30 --augment
    python -m src.train --model crnn   --data species40 --epochs 30 --augment --clip 1
    python -m src.train --model effnet --data species40 --epochs 15 --lr 3e-4 --augment --amp
    python -m src.train --model ast    --data species40 --epochs 10 --lr 5e-5 --batch 8 --augment --amp --patchout 0.5
Controlled experiments:
    --long-tail 10 --weights effective | --sampler balanced | --loss logit-adjusted
    --temperature real | shuffled      (Orthoptera temperature subset)
    --augment --no-mixup | --no-specaugment | --no-shift
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src import config as C
from src.eval import metrics
from src.features.augment import augment_batch
from src.models.cnn import LogMelCNN
from src.models.crnn import CRNN
from src.models.mlp import MLP
from src.utils import imbalance

INPUT = {"mlp": "hand", "cnn": "logmel", "crnn": "logmel", "effnet": "logmel", "ast": "logmel"}
HEADS = ("head", "fc", "classifier", "film")   # top-level names of the new, randomly initialised layers


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_data(name):
    path = Path(name)
    if path.is_dir():
        return path, path.name
    return C.PROCESSED_DIR / name, name


def load_split(data_dir, split, kind, in_memory):
    idx = pd.read_csv(data_dir / f"{split}_index.csv")
    x = np.load(data_dir / f"{split}_{kind}.npy", mmap_mode="r")
    return idx, (np.ascontiguousarray(x) if in_memory else x)


def fits_in_memory(data_dir, kind, limit_gb=2.0):
    """Small caches are read into RAM (random disk reads slow every batch); big ones stay memory-mapped."""
    size = sum((data_dir / f"{s}_{kind}.npy").stat().st_size for s in C.SPLITS)
    return size < limit_gb * 1e9


def take(x, rows):
    """Rows of a memory-mapped array, read in ascending order (fast on disk)."""
    return np.asarray(x[np.sort(rows)])


def normalisation(x, rows, kind, block=256):
    """Train-split statistics: per-dimension for handcrafted, one global mean / std for log-mel."""
    if kind == "hand":
        sub = take(x, rows).astype(np.float64)
        return sub.mean(0), sub.std(0) + 1e-6
    total, total_sq, n = 0.0, 0.0, 0
    rows = np.sort(rows)
    for i in range(0, len(rows), block):
        b = np.asarray(x[rows[i:i + block]], dtype=np.float64)
        total += b.sum()
        total_sq += np.square(b).sum()
        n += b.size
    mean = total / n
    return mean, float(np.sqrt(total_sq / n - mean ** 2))


def build(args, n_classes, in_dim):
    if args.model == "mlp":
        return MLP(in_dim, n_classes)
    if args.model == "cnn":
        return LogMelCNN(n_classes)
    if args.model == "crnn":
        return CRNN(n_classes, rnn=args.rnn)
    if args.model == "effnet":
        from src.models.effnet import EfficientNetV2S
        return EfficientNetV2S(n_classes, pretrained=not args.scratch)
    if args.model == "ast":
        from src.models.ast import AST
        return AST(n_classes, stride=args.ast_stride, patchout=args.patchout, freeze_blocks=args.freeze,
                   pretrained=not args.scratch, checkpoint=args.checkpoint)
    raise ValueError(args.model)


def file_temperatures(index_df, mode, train_files=None, seed=0):
    """Per-chunk (temperature, missing) inputs; 'shuffled' permutes the labelled training files' values."""
    from src.data.recordist import attach_metadata
    per_file = attach_metadata(index_df.drop_duplicates("file_name"))[["file_name", "temperature"]]
    temps = per_file.set_index("file_name").temperature
    if mode == "shuffled" and train_files is not None:
        labelled = temps[temps.notna() & temps.index.isin(train_files)]
        rng = np.random.default_rng(seed)
        temps.loc[labelled.index] = rng.permutation(labelled.values)
    return temps.reindex(index_df.file_name).values


class Inputs:
    """Normalises a batch and moves it to the device; log-mel batches get a channel axis."""

    def __init__(self, kind, mean, std, device):
        self.kind, self.device = kind, device
        if kind == "hand":
            self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
            self.std = torch.tensor(std, dtype=torch.float32, device=device)
        else:
            self.mean, self.std = float(mean), float(std)

    def __call__(self, xb):
        xb = (torch.from_numpy(xb).float().to(self.device) - self.mean) / self.std
        return xb.unsqueeze(1) if self.kind == "logmel" else xb


def forward(net, xb, tb):
    return net(xb) if tb is None else net(xb, tb)


@torch.no_grad()
def predict(net, x, prep, temps, device, args):
    net.eval()
    out = []
    for i in range(0, len(x), args.eval_batch):
        rows = np.arange(i, min(i + args.eval_batch, len(x)))
        tb = None if temps is None else torch.from_numpy(temps[rows]).to(device)
        with autocast(device, args.amp):
            logits = forward(net, prep(np.asarray(x[rows])), tb)
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy()


def autocast(device, enabled):
    """float16 mixed precision with loss scaling, CUDA only. On Apple MPS (torch 2.5) float16 training
    gives NaN, bfloat16 autocast is not supported and the gradient scaler needs float64, so runs stay float32."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=enabled and device.type == "cuda")


def param_groups(net, lr, head_mult):
    head, body = [], []
    for name, p in net.named_parameters():
        if p.requires_grad:
            (head if name.split(".")[0] in HEADS else body).append(p)
    if head_mult == 1 or not body:
        return [{"params": head + body, "lr": lr}]
    return [{"params": body, "lr": lr}, {"params": head, "lr": lr * head_mult}]


def save_predictions(probs, index_df, classes, path):
    files, labels, file_probs = metrics.file_level(probs, index_df)
    np.savez_compressed(path, files=files.astype(str), labels=labels.astype(np.int64),
                        probs=file_probs.astype(np.float32), classes=np.array(classes))
    return files, labels, file_probs


def recordist_scores(files, labels, file_probs, te_idx, tr_idx, n_classes):
    """Test scores split by whether the recordist also recorded that species in the training files."""
    from src.data.recordist import seen_mask
    seen = seen_mask(te_idx, tr_idx).reindex(files).fillna(False).values.astype(bool)
    out = {}
    for name, mask in (("recordist_seen", seen), ("recordist_unseen", ~seen)):
        if mask.any():
            s = metrics.scores(labels[mask], file_probs[mask].argmax(1), n_classes, present_only=True)
            out[name] = {**s, "n_files": int(mask.sum()), "n_species": int(len(np.unique(labels[mask])))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(INPUT), required=True)
    ap.add_argument("--data", default="pilot", help="cache name under data/processed or a directory")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr-mult", type=float, default=1.0, help="learning-rate multiplier for the new head")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0 = off)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--amp", action="store_true", help="float16 mixed precision (CUDA only)")
    ap.add_argument("--patience", type=int, default=0, help="stop after n epochs without improvement (0 = off)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--no-specaugment", action="store_true")
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--weights", default="none", choices=["none", "inverse", "effective"])
    ap.add_argument("--sampler", default="shuffle", choices=["shuffle", "balanced"])
    ap.add_argument("--loss", default="ce", choices=["ce", "logit-adjusted"])
    ap.add_argument("--long-tail", type=float, default=1.0, help="imbalance ratio of the training subset (1 = off)")
    ap.add_argument("--lt-seed", type=int, default=C.SEED, help="seed of the long-tail file selection")
    ap.add_argument("--temperature", default="none", choices=["none", "real", "shuffled"])
    ap.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--ast-stride", type=int, default=10)
    ap.add_argument("--patchout", type=float, default=0.0)
    ap.add_argument("--freeze", type=int, default=0, help="AST: number of frozen transformer blocks")
    ap.add_argument("--checkpoint", action="store_true", help="AST: gradient checkpointing")
    ap.add_argument("--scratch", action="store_true", help="random initialisation instead of pretrained weights")
    ap.add_argument("--mmap", action="store_true", help="always read inputs through memory maps")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    kind = INPUT[args.model]
    data_dir, data_name = resolve_data(args.data)
    classes = json.loads((data_dir / "classes.json").read_text())
    n_classes = len(classes)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and kind == "logmel":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    in_memory = not args.mmap and fits_in_memory(data_dir, kind)
    tr_idx, x_tr = load_split(data_dir, "Train", kind, in_memory)
    va_idx, x_va = load_split(data_dir, "Validation", kind, in_memory)
    te_idx, x_te = load_split(data_dir, "Test", kind, in_memory)

    rows = np.arange(len(tr_idx))
    if args.long_tail > 1:
        rows = np.sort(imbalance.long_tail_subset(tr_idx, args.long_tail, seed=args.lt_seed))
    used_idx = tr_idx.iloc[rows].reset_index(drop=True)
    y_tr = used_idx.label.values

    # normalisation statistics from the (used) training split only
    prep = Inputs(kind, *normalisation(x_tr, rows, kind), device)

    temps = {}
    if args.temperature != "none":
        raw = {s: file_temperatures(i, args.temperature, set(used_idx.file_name), args.seed)
               for s, i in (("train", used_idx), ("val", va_idx), ("test", te_idx))}
        labelled = pd.Series(raw["train"], index=used_idx.file_name).groupby(level=0).first().dropna()
        if len(labelled) < 2:
            raise ValueError(f"--temperature needs temperature-labelled training files, found {len(labelled)}")
        from src.models.conditioning import temperature_inputs
        temps = {s: temperature_inputs(v, labelled.mean(), labelled.std()) for s, v in raw.items()}

    net = build(args, n_classes, x_tr.shape[1] if kind == "hand" else None)
    if args.temperature != "none":
        from src.models.conditioning import TemperatureConditioned
        net = TemperatureConditioned(net, n_classes, dropout=0.0 if args.model == "ast" else 0.3)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    steps_per_epoch = len(rows) // args.batch
    groups = param_groups(net, args.lr, args.head_lr_mult)
    # on MPS (torch 2.5) the default single-tensor AdamW update turned the partly frozen AST's weights
    # into NaN after two steps; the multi-tensor ("foreach") implementation trains normally
    opt = torch.optim.AdamW(groups, weight_decay=args.wd, foreach=True if device.type == "mps" else None)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], pct_start=0.1,
        total_steps=args.epochs * max(steps_per_epoch // args.accum, 1) + 1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    cw = imbalance.class_weights(y_tr, n_classes, args.weights).to(device)
    offsets = None
    if args.loss == "logit-adjusted":
        offsets = imbalance.logit_adjustment(imbalance.class_counts(y_tr, n_classes)).to(device)

    run = f"{args.model}_{data_name}{args.tag}"
    out = C.RESULTS_DIR
    for sub in ("metrics", "figures", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ckpt = C.ROOT / "outputs" / f"{run}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    history, best_f1, stale, t0 = [], -1.0, 0, time.time()
    print(f"{run}: {n_params / 1e6:.2f}M params ({n_trainable / 1e6:.2f}M trainable), "
          f"{len(rows)} train chunks, {n_classes} classes, device={device}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.sampler == "balanced":
            order = np.array(list(imbalance.balanced_sampler(y_tr, n_classes)))
        else:
            order = torch.randperm(len(rows), generator=gen).numpy()
        running = 0.0
        opt.zero_grad()
        for step in range(steps_per_epoch):
            pick = np.sort(order[step * args.batch:(step + 1) * args.batch])
            xb = prep(np.asarray(x_tr[rows[pick]]))
            yb = torch.from_numpy(y_tr[pick]).to(device)
            tb = torch.from_numpy(temps["train"][pick]).to(device) if temps else None
            target = F.one_hot(yb, n_classes).float()
            if args.augment and kind == "logmel":
                xb, target = augment_batch(xb, target, use_mixup=not args.no_mixup,
                                           use_specaugment=not args.no_specaugment, use_shift=not args.no_shift)
            with autocast(device, args.amp):
                logits = forward(net, xb, tb).float()
            if offsets is not None:
                logits = imbalance.adjusted_logits(logits, offsets)
            loss = -(target * F.log_softmax(logits, 1) * cw).sum(1).mean()
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                if args.clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            running += loss.item()

        if not np.isfinite(running):
            raise FloatingPointError(f"training loss became {running} in epoch {epoch}")
        va_probs = predict(net, x_va, prep, temps.get("val"), device, args)
        _, y_f, p_f = metrics.file_level(va_probs, va_idx)
        va = metrics.scores(y_f, p_f.argmax(1), n_classes)
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch,
                        "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
                        "seconds": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(out / "metrics" / f"{run}_history.csv", index=False)
        if va["macro_f1"] > best_f1:
            best_f1, stale = va["macro_f1"], 0
            torch.save(net.state_dict(), ckpt)
        else:
            stale += 1
        print(f"epoch {epoch:3d}  loss {running / steps_per_epoch:.3f}  val acc {va['accuracy']:.3f}  "
              f"val macro-F1 {va['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if args.patience and stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs", flush=True)
            break

    # test once, with the checkpoint chosen on validation
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    te_probs = predict(net, x_te, prep, temps.get("test"), device, args)
    va_probs = predict(net, x_va, prep, temps.get("val"), device, args)
    report, per_cls, cm = metrics.full_report(te_probs, te_idx, classes)
    va_report, _, _ = metrics.full_report(va_probs, va_idx, classes)
    save_predictions(va_probs, va_idx, classes, out / "predictions" / f"{run}_validation.npz")
    files, labels, file_probs = save_predictions(te_probs, te_idx, classes, out / "predictions" / f"{run}_test.npz")
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    result = {"run": run, "model": args.model, "data": data_name, "params": n_params, "n_trainable": n_trainable,
              "config": {**vars(args), "data": data_name}, "best_epoch": best_epoch,
              "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
              "device": str(device), "env": {"torch": torch.__version__, "platform": platform.platform()},
              "validation": va_report, "test": report,
              "test_subsets": recordist_scores(files, labels, file_probs, te_idx, used_idx, n_classes)}

    metrics.save_json(result, out / "metrics" / f"{run}.json")
    per_cls.to_csv(out / "metrics" / f"{run}_per_class.csv", index=False)
    metrics.plot_confusion(cm, classes, out / "figures" / f"confusion_{run}.png",
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
    fig.savefig(out / "figures" / f"curves_{run}.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"test": result["test"], "test_subsets": result["test_subsets"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
