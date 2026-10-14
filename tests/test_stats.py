import json
import math
import statistics
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.eval import metrics, stats


def synthetic(n_files, n_classes, acc, seed=0):
    """Labels covering every class and probabilities whose arg-max is right with probability acc."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n_files) % n_classes
    wrong = (labels + rng.integers(1, n_classes, n_files)) % n_classes
    target = np.where(rng.random(n_files) < acc, labels, wrong)
    probs = rng.dirichlet(np.ones(n_classes), n_files)
    probs[np.arange(n_files), target] += 1
    return labels, probs / probs.sum(1, keepdims=True)


def mcnemar_case(b, c, both_right=5, both_wrong=3):
    """Labels and two prediction vectors with a chosen 2 x 2 agreement table."""
    n = b + c + both_right + both_wrong
    labels = np.zeros(n, dtype=int)
    right_a = np.r_[np.ones(b), np.zeros(c), np.ones(both_right), np.zeros(both_wrong)].astype(bool)
    right_b = np.r_[np.zeros(b), np.ones(c), np.ones(both_right), np.zeros(both_wrong)].astype(bool)
    return labels, np.where(right_a, 0, 1), np.where(right_b, 0, 2)


def test_macro_f1_matches_metrics_module():
    labels, probs = synthetic(300, 12, acc=0.6)
    pred = probs.argmax(1)
    assert stats.macro_f1(labels, pred, 12) == pytest.approx(metrics.scores(labels, pred, 12)["macro_f1"])


def test_macro_f1_on_a_subset_averages_present_species_only():
    labels, probs = synthetic(300, 12, acc=0.6)
    pred = probs.argmax(1)
    sub = labels < 5                 # 5 species present, the others can still be predicted
    expected = metrics.scores(labels[sub], pred[sub], 12, present_only=True)["macro_f1"]
    assert stats.macro_f1(labels[sub], pred[sub], 12) == pytest.approx(expected)
    assert stats.macro_f1(labels[sub], pred[sub], 12) > metrics.scores(labels[sub], pred[sub], 12)["macro_f1"]
    assert stats.macro_f1([0, 1, 1], [0, 1, 0], 2) == pytest.approx(np.mean([2 / 3, 2 / 3]))


def test_bootstrap_ci_contains_point_and_narrows_with_more_files():
    widths = []
    for n in (100, 2000):
        labels, probs = synthetic(n, 10, acc=0.7, seed=n)
        ci = stats.bootstrap_ci(labels, probs, 10, n_boot=300)
        for metric, (point, lo, hi) in ci.items():
            assert lo <= point <= hi, metric
        assert ci["accuracy"][0] == pytest.approx(np.mean(probs.argmax(1) == labels))
        widths.append(ci["macro_f1"][2] - ci["macro_f1"][1])
    assert widths[1] < widths[0] / 2


def test_paired_bootstrap_model_against_itself():
    labels, probs = synthetic(200, 8, acc=0.6)
    out = stats.paired_bootstrap(labels, probs, probs, 8, n_boot=200)
    assert out["delta_macro_f1"] == 0
    assert out["ci"] == (0.0, 0.0)
    assert out["p_value"] == 1.0


def test_paired_bootstrap_detects_a_clear_difference():
    labels, good = synthetic(400, 8, acc=0.9, seed=1)
    _, bad = synthetic(400, 8, acc=0.4, seed=2)
    out = stats.paired_bootstrap(labels, good, bad, 8, n_boot=300)
    assert out["delta_macro_f1"] > 0.3
    assert out["ci"][0] > 0
    assert out["p_value"] < 0.01
    assert stats.paired_bootstrap(labels, bad, good, 8, n_boot=300)["delta_macro_f1"] < 0


def test_mcnemar_exact_binomial_small_table():
    out = stats.mcnemar(*mcnemar_case(b=10, c=2))
    assert (out["b"], out["c"]) == (10, 2)
    # two-sided exact: 2 * P(X <= 2), X ~ Bin(12, 0.5) = 2 * (1 + 12 + 66) / 4096
    assert out["p_value"] == pytest.approx(2 * 79 / 4096)


def test_mcnemar_chi_square_large_table():
    out = stats.mcnemar(*mcnemar_case(b=30, c=10))
    assert out["statistic"] == pytest.approx((20 - 1) ** 2 / 40)
    # chi-square with 1 df: P(X > s) = erfc(sqrt(s / 2))
    assert out["p_value"] == pytest.approx(math.erfc(math.sqrt(9.025 / 2)))


def test_mcnemar_without_disagreements():
    out = stats.mcnemar(*mcnemar_case(b=0, c=0))
    assert out["p_value"] == 1.0


def test_aggregate_seeds():
    f1 = [0.5975, 0.5971, 0.5799, 0.6196, 0.5985]       # MLP species40, seeds 42-46
    runs = [{"test_f1": v, "unseen_f1": None} for v in f1]
    runs[0]["unseen_f1"] = 0.5
    runs.append({"test_f1": float("nan")})
    out = stats.aggregate_seeds(runs)
    assert out["test_f1"]["mean"] == pytest.approx(statistics.mean(f1))
    assert out["test_f1"]["sd"] == pytest.approx(statistics.stdev(f1))
    assert out["test_f1"]["n"] == 5
    assert out["unseen_f1"]["mean"] == 0.5 and out["unseen_f1"]["n"] == 1
    assert math.isnan(out["unseen_f1"]["sd"])


def save_npz(root, run, split, files, labels, probs, classes):
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez(root / "predictions" / f"{run}_{split}.npz", files=np.array(files, dtype=str),   # as src/train.py
             labels=np.asarray(labels, dtype=np.int64), probs=np.asarray(probs, dtype=np.float32),
             classes=np.array(classes))


def test_load_predictions_round_trip(tmp_path):
    labels, probs = synthetic(10, 3, acc=0.5)
    save_npz(tmp_path, "cnn_x_s1", "test", [f"f{i}.wav" for i in range(10)], labels, probs, ["a", "b", "c"])
    p = stats.load_predictions("cnn_x_s1", "test", tmp_path)
    assert p["files"][3] == "f3.wav" and p["classes"].tolist() == ["a", "b", "c"]
    assert p["labels"].dtype == np.int64 and p["probs"].shape == (10, 3)


def run_cli(*args):
    res = subprocess.run([sys.executable, *args], cwd=C.ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    return res.stdout


def result_json(run, model, data, f1, seconds, unseen=None):
    block = {"accuracy": f1 + 0.01, "macro_precision": f1, "macro_recall": f1, "macro_f1": f1, "weighted_f1": f1}
    r = {"model": model, "data": data, "params": 2_000_000, "config": {}, "best_epoch": 3,
         "train_seconds": seconds, "validation": {"file": block, "chunk": block},
         "test": {"file": block, "chunk": block}}
    if unseen is not None:
        r.update(run=run, n_trainable=2_000_000,
                 test_subsets={"recordist_seen": {**block, "n_files": 50},
                               "recordist_unseen": {**block, "macro_f1": unseen, "n_files": 20}})
    return r


def test_compare_end_to_end(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    files = {
        "cnn_species40_s42": result_json("cnn_species40_s42", "cnn", "species40", 0.70, 100, unseen=0.5),
        "cnn_species40_s43": result_json("cnn_species40_s43", "cnn", "species40", 0.74, 120, unseen=0.6),
        "effnet_species40_lt10_effective_s42": result_json(
            "effnet_species40_lt10_effective_s42", "effnet", "species40", 0.8, 300, unseen=0.7),
        "mlp_species40": result_json(None, "mlp", "species40", 0.6, 50),           # legacy: no seed, no new keys
        "cnn_pilot": result_json(None, "cnn", "pilot", 0.9, 10),
        "cnn_species40b_s42": result_json(None, "cnn", "species40b", 0.1, 10),    # other dataset, same prefix
        "error_cmp_species40": {"name": "cmp", "runs": []},                       # not a training run
    }
    for name, r in files.items():
        (m / f"{name}.json").write_text(json.dumps(r))

    out = run_cli("-m", "src.eval.compare", "--data", "species40", "--results", str(tmp_path), "--markdown")
    assert "EfficientNetV2-S (ImageNet)" in out
    table = pd.read_csv(m / "comparison_species40.csv")
    assert table.model.tolist() == ["MLP (handcrafted)", "CNN (log-mel)", "EfficientNetV2-S (ImageNet)"]
    cnn = table.iloc[1]
    assert cnn.seeds == 2 and cnn.variant == "-"
    assert cnn["test macro-F1 mean"] == pytest.approx(0.72)
    assert cnn["test macro-F1 sd"] == pytest.approx(statistics.stdev([0.70, 0.74]), abs=1e-4)
    assert cnn["recordist-unseen macro-F1 mean"] == pytest.approx(0.55)
    assert cnn["train time (s) mean"] == pytest.approx(110)
    assert table.iloc[2].variant == "lt10_effective"
    assert pd.isna(table.iloc[0]["recordist-unseen macro-F1 mean"])
    assert (tmp_path / "comparison_species40.md").exists()

    run_cli("-m", "src.eval.compare", "--data", "pilot", "--results", str(tmp_path))
    pilot = pd.read_csv(m / "comparison_pilot.csv")
    assert len(pilot) == 1 and pilot.iloc[0]["test macro-F1 mean"] == pytest.approx(0.9)


def one_hot_probs(pred, n_classes, confidence):
    probs = np.full((len(pred), n_classes), (1 - confidence) / (n_classes - 1))
    probs[np.arange(len(pred)), pred] = confidence
    return probs


def test_error_analysis_end_to_end(tmp_path):
    meta = pd.read_csv(C.META_CSV, usecols=["species_name", "group"]).drop_duplicates("species_name")
    orth = sorted(meta[meta.group == "Orthoptera"].species_name)[:3]
    cic = sorted(meta[meta.group == "Cicadidae"].species_name)[:3]
    classes = orth + cic
    labels = np.repeat(np.arange(6), 4)
    files = [f"file{i:02d}.wav" for i in range(24)]
    pred_a, pred_b = labels.copy(), labels.copy()
    pred_a[0], pred_a[4] = 1, 3          # O1 -> O2, and O2 -> C1 (crosses the group boundary)
    pred_b[0], pred_b[8] = 1, 0          # O1 -> O2 again, and O3 -> O1
    probs_a, probs_b = one_hot_probs(pred_a, 6, 0.7), one_hot_probs(pred_b, 6, 0.9)
    save_npz(tmp_path, "a_syn_s1", "test", files, labels, probs_a, classes)
    # second run lists the files in reverse order and has one file the first run lacks
    save_npz(tmp_path, "b_syn_s1", "test", files[::-1] + ["extra.wav"], np.r_[labels[::-1], 0],
             np.vstack([probs_b[::-1], one_hot_probs([0], 6, 0.9)]), classes)

    run_cli("analysis/error_analysis.py", "--runs", "a_syn_s1", "b_syn_s1", "--name", "syn",
            "--results", str(tmp_path))
    rep = json.loads((tmp_path / "metrics" / "error_syn.json").read_text())
    assert "model" not in rep                                   # compare.py must never pick it up
    assert rep["n_files"] == 24 and rep["files_dropped_to_align"] == {"a_syn_s1": 0, "b_syn_s1": 1}
    assert rep["agreement"] == {"all_right": 21, "all_wrong": 1, "oracle_accuracy": pytest.approx(23 / 24)}
    assert rep["all_wrong_files"] == [{"file": "file00.wav", "true": classes[0],
                                       "a_syn_s1": classes[1], "b_syn_s1": classes[1]}]
    assert rep["groups"]["a_syn_s1"]["cross_group_errors"] == 1
    assert rep["groups"]["b_syn_s1"]["cross_group_errors"] == 0
    assert rep["groups"]["a_syn_s1"]["Cicadidae"] == {"n_files": 12, "accuracy": 1.0}
    assert [(p["a"], p["b"]) for p in rep["shared_confusions"]] == [(classes[0], classes[1])]
    assert rep["top_confusions"]["a_syn_s1"][0]["count"] == 1
    assert rep["scores"]["a_syn_s1"]["accuracy"] == pytest.approx(22 / 24)
    ensemble = ((probs_a + probs_b) / 2).argmax(1)
    expected = metrics.scores(labels, ensemble, 6)
    assert rep["ensemble_mean_probs"]["macro_f1"] == pytest.approx(expected["macro_f1"])

    per_class = pd.read_csv(tmp_path / "metrics" / "error_syn_per_class_f1.csv")
    assert list(per_class.columns) == ["species", "group", "support", "a_syn_s1", "b_syn_s1", "mean_f1"]
    assert per_class.mean_f1.is_monotonic_increasing and len(per_class) == 6
    assert (tmp_path / "figures" / "error_syn_per_class_f1.png").stat().st_size > 0
