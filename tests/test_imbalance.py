import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src import config as C
from src.utils.imbalance import (adjusted_logits, balanced_sampler, class_weights,
                                 logit_adjustment, long_tail_subset)


def make_index(files_per_class=(24, 20, 22, 30, 21, 25), seed=0):
    """Synthetic chunk index: every recording has 1-4 chunk rows, rows shuffled."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, n_files in enumerate(files_per_class):
        for f in range(n_files):
            for chunk in range(rng.integers(1, 5)):
                rows.append({"file_name": f"sp{label}_rec{f}.wav", "species_name": f"sp{label}",
                             "label": label, "chunk": chunk})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def files_per_class(index):
    return index.groupby("label").file_name.nunique()


def files_per_species(index):
    return index.groupby("species_name").file_name.nunique()


def test_long_tail_keeps_whole_files():
    index = make_index()
    kept = index.iloc[long_tail_subset(index, ratio=10, seed=42)]
    chunks_all = index.file_name.value_counts()
    chunks_kept = kept.file_name.value_counts()
    assert (chunks_kept == chunks_all.loc[chunks_kept.index]).all()


@pytest.mark.parametrize("ratio", [5, 10, 100])
def test_long_tail_exponential_profile(ratio):
    index = make_index()
    counts = files_per_class(index.iloc[long_tail_subset(index, ratio, seed=7)])
    n_max, n_cls = files_per_class(index).min(), index.label.nunique()
    expected = [max(1, round(n_max * ratio ** (-r / (n_cls - 1)))) for r in range(n_cls)]
    observed = sorted(counts.values, reverse=True)
    assert len(counts) == n_cls                       # every class keeps at least one file
    assert observed == expected
    assert observed[0] == n_max
    # head / tail = ratio, unless the tail hits the 1-file floor
    assert observed[0] / observed[-1] == pytest.approx(min(ratio, n_max), rel=0.25)


def test_long_tail_deterministic_per_seed():
    index = make_index()
    a = long_tail_subset(index, 10, seed=1)
    assert np.array_equal(a, long_tail_subset(index, 10, seed=1))
    assert not np.array_equal(a, long_tail_subset(index, 10, seed=2))


def test_long_tail_ignores_row_order():
    # the analysis script must rebuild exactly the subset train.py trained on
    index = make_index()
    shuffled = index.sample(frac=1, random_state=5).reset_index(drop=True)
    a = index.iloc[long_tail_subset(index, 10, seed=42)].file_name
    b = shuffled.iloc[long_tail_subset(shuffled, 10, seed=42)].file_name
    assert set(a) == set(b)


def test_long_tail_ratio_one_keeps_everything():
    index = make_index()
    assert np.array_equal(long_tail_subset(index, 1, seed=3), np.arange(len(index)))
    with pytest.raises(ValueError):
        long_tail_subset(index, 0.5)


def test_logit_adjustment_is_log_prior():
    counts = np.array([10, 30, 60])
    offsets = logit_adjustment(counts)
    assert offsets.dtype == torch.float32                  # MPS rejects float64
    assert torch.allclose(offsets, torch.log(torch.tensor([0.1, 0.3, 0.6])))
    assert torch.allclose(logit_adjustment(counts / counts.sum()), offsets)   # prior works too
    assert torch.allclose(logit_adjustment(counts, tau=2.0), 2 * offsets)
    logits = torch.randn(4, 3)
    assert torch.allclose(adjusted_logits(logits, offsets), logits + offsets)


def test_class_weights_unchanged():
    labels = np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 2])     # counts 1, 2, 7 and an empty class
    assert torch.equal(class_weights(labels, 4, "none"), torch.ones(4))
    assert torch.allclose(class_weights(labels, 4, "inverse"),
                          torch.tensor([1.5135136, 0.7567568, 0.2162162, 1.5135136]))
    assert torch.allclose(class_weights(labels, 4, "effective"),
                          torch.tensor([1.5131246, 0.7569408, 0.2168100, 1.5131246]))
    assert len(balanced_sampler(labels, 4)) == len(labels)


def run_analysis(monkeypatch, tmp_path, *argv):
    """Run analysis/long_tail_analysis.py as a script with results/ and data/ redirected to tmp_path."""
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(C, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(sys, "argv", ["long_tail_analysis.py", *argv])
    runpy.run_path(str(C.ROOT / "analysis" / "long_tail_analysis.py"), run_name="__main__")


def test_long_tail_analysis_end_to_end(tmp_path, monkeypatch, capsys):
    data_dir, metrics_dir = tmp_path / "processed" / "syn", tmp_path / "results" / "metrics"
    data_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    index = make_index(files_per_class=(30, 25, 20, 15, 10, 5))
    index.to_csv(data_dir / "Train_index.csv", index=False)
    f1 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]                       # falls with training files
    per_class = pd.DataFrame({"species": [f"sp{i}" for i in range(6)], "precision": f1,
                              "recall": f1, "f1": f1, "support": 10})
    per_class.to_csv(metrics_dir / "crnn_syn_s42_per_class.csv", index=False)       # no JSON, like old runs
    per_class.to_csv(metrics_dir / "crnn_syn_lt5_s43_per_class.csv", index=False)
    config = {"long_tail": 5, "lt_seed": 42, "seed": 43}       # subset drawn with lt_seed, not seed
    (metrics_dir / "crnn_syn_lt5_s43.json").write_text(json.dumps({"config": config}))

    run_analysis(monkeypatch, tmp_path, "--data", "syn", "--name", "test",
                 "--runs", "crnn_syn_s42", "crnn_syn_lt5_s43", "missing_run")

    assert "skipping missing_run" in capsys.readouterr().out
    assert (tmp_path / "results" / "figures" / "longtail_test.png").exists()
    table = pd.read_csv(metrics_dir / "longtail_test.csv").set_index("run")
    assert list(table.index) == ["crnn_syn_s42", "crnn_syn_lt5_s43"]
    full = table.loc["crnn_syn_s42"]
    assert full.train_files == 105 and full.spearman_rho == pytest.approx(1.0)
    assert full.head_f1 == pytest.approx(0.85) and full.tail_f1 == pytest.approx(0.45)
    assert full.head_files == "25-30" and full.tail_files == "5-10"
    lt = table.loc["crnn_syn_lt5_s43"]
    kept = index.iloc[long_tail_subset(index, 5, seed=42)]
    assert lt.long_tail == 5 and lt.train_files == kept.file_name.nunique()


def test_long_tail_analysis_counts_the_run_subset():
    train_files = runpy.run_path(str(C.ROOT / "analysis" / "long_tail_analysis.py"))["train_files"]
    index = make_index()
    expected = files_per_species(index.iloc[long_tail_subset(index, 10, seed=42)])
    assert not expected.equals(files_per_species(index.iloc[long_tail_subset(index, 10, seed=43)]))
    assert train_files(index, {"long_tail": 10, "lt_seed": 42, "seed": 43}).equals(expected)
    assert train_files(index, {"long_tail": 10, "seed": 42}).equals(expected)     # runs without lt_seed
    assert train_files(index, {}).equals(files_per_species(index))                # full training set


def test_spearman_without_scipy_matches_scipy(monkeypatch):
    spearman = runpy.run_path(str(C.ROOT / "analysis" / "long_tail_analysis.py"))["spearman"]
    rng = np.random.default_rng(0)
    x, y = pd.Series(rng.integers(1, 6, 40)), pd.Series(rng.random(40).round(1))   # with ties
    with_scipy = spearman(x, y)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)                   # import now fails
    assert spearman(x, y) == pytest.approx(with_scipy)
    assert np.isnan(spearman(pd.Series([30] * 5), y[:5]))                   # equal counts


def test_long_tail_analysis_missing_index(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="Train_index.csv not found"):
        run_analysis(monkeypatch, tmp_path, "--data", "nothing", "--name", "x", "--runs", "r")
