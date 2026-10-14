"""Vectorised augmentation keeps the semantics of the per-example loops it replaced.

Small random tensors on the CPU; a fixed seed makes every run identical.
"""
import pytest
import torch

from src.features.augment import augment_batch, mixup, spec_augment, time_shift


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def ramp(b=8, f=6, t=50):
    """Row r holds 1000 * r + (0, 1, ..., t-1), so every value tells where it came from."""
    return (torch.arange(t) + 1000 * torch.arange(b * f)[:, None]).float().reshape(b, 1, f, t)


def labels(b=8, n=4):
    return torch.eye(n)[torch.randint(0, n, (b,))]


def band_widths(mask):
    """Width of the single True run in each row of a (B, n) mask; fails if a row has two runs."""
    widths = []
    for row in mask:
        idx = row.nonzero().flatten()
        if len(idx):
            assert idx[-1] - idx[0] + 1 == len(idx)
        widths.append(len(idx))
    return widths


def test_shapes_preserved():
    x, y = torch.randn(3, 1, 128, 431), labels(3)
    assert time_shift(x).shape == x.shape
    assert spec_augment(x).shape == x.shape
    xa, ya = augment_batch(x, y)
    assert xa.shape == x.shape and ya.shape == y.shape


def test_time_shift_is_a_circular_roll_per_example():
    x = ramp()
    T = x.shape[-1]
    out = time_shift(x, max_frac=0.2)
    assert torch.equal(out.sort(-1).values, x.sort(-1).values)   # only a reordering along time
    shifts = []
    for i in range(len(x)):
        s = -(int(out[i, 0, 0, 0]) % 1000) % T                   # frame 0 now holds frame -s
        assert torch.equal(out[i], torch.roll(x[i], s, dims=-1))  # same roll on every mel row
        assert min(s, T - s) <= 0.2 * T
        shifts.append(s)
    assert any(0 < s < T / 2 for s in shifts) and any(s > T / 2 for s in shifts)   # own shift, both directions


def test_time_shift_zero_is_identity():
    x = ramp()
    assert torch.equal(time_shift(x, max_frac=0), x)


def test_frequency_masks_are_whole_bands_within_width():
    x = torch.randn(64, 1, 32, 40)
    out = spec_augment(x, freq_masks=1, freq_width=6, time_masks=0)
    changed = (out != x)[:, 0]
    rows = changed.all(-1)                                        # (B, F) masked mel bins
    assert torch.equal(changed.any(-1), rows)                     # bins are masked across all frames
    widths = band_widths(rows)
    assert max(widths) <= 6 and len(set(widths)) > 1
    starts = rows.float().argmax(-1)[rows.any(-1)]
    assert starts.max() > 32 // 2                                 # bands land anywhere, not only at the bottom
    assert torch.all(out[out != x] == x.mean())


def test_time_masks_are_whole_bands_within_width():
    x = torch.randn(64, 1, 32, 40)
    out = spec_augment(x, freq_masks=0, time_masks=1, time_width=9)
    changed = (out != x)[:, 0]
    cols = changed.all(-2)                                        # (B, T) masked frames
    assert torch.equal(changed.any(-2), cols)
    widths = band_widths(cols)
    assert max(widths) <= 9 and len(set(widths)) > 1
    assert torch.all(out[out != x] == x.mean())


def test_default_masks_bounds_and_fill():
    x = torch.randn(4, 1, 128, 431)
    out = spec_augment(x)
    changed = (out != x)[:, 0]
    rows, cols = changed.all(-1), changed.all(-2)
    assert torch.equal(changed, rows[:, :, None] | cols[:, None, :])
    assert (rows.sum(-1) <= 2 * 16).all() and (cols.sum(-1) <= 2 * 40).all()
    assert torch.all(out[out != x] == x.mean())


def test_masks_differ_across_examples():
    x = torch.randn(1, 1, 64, 200).repeat(8, 1, 1, 1)             # identical examples
    changed = spec_augment(x) != x
    assert any(not torch.equal(changed[0], changed[i]) for i in range(1, 8))


def test_augment_batch_all_off_is_identity():
    x, y = torch.randn(8, 1, 16, 30), labels()
    xa, ya = augment_batch(x, y, use_mixup=False, use_specaugment=False, use_shift=False)
    assert torch.equal(xa, x) and torch.equal(ya, y)


def test_augment_batch_flags_turn_parts_on_one_at_a_time():
    x, y = torch.randn(8, 1, 16, 30), labels()

    xa, ya = augment_batch(x, y, use_mixup=False, use_specaugment=False)
    assert not torch.equal(xa, x) and torch.equal(xa.sort(-1).values, x.sort(-1).values)
    assert torch.equal(ya, y)

    xa, ya = augment_batch(x, y, use_mixup=False, use_shift=False)
    assert (xa != x).any() and torch.all(xa[xa != x] == x.mean())
    assert torch.equal(ya, y)

    xa, ya = augment_batch(x, y, use_specaugment=False, use_shift=False)
    assert not torch.equal(ya, y)
    assert torch.allclose(ya.sum(1), torch.ones(8))


def test_mixup_labels_are_distributions_dominated_by_the_first_example():
    x, y = torch.randn(32, 1, 8, 10), labels(32, 5)
    xm, ym = mixup(x, y)
    assert xm.shape == x.shape
    assert torch.allclose(ym.sum(1), torch.ones(32))
    assert torch.all((ym * y).sum(1) >= 0.5 - 1e-6)               # own label keeps at least half
