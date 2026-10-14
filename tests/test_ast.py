"""Tests for the Audio Spectrogram Transformer (src/models/ast.py). CPU only.

The first test checks our compact implementation against HuggingFace's
ASTModel with the same AudioSet weights; the others use random weights so
they run without the 350 MB download.
"""
import gc

import huggingface_hub
import pytest
import torch
import torch.nn.functional as F

from src.models.ast import AST, HF_REPO, adapt_pos_embed

# AST(...) downloads only the weights; HF's reference ASTModel also needs the config
WEIGHTS_CACHED = all(isinstance(huggingface_hub.try_to_load_from_cache(HF_REPO, f), str)
                     for f in ("model.safetensors", "config.json"))


@pytest.mark.skipif(not WEIGHTS_CACHED, reason="AudioSet AST weights / config not in the HuggingFace cache")
def test_matches_huggingface_ast():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    x = torch.randn(2, 1, 128, 1024)          # pretrained size: 128 mel bins x 1024 frames

    net = AST(10, n_frames=1024).eval()
    with torch.no_grad():
        ours = net.embed(x)
    del net
    gc.collect()

    hf = transformers.ASTModel.from_pretrained(HF_REPO, local_files_only=True).eval()
    with torch.no_grad():
        ref = hf(x[:, 0].transpose(1, 2) * 0.5).pooler_output     # HF input is (B, frames, mels)
    del hf
    gc.collect()
    assert (ours - ref).abs().max() < 1e-4


@pytest.mark.parametrize("stride, grid", [(10, (12, 42)), (16, (8, 26))])
def test_shapes_and_head(stride, grid, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("pretrained=False must not load weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)

    torch.manual_seed(0)
    net = AST(5, stride=stride, pretrained=False).eval()
    x = torch.randn(2, 1, 128, 431)
    with torch.no_grad():
        tokens = net.tokens(x)
        emb = net.embed(x)
        logits = net(x)
    assert net.grid == grid
    assert tokens.shape == (2, 2 + grid[0] * grid[1], 768)
    assert net.embed_dim == 768 and emb.shape == (2, 768)
    assert logits.shape == (2, 5)
    assert torch.equal(logits, net.fc(emb))


def test_pos_embed_adaptation():
    # each pretrained position embedding stores its own (freq, time) grid index
    f, t = torch.meshgrid(torch.arange(12.0), torch.arange(101.0), indexing="ij")
    pos = torch.stack([f, t], -1).reshape(1, 12 * 101, 2)

    assert torch.equal(adapt_pos_embed(pos, 128, 1024, 10), pos)      # pretrained size: unchanged
    ours = adapt_pos_embed(pos, 128, 431, 10).reshape(12, 42, 2)
    assert torch.equal(ours[..., 0], f[:, :42])
    assert torch.equal(ours[0, :, 1], torch.arange(29.0, 71.0))       # centre 42 of the 101 time steps
    s16 = adapt_pos_embed(pos, 128, 431, 16).reshape(8, 26, 2)        # resized from the centre slice
    assert s16[..., 1].min() >= 29 and s16[..., 1].max() <= 70
    assert adapt_pos_embed(pos, 128, 2048, 10).shape == (1, 12 * 204, 2)   # longer input: stretched


def test_patchout_only_in_training():
    torch.manual_seed(0)
    net = AST(5, stride=16, patchout=0.5, pretrained=False)
    x = torch.randn(2, 1, 128, 431)
    with torch.no_grad():
        kept = net.train().tokens(x)
        full = net.eval().tokens(x)
    assert kept.shape == (2, 2 + 104, 768) and full.shape == (2, 2 + 208, 768)
    assert torch.equal(kept[:, :2], full[:, :2])                       # cls / distillation tokens always kept
    for b in range(2):     # each kept token is a distinct token of the same example, position included
        match = (kept[b, 2:, None] == full[b, None, 2:]).all(-1)       # (104, 208)
        assert match.sum(1).eq(1).all() and match.any(0).sum() == 104


@pytest.mark.parametrize("k", [0, 10])
def test_freeze_and_backward_with_patchout(k):
    torch.manual_seed(0)
    net = AST(5, stride=16, patchout=0.5, freeze_blocks=k, pretrained=False).train()
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    expected = {n for n, _ in net.named_parameters() if k and (
        n.split(".")[0] in ("patch", "cls_token", "dist_token", "pos_embed")
        or n.startswith("blocks.") and int(n.split(".")[1]) < k)}
    assert frozen == expected

    F.cross_entropy(net(torch.randn(2, 1, 128, 431)), torch.tensor([0, 3])).backward()
    for n, p in net.named_parameters():
        assert (p.grad is None) == (n in frozen), n
    assert net.blocks[k].qkv.weight.grad.abs().sum() > 0
    assert net.fc[1].weight.grad.abs().sum() > 0


def test_checkpoint_gives_same_gradients():
    torch.manual_seed(0)
    net = AST(5, stride=16, freeze_blocks=10, pretrained=False).train()
    x, y = torch.randn(2, 1, 128, 431), torch.tensor([0, 3])
    grads = []
    for use_checkpoint in (False, True):
        net.checkpoint = use_checkpoint
        net.zero_grad()
        F.cross_entropy(net(x), y).backward()
        grads.append(net.blocks[10].fc1.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
