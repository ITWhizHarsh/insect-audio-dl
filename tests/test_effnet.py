"""CNN family: EfficientNetV2-S wrapper and the embed() contract shared with train.py.

The ImageNet weights come from the local torchvision cache, so these tests run
offline. Everything stays on the CPU with a batch of two spectrograms.
"""
import pytest
import torch
import torch.nn.functional as F

from src import config as C
from src.models.cnn import LogMelCNN
from src.models.effnet import EfficientNetV2S

N_FRAMES = 431


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return EfficientNetV2S(n_classes=7).eval()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 1, C.N_MELS, N_FRAMES)


def test_mono_stem_equals_rgb_stem_on_repeated_input(net, x):
    from torchvision.models import EfficientNet_V2_S_Weights
    w_rgb = EfficientNet_V2_S_Weights.IMAGENET1K_V1.get_state_dict(progress=False)["features.0.0.weight"]
    stem = net.features[0][0]
    rgb = F.conv2d(x.repeat(1, 3, 1, 1), w_rgb, stride=2, padding=1)   # torchvision's 3x3 stride-2 stem
    assert stem.in_channels == 1
    assert torch.allclose(stem(x), rgb, atol=1e-5)


def test_logits_shape_and_size(net, x):
    with torch.no_grad():
        assert net(x).shape == (2, 7)
    n_params = sum(p.numel() for p in net.parameters())
    assert 19e6 < n_params < 22e6


def test_forward_is_head_of_embed(net, x):
    with torch.no_grad():
        e = net.embed(x)
        assert e.shape == (2, net.embed_dim) == (2, 1280)
        assert torch.equal(net(x), net.fc(net.dropout(e)))


def test_untrained_variant_needs_no_download(monkeypatch, x):
    def no_network(*args, **kwargs):
        raise AssertionError("tried to fetch weights")
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", no_network)
    net = EfficientNetV2S(n_classes=3, pretrained=False, stochastic_depth=0.0).eval()
    with torch.no_grad():
        assert net(x).shape == (2, 3)
    depth_probs = [m.p for m in net.modules() if type(m).__name__ == "StochasticDepth"]
    assert depth_probs and max(depth_probs) == 0.0


def test_stochastic_depth_is_configurable(net):
    depth_probs = [m.p for m in net.modules() if type(m).__name__ == "StochasticDepth"]
    assert 0.0 < max(depth_probs) < 0.2       # torchvision ramps it up linearly with depth


def test_cnn_forward_is_head_of_embed(x):
    torch.manual_seed(0)
    cnn = LogMelCNN(n_classes=5).eval()
    with torch.no_grad():
        h = cnn.features(x).mean(dim=2)        # the forward pass before embed() existed
        before = cnn.fc(cnn.dropout(torch.cat([h.mean(-1), h.amax(-1)], dim=1)))
        e = cnn.embed(x)
        assert e.shape == (2, cnn.embed_dim) == (2, 512)
        assert torch.equal(cnn(x), before)


@pytest.mark.skipif(not (C.ROOT / "outputs/cnn_pilot.pt").exists(), reason="pilot checkpoint not present")
def test_cnn_pilot_checkpoint_still_loads():
    state = torch.load(C.ROOT / "outputs/cnn_pilot.pt", map_location="cpu", weights_only=True)
    LogMelCNN(n_classes=state["fc.weight"].shape[0]).load_state_dict(state, strict=True)
