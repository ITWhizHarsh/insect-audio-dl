import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.conditioning import TemperatureConditioned, temperature_inputs


class TinyBackbone(nn.Module):
    """Stands in for any of our models: exposes embed(x) and embed_dim."""
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, self.embed_dim)

    def embed(self, x):
        return torch.relu(self.lin(x))


def make(seed=0):
    torch.manual_seed(seed)
    return TemperatureConditioned(TinyBackbone(), n_classes=4, hidden=16)


def random_temps(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.randn(n, 1, generator=g) * 3, (torch.rand(n, 1, generator=g) > 0.5).float()], 1)


def test_zero_init_equals_plain_head_for_any_temperature():
    net = make().eval()
    x = torch.randn(6, 5)
    plain = net.fc(net.backbone.embed(x))
    for seed in range(3):
        assert torch.equal(net(x, random_temps(6, seed)), plain)
    assert torch.equal(net.embed(x, random_temps(6)), net.backbone.embed(x))
    assert net.embed_dim == 8


def test_gradients_reach_film_and_temperature_matters_after_training():
    net = make()
    x, y, temp = torch.randn(16, 5), torch.randint(0, 4, (16,)), random_temps(16)
    opt = torch.optim.SGD(net.parameters(), lr=0.5)

    nn.functional.cross_entropy(net(x, temp), y).backward()
    last = net.film[-1]
    assert last.weight.grad.abs().sum() > 0 and last.bias.grad.abs().sum() > 0
    opt.step()
    opt.zero_grad()
    # once the last layer is non-zero the gradient flows back to the first FiLM layer too
    nn.functional.cross_entropy(net(x, temp), y).backward()
    assert net.film[0].weight.grad.abs().sum() > 0

    net.eval()
    assert not torch.allclose(net(x, temp), net(x, random_temps(16, seed=1)))


def test_temperature_inputs_standardise_and_flag_missing():
    temps = pd.Series([24.0, np.nan, 30.0, 60.0, None])
    out = temperature_inputs(temps, mean=24.0, std=2.0)
    assert out.dtype == np.float32 and out.shape == (5, 2)
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 3.0, 0.0, 0.0])
    np.testing.assert_array_equal(out[:, 1], [0, 1, 0, 1, 1])   # 60 C is outside the audit's valid range
    assert not np.isnan(out).any()


def test_temperature_inputs_plain_array():
    out = temperature_inputs(np.array([10.0, 20.0]), mean=15.0, std=5.0)
    np.testing.assert_allclose(out, [[-1, 0], [1, 0]])


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_real_backbones_at_contract_shapes(name):
    from src.models.cnn import LogMelCNN
    from src.models.mlp import MLP
    torch.manual_seed(0)
    backbone, x = {"mlp": (MLP(146, 5), torch.randn(2, 146)),
                   "cnn": (LogMelCNN(5), torch.randn(2, 1, 128, 431))}[name]
    net = TemperatureConditioned(backbone, n_classes=5).eval()
    temp = torch.tensor(temperature_inputs([20.0, np.nan], 24.0, 4.5))
    with torch.no_grad():
        out, plain = net(x, temp), net.fc(backbone.embed(x))
    assert out.shape == (2, 5) and net.embed_dim == backbone.embed_dim
    assert torch.equal(out, plain)
