import pytest
import torch
import torch.nn.functional as F

from src import config as C
from src.models.mlp import MLP

PILOT_CKPT = C.ROOT / "outputs/mlp_pilot.pt"


def test_forward_is_head_of_embed():
    torch.manual_seed(0)
    net = MLP(146, 7).eval()
    x = torch.randn(5, 146)
    z = net.embed(x)
    assert z.shape == (5, net.embed_dim) and net.embed_dim == 256
    assert torch.equal(net(x), net.net[-1](z))   # dropout is the identity in eval mode
    assert torch.equal(net(x), net.net(x))       # same result as before embed() existed


def test_forward_unchanged_in_train_mode():
    net = MLP(146, 7).train()
    x = torch.randn(8, 146)
    torch.manual_seed(1)
    a = net(x)
    torch.manual_seed(1)
    b = net.net(x)
    assert torch.equal(a, b)


def test_embed_stops_before_final_dropout():
    # in train mode the head must be exactly dropout + linear applied to embed(x);
    # an embedding taken after the final dropout would consume the RNG differently
    net = MLP(146, 7, dropout=0.3).train()
    x = torch.randn(8, 146)
    torch.manual_seed(2)
    a = net(x)
    torch.manual_seed(2)
    b = net.net[-1](F.dropout(net.embed(x), 0.3, training=True))
    assert torch.equal(a, b)


def test_state_dict_keys_unchanged():
    keys = list(MLP(146, 12).state_dict())
    bn = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    assert keys == (["net.0.weight", "net.0.bias"] + [f"net.1.{k}" for k in bn]
                    + ["net.4.weight", "net.4.bias"] + [f"net.5.{k}" for k in bn]
                    + ["net.8.weight", "net.8.bias"])


@pytest.mark.skipif(not PILOT_CKPT.exists(), reason="pilot checkpoint not present")
def test_pilot_checkpoint_loads_strict():
    state = torch.load(PILOT_CKPT, map_location="cpu", weights_only=True)
    net = MLP(state["net.0.weight"].shape[1], state["net.8.weight"].shape[0])
    net.load_state_dict(state, strict=True)
    net.eval()
    x = torch.randn(3, net.net[0].in_features)
    assert net(x).shape == (3, state["net.8.weight"].shape[0])
