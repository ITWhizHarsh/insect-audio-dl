import pytest
import torch

from src import config as C
from src.models.crnn import CRNN

N_CLASSES = 5


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_output_and_embedding_shapes():
    torch.manual_seed(0)
    net = CRNN(N_CLASSES).eval()
    x = torch.randn(2, 1, 128, 431)
    with torch.no_grad():
        assert net(x).shape == (2, N_CLASSES)
        assert net.embed(x).shape == (2, net.embed_dim)
    assert net.embed_dim == 256


def test_forward_is_head_of_embedding():
    torch.manual_seed(0)
    net = CRNN(N_CLASSES).eval()
    x = torch.randn(2, 1, 128, 431)
    with torch.no_grad():
        assert torch.equal(net(x), net.fc(net.embed(x)))


@pytest.mark.parametrize("rnn", ["lstm", "gru"])
def test_forward_backward(rnn):
    torch.manual_seed(0)
    net = CRNN(N_CLASSES, rnn=rnn).train()
    loss = net(torch.randn(2, 1, 128, 431)).logsumexp(1).mean()
    loss.backward()
    grads = [p.grad for p in net.rnn.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_gru_saves_a_quarter_of_the_recurrent_weights():
    lstm, gru = CRNN(N_CLASSES, rnn="lstm"), CRNN(N_CLASSES, rnn="gru")
    assert count(gru) < count(lstm)
    assert count(lstm) - count(gru) == count(lstm.rnn) // 4     # 3 gates instead of 4
    with pytest.raises(KeyError):
        CRNN(N_CLASSES, rnn="transformer")


def test_pilot_checkpoint_still_loads():
    ckpt = C.ROOT / "outputs" / "crnn_pilot.pt"
    if not ckpt.exists():
        pytest.skip("no pilot checkpoint")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    CRNN(state["fc.1.weight"].shape[0]).load_state_dict(state, strict=True)
