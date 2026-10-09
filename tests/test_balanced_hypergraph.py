import pytest
import torch

from lgmrec.models.balanced_hypergraph import BalancedHGNNLayer
from lgmrec.models.lgmrec import HGNNLayer
from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


@pytest.mark.parametrize('layers', [1, 2])
@pytest.mark.parametrize('power', [0., .5, 1.])
def test_dense_formula_and_gradients(layers, power):
    incidence = torch.tensor([[.8, .2, 0.], [.9, .1, 0.], [.7, .3, 0.]], requires_grad=True)
    users = torch.tensor([[.4, .6, 0.]])
    embeds = torch.tensor([[1., 2.], [3., 1.], [2., 4.]], requires_grad=True)
    layer = BalancedHGNNLayer(layers, power)
    actual_u, actual_i = layer(incidence, users, embeds)
    expected = embeds
    for _ in range(layers):
        latent = incidence.T @ expected
        if power:
            latent = latent / incidence.sum(0).clamp_min(torch.finfo(incidence.dtype).eps).pow(power)[:, None]
        expected = incidence @ latent
    torch.testing.assert_close(actual_i, expected)
    torch.testing.assert_close(actual_u, users @ latent)
    (actual_u.sum() + actual_i.sum()).backward()
    assert torch.isfinite(incidence.grad).all() and torch.isfinite(embeds.grad).all()
    assert not layer.state_dict() and not list(layer.parameters())
    if power == 0:
        baseline = HGNNLayer(layers)(incidence, users, embeds)
        assert torch.equal(actual_u, baseline[0]) and torch.equal(actual_i, baseline[1])


@pytest.mark.parametrize('power', [.5, 1.])
def test_empty_edges(power):
    incidence = torch.zeros(4, 3, requires_grad=True)
    result = BalancedHGNNLayer(2, power)(incidence, torch.ones(2, 3), torch.ones(4, 8))
    assert all(torch.isfinite(x).all() and x.count_nonzero() == 0 for x in result)
    sum(x.sum() for x in result).backward()
    assert torch.isfinite(incidence.grad).all()


def test_mean_pooling_is_invariant_to_replicating_members():
    h = torch.tensor([[.9, .1], [.8, .2], [.4, .6]], dtype=torch.double, requires_grad=True)
    u = torch.tensor([[.5, .5]], dtype=torch.double)
    e = torch.tensor([[1., 0.], [0., 1.], [2., 1.]], dtype=torch.double, requires_grad=True)
    layer = BalancedHGNNLayer(1, 1.)
    users, items = layer(h, u, e)
    doubled_users, doubled_items = layer(h.repeat(2, 1), u, e.repeat(2, 1))
    torch.testing.assert_close(users, doubled_users)
    torch.testing.assert_close(items, doubled_items[:3])
    assert torch.autograd.gradcheck(lambda incidence, embeds: layer(incidence, u, embeds), (h, e))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_model_training_recovery_and_rng(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    options = dict(behavior_view_mode='msca_struct', use_gpu=device == 'cuda', gpu_id=0)
    torch.manual_seed(17)
    old = LGMRecOpt(tiny_config(tmp_path, **options), Dataset())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(17)
    cfg = tiny_config(tmp_path, hyper_degree_power=.5, **options)
    data = Dataset()
    model = LGMRecOpt(cfg, data).to(device)
    assert data.calls == 1 and torch.equal(expected_rng, torch.get_rng_state())
    assert set(model.state_dict()) == set(old.state_dict())
    model.load_state_dict(old.state_dict(), strict=True)
    batch = torch.tensor([[0, 1], [0, 2], [4, 5]], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    initial = model.v_hyper.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.v_hyper.grad).all()
        optimizer.step()
    assert not torch.equal(initial, model.v_hyper)
    model.eval()
    restored = LGMRecOpt(cfg, Dataset()).to(device).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        result = model.full_sort_predict(batch[:1])
        assert torch.equal(result, model.full_sort_predict(batch[:1]))
        torch.testing.assert_close(result, restored.full_sort_predict(batch[:1]), rtol=0, atol=0)


@pytest.mark.parametrize('power', [-1., 1.1, float('nan'), float('inf')])
def test_invalid_power(tmp_path, power):
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, hyper_degree_power=power), Dataset())
