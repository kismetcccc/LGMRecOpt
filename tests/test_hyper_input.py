"""Normalize only the hyperedge routing input; keep the B default exact."""
import pytest
import torch
from torch.nn import functional as F
from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


def test_hyper_input_scale_invariance_and_zero_rows(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hyper_input_norm='l2'), Dataset())
    x = torch.randn(6, 6)
    x[0] = 0
    scale = torch.tensor([1., 3., .1, 100., 2., 4.])[:, None]
    logits = model.hyperedge_logits(x, model.v_hyper)
    torch.testing.assert_close(logits, model.hyperedge_logits(x * scale, model.v_hyper))
    torch.testing.assert_close(logits, F.normalize(x, dim=1) @ model.v_hyper)
    assert torch.isfinite(logits).all() and logits[0].eq(0).all()
    logits.square().sum().backward()
    assert model.v_hyper.grad is not None and model.v_hyper.grad.norm() > 0


@pytest.mark.parametrize('candidate_mode', ['l2', 'cosine', 'centered_l2'])
def test_default_parameters_rng_and_local_branch_preserved(tmp_path, candidate_mode):
    models, states = [], []
    for mode in ('off', candidate_mode):
        torch.manual_seed(11)
        models.append(LGMRecOpt(tiny_config(tmp_path, hyper_input_norm=mode,
                               behavior_view_mode='msca_struct'), Dataset()))
        states.append(torch.random.get_rng_state())
    baseline, candidate = models
    assert torch.equal(*states)
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    for modality in ('v', 't'):
        torch.testing.assert_close(baseline.mge(modality), candidate.mge(modality), rtol=0, atol=0)
    assert baseline.behavior_view.cache_fingerprint == candidate.behavior_view.cache_fingerprint
    x = baseline.image_embedding.weight
    torch.testing.assert_close(baseline.hyperedge_logits(x, baseline.v_hyper), x @ baseline.v_hyper, rtol=0, atol=0)
    loss = candidate.calculate_loss(torch.tensor([[0, 1, 2], [0, 1, 3], [4, 5, 4]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
    with pytest.raises(ValueError, match='hyper_input_norm'):
        LGMRecOpt(tiny_config(tmp_path, hyper_input_norm='bad'), Dataset())


def test_cosine_bounds_logits_and_removes_prototype_scale(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hyper_input_norm='cosine'), Dataset())
    x = torch.randn(6, 6)
    w = model.v_hyper
    logits = model.hyperedge_logits(x, w)
    scaled = model.hyperedge_logits(x * 75, w * torch.tensor([20., .05]))
    torch.testing.assert_close(logits, scaled)
    assert logits.abs().max() <= 1.000001
    torch.testing.assert_close(logits, F.normalize(x, dim=1) @ F.normalize(w, dim=0))


def test_centering_uses_only_train_items_and_matches_feature_centering_gradient(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hyper_input_norm='centered_l2'), Dataset())
    x = torch.randn(6, 6)
    unit = F.normalize(x, dim=1)
    # Tiny TRAIN graph observes items 0..3, never items 4 or 5.
    assert model._hyper_train_items.tolist() == [True, True, True, True, False, False]
    actual = model.hyperedge_logits(x, model.v_hyper)
    expected = (unit - unit[:4].mean(dim=0)) @ model.v_hyper
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:4].mean(dim=0), torch.zeros(2), atol=1e-7, rtol=0)
    before = actual[:4].detach().clone()
    x[4:] = torch.randn(2, 6) * 100
    torch.testing.assert_close(before, model.hyperedge_logits(x, model.v_hyper)[:4])
    a, = torch.autograd.grad(actual.square().sum(), model.v_hyper, retain_graph=True)
    b, = torch.autograd.grad(expected.square().sum(), model.v_hyper)
    torch.testing.assert_close(a, b)
