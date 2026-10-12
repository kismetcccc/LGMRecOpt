"""History normalization changes assignment, not graphs or parameter sets."""
import pytest
import torch
from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


def test_mean_logits_reference_and_gradient(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, user_hyper_aggregation='mean'), Dataset())
    logits = torch.randn(6, 2, requires_grad=True)
    actual = model.user_hyperedge_logits(logits)
    expected = torch.stack([logits[[0, 1, 2]].mean(0), logits[[0, 1, 2]].mean(0),
                            logits[[0, 1, 3]].mean(0), logits[[0, 1, 3]].mean(0)])
    torch.testing.assert_close(actual, expected)
    left, = torch.autograd.grad(actual.square().sum(), logits, retain_graph=True)
    right, = torch.autograd.grad(expected.square().sum(), logits)
    torch.testing.assert_close(left, right)


def test_equal_average_histories_have_equal_assignments(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, user_hyper_aggregation='mean'), Dataset())
    # Two distinct histories, same mean content, lengths one and three.
    model.adj = torch.tensor([[1., 0., 0.], [1., 1., 1.]]).to_sparse()
    model.n_users = 2
    model.num_inters = torch.tensor([[1.], [1 / 3]])
    logits = torch.tensor([[.1, 0.], [.2, 0.], [0., 0.]])
    model.eval()
    mean = model._hyperedge_distribution(model.user_hyperedge_logits(logits))
    torch.testing.assert_close(mean[0], mean[1])
    model.user_hyper_aggregation = 'sum'
    summed = model._hyperedge_distribution(model.user_hyperedge_logits(logits))
    assert summed[1, 0] > summed[0, 0]


def test_initialization_default_and_training(tmp_path):
    models = []
    states = []
    for mode in ('sum', 'mean'):
        torch.manual_seed(9)
        models.append(LGMRecOpt(tiny_config(tmp_path, user_hyper_aggregation=mode,
                               behavior_view_mode='msca_struct'), Dataset()))
        states.append(torch.random.get_rng_state())
    old, new = models
    assert torch.equal(*states)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    assert old.behavior_view.cache_fingerprint == new.behavior_view.cache_fingerprint
    logits = torch.randn(6, 2)
    torch.testing.assert_close(old.user_hyperedge_logits(logits), torch.mm(old.adj, logits), rtol=0, atol=0)
    loss = new.calculate_loss(torch.tensor([[0, 1, 2], [0, 1, 3], [4, 5, 4]]))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (new.v_hyper, new.t_hyper):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    with pytest.raises(ValueError, match='user_hyper_aggregation'):
        LGMRecOpt(tiny_config(tmp_path, user_hyper_aggregation='invalid'), Dataset())
