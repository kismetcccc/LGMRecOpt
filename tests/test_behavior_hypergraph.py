"""Structural, numerical and TRAIN-only contracts for the opt-in candidate."""
import numpy as np
import pytest
import torch

from lgmrec.models.behavior_hypergraph import BehaviorHypergraphRefiner
from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


def graph():
    # Asymmetric weighted neighbors and an isolated/self-only item.
    return torch.tensor([[1., 2., 1., 0.], [1., 1., 0., 0.],
                         [0., 3., 1., 0.], [0., 0., 0., 1.]]).to_sparse()


def test_dense_formula_isolation_and_gradient():
    refiner = BehaviorHypergraphRefiner(graph(), .2)
    p = torch.tensor([[0., 2/3, 1/3, 0.], [1., 0., 0., 0.],
                      [0., 1., 0., 0.], [0., 0., 0., 1.]])
    torch.testing.assert_close(refiner.transition.to_dense(), p)
    logits = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    actual = refiner(logits)
    torch.testing.assert_close(actual, logits + .2 * (p @ logits - logits))
    assert torch.equal(actual[3], logits[3])
    actual.sum().backward()
    torch.testing.assert_close(logits.grad, (.8 * torch.eye(4) + .2 * p).sum(0)[:, None].expand(4, 3))
    assert not list(refiner.parameters()) and not refiner.state_dict()
    assert refiner.metadata['items_with_neighbors'] == 3


def test_random_control_preserves_topology_and_global_rng():
    torch_before, np_before = torch.get_rng_state(), np.random.get_state()
    real = BehaviorHypergraphRefiner(graph(), .2)
    control = BehaviorHypergraphRefiner(graph(), .2, graph_mode='random_relabel', seed=19)
    permutation = np.random.default_rng(19).permutation(4)
    torch.testing.assert_close(control.transition.to_dense()[permutation][:, permutation],
                               real.transition.to_dense())
    assert torch.equal(torch_before, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_before, np.random.get_state()))
    assert real.metadata['fingerprint'] != control.metadata['fingerprint']


def test_empty_graph_and_zero_weight_are_identity():
    empty = torch.zeros(4, 4).to_sparse()
    logits = torch.randn(4, 3)
    assert torch.equal(BehaviorHypergraphRefiner(empty, .4)(logits), logits)
    assert BehaviorHypergraphRefiner(graph(), 0)(logits) is logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_sparse_refinement_and_model_step(tmp_path):
    device = torch.device('cuda:0')
    refiner = BehaviorHypergraphRefiner(graph().to(device), .2)
    logits = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3).requires_grad_()
    torch.testing.assert_close(refiner(logits).cpu(),
                               BehaviorHypergraphRefiner(graph(), .2)(logits.cpu()))
    refiner(logits).sum().backward()
    assert torch.isfinite(logits.grad).all()
    cfg = tiny_config(tmp_path, use_gpu=True, gpu_id=0,
                      behavior_view_mode='msca_struct', hyper_behavior_weight=.2)
    model = LGMRecOpt(cfg, Dataset()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    batch = torch.tensor([[0, 1], [0, 2], [4, 5]], device=device)
    optimizer.zero_grad()
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.v_hyper.grad).all()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        first = model.full_sort_predict(batch[:1])
        second = model.full_sort_predict(batch[:1])
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['cooccurrence', 'random_relabel'])
def test_zero_matches_stage1_loss_gradients_and_rng(tmp_path, mode):
    def make(**options):
        torch.manual_seed(17)
        return LGMRecOpt(tiny_config(tmp_path, behavior_view_mode='msca_struct',
            keep_rate=.5, **options), Dataset())
    old = make()
    init_rng = torch.get_rng_state()
    new = make(hyper_behavior_weight=0, hyper_behavior_graph_mode=mode)
    assert new.hyper_behavior is None
    assert torch.equal(init_rng, torch.get_rng_state())
    batch = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 5]])
    torch.manual_seed(23)
    left = old.calculate_loss(batch)
    left.backward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(23)
    right = new.calculate_loss(batch)
    right.backward()
    assert torch.equal(left, right) and torch.equal(expected_rng, torch.get_rng_state())
    for name, parameter in old.named_parameters():
        other = dict(new.named_parameters())[name]
        assert torch.equal(parameter, other)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name


@pytest.mark.parametrize('mode', ['cooccurrence', 'random_relabel'])
def test_active_hypergraph_training_checkpoint_and_deterministic_eval(tmp_path, mode):
    dataset = Dataset()
    torch.manual_seed(7)
    base = LGMRecOpt(tiny_config(tmp_path, behavior_view_mode='msca_struct'), Dataset())
    torch.manual_seed(7)
    cfg = tiny_config(tmp_path, behavior_view_mode='msca_struct',
                      hyper_behavior_weight=.2, hyper_behavior_graph_mode=mode)
    model = LGMRecOpt(cfg, dataset)
    assert dataset.calls == 1  # No validation/test graph or extra data access.
    assert set(base.state_dict()) == set(model.state_dict())
    for key in base.state_dict():
        assert torch.equal(base.state_dict()[key], model.state_dict()[key])
    assert torch.equal(base.behavior_view.structural_adjacency.to_dense(),
                       model.behavior_view.structural_adjacency.to_dense())
    model.eval(); base.eval()
    assert not torch.equal(model.full_sort_predict(torch.tensor([[0, 1]])),
                           base.full_sort_predict(torch.tensor([[0, 1]])))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    batch = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 3], [5, 5, 5, 5]])
    initial = model.v_hyper.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in (model.v_hyper, model.t_hyper, model.item_id_embedding.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(initial, model.v_hyper)
    model.eval()
    rng = torch.get_rng_state()
    scores = model.full_sort_predict(batch[:1])
    assert torch.equal(scores, model.full_sort_predict(batch[:1]))
    assert torch.equal(rng, torch.get_rng_state())
    restored = LGMRecOpt(cfg, Dataset()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(scores, restored.full_sort_predict(batch[:1]))


@pytest.mark.parametrize('overrides', [
    {'hyper_behavior_weight': -.1}, {'hyper_behavior_weight': 1.1},
    {'hyper_behavior_weight': float('nan')}, {'hyper_behavior_weight': float('inf')},
    {'hyper_behavior_graph_mode': 'unknown'},
    {'hyper_behavior_weight': .2, 'behavior_view_mode': 'off'},
    {'hyper_behavior_weight': .2, 'behavior_graph_mode': 'random_relabel'},
    {'hyper_behavior_weight': .2, 'alpha': 0},
    {'hyper_behavior_weight': .2, 'n_hyper_layer': 0},
])
def test_invalid_settings(tmp_path, overrides):
    options = dict(behavior_view_mode='msca_struct')
    options.update(overrides)
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, **options), Dataset())
