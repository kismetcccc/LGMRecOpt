"""Sparse GHE integration and exact default-path compatibility contracts."""
import io
from pathlib import Path
import subprocess
import types

import pytest
import torch

from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


@pytest.fixture(scope='module')
def previous_models():
    """An independent in-memory oracle; source archives need no Git history."""
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for name in ('lgmrec', 'lgmrecopt'):
        try:
            result = subprocess.run(
                ['git', 'show', f'3e73d65:src/lgmrec/models/{name}.py'],
                cwd=root, capture_output=True, text=True, encoding='utf-8',
            )
        except FileNotFoundError:
            pytest.skip('Exact historical oracle requires Git and commit 3e73d65')
        if result.returncode:
            pytest.skip('Exact historical oracle requires commit 3e73d65; absent in source-only packages')
        sources[name] = result.stdout
    base = types.ModuleType('lgmrec.models._pre_sparse_base')
    base.__package__ = 'lgmrec.models'
    exec(compile(sources['lgmrec'], '<3e73d65/lgmrec.py>', 'exec'), base.__dict__)
    extension = types.ModuleType('lgmrec.models._pre_sparse_extension')
    extension.__package__ = 'lgmrec.models'
    extension.LGMRec = base.LGMRec
    base_import = 'from .lgmrec import LGMRec'
    assert sources['lgmrecopt'].count(base_import) == 1
    source = sources['lgmrecopt'].replace(base_import, '# Independent historical base injected above.')
    exec(compile(source, '<3e73d65/lgmrecopt.py>', 'exec'), extension.__dict__)
    return base.LGMRec, extension.LGMRecOpt


def assert_same_output(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_output(a, b)


@pytest.mark.parametrize('kind', ['baseline', 'B'])
def test_prototype_matches_pre_refactor_forward_gradients_and_rng(tmp_path, previous_models, kind):
    reference_class = previous_models[0 if kind == 'baseline' else 1]
    current_class = LGMRec if kind == 'baseline' else LGMRecOpt
    cfg = tiny_config(tmp_path, keep_rate=.5, behavior_view_mode='msca_struct')
    models, init_states = [], []
    for model_class in (reference_class, current_class):
        torch.manual_seed(71)
        models.append(model_class(cfg, Dataset()))
        init_states.append(torch.get_rng_state())
    reference, current = models
    assert torch.equal(*init_states)
    assert reference.state_dict().keys() == current.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, current.state_dict()[key]), key

    torch.manual_seed(73)
    expected = reference.forward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(73)
    actual = current.forward()
    assert_same_output(expected, actual)
    assert torch.equal(expected_rng, torch.get_rng_state())

    batch = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 5]])
    torch.manual_seed(79)
    expected_loss = reference.calculate_loss(batch)
    expected_loss.backward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(79)
    actual_loss = current.calculate_loss(batch)
    actual_loss.backward()
    assert torch.equal(expected_loss, actual_loss)
    assert torch.equal(expected_rng, torch.get_rng_state())
    current_parameters = dict(current.named_parameters())
    for name, parameter in reference.named_parameters():
        actual_grad = current_parameters[name].grad
        assert (parameter.grad is None) == (actual_grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, actual_grad), name
    reference.eval()
    current.eval()
    before = torch.get_rng_state()
    assert_same_output(reference.full_sort_predict(batch[:1]), current.full_sort_predict(batch[:1]))
    assert torch.equal(before, torch.get_rng_state())


def sparse_config(tmp_path, **overrides):
    options = dict(behavior_view_mode='msca_struct', hypergraph_mode='sparse_group')
    options.update(overrides)
    return tiny_config(tmp_path, **options)


@pytest.mark.parametrize('weighting', ['attention', 'uniform'])
def test_sparse_path_never_calls_prototypes_and_preserves_local_initialization(tmp_path, monkeypatch, weighting):
    torch.manual_seed(83)
    baseline = LGMRecOpt(tiny_config(tmp_path, behavior_view_mode='msca_struct'), Dataset())
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(83)
    dataset = Dataset()
    model = LGMRecOpt(sparse_config(tmp_path, hyper_group_weighting=weighting), dataset)
    assert dataset.calls == 1
    assert torch.equal(baseline_rng, torch.get_rng_state())
    assert baseline.state_dict().keys() == model.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    assert_same_output(baseline.cge(), model.cge())
    for modality in ('v', 't'):
        assert_same_output(baseline.mge(modality), model.mge(modality))
    assert baseline.behavior_view.cache_fingerprint == model.behavior_view.cache_fingerprint
    assert not model.v_hyper.requires_grad and not model.t_hyper.requires_grad
    assert model.item_image_trs.requires_grad and model.item_text_trs.requires_grad

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Sparse GHE must not execute prototype assignment, Gumbel, or prototype HGNN')

    for name in ('hyperedge_logits', 'user_hyperedge_logits', '_hyperedge_distribution'):
        monkeypatch.setattr(model, name, forbidden)
    monkeypatch.setattr(model.hgnnLayer, 'forward', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'gumbel_softmax', forbidden)
    before = torch.get_rng_state()
    model.train()
    training_output = model.forward()
    model.eval()
    assert_same_output(training_output, model.forward())
    assert torch.equal(before, torch.get_rng_state())


def test_sparse_support_is_train_only_and_not_stored_in_checkpoints(tmp_path):
    cfg = sparse_config(tmp_path)
    model = LGMRecOpt(cfg, Dataset()).eval()
    layer = model.sparse_hypergraph
    groups = [set(row[valid].tolist()) for row, valid in zip(layer.members, layer.mask)]
    # TRAIN observes 0..3; 2 and 3 never co-occur, and catalog items 4/5 stay singleton.
    assert groups == [{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {4}, {5}]
    assert not layer.state_dict()
    assert not list(layer.parameters())
    assert not any(name.startswith('sparse_hypergraph.') for name in model.state_dict())

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = LGMRecOpt(cfg, Dataset()).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(layer.members, restored.sparse_hypergraph.members)
    assert torch.equal(layer.mask, restored.sparse_hypergraph.mask)
    users = torch.tensor([[0, 1, 2]])
    assert_same_output(model.full_sort_predict(users), restored.full_sort_predict(users))


def test_attention_hypergraph_itself_trains_reused_modal_projections(tmp_path):
    torch.manual_seed(89)
    model = LGMRecOpt(sparse_config(tmp_path), Dataset())
    _, _, hyper = model.forward()
    hyper_only_loss = sum(embedding.square().sum() for embedding in hyper)
    gradients = torch.autograd.grad(
        hyper_only_loss, (model.item_image_trs, model.item_text_trs),
    )
    # Checking only the new GHE avoids crediting the unchanged MGE gradient.
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize('weighting', ['attention', 'uniform'])
def test_sparse_training_loss_and_gradients_are_finite(tmp_path, weighting):
    torch.manual_seed(97)
    model = LGMRecOpt(sparse_config(tmp_path, hyper_group_weighting=weighting), Dataset())
    batch = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 5]])
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    original_prototypes = (model.v_hyper.detach().clone(), model.t_hyper.detach().clone())
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        assert model.v_hyper.grad is None and model.t_hyper.grad is None
        assert model.item_image_trs.grad is not None and model.item_text_trs.grad is not None
        optimizer.step()
    assert torch.equal(original_prototypes[0], model.v_hyper)
    assert torch.equal(original_prototypes[1], model.t_hyper)


@pytest.mark.parametrize('options', [
    {'hypergraph_mode': 'unknown'}, {'hyper_group_weighting': 'unknown'},
    {'behavior_view_mode': 'off'}, {'behavior_graph_mode': 'identity'},
    {'behavior_graph_mode': 'random_relabel'}, {'behavior_neighbor_score': 'cosine'},
    {'behavior_self_loop': False}, {'behavior_edge_weight': 'count'},
    {'hyper_input_norm': 'l2'}, {'hyper_input_norm': 'centered_l2'},
    {'user_hyper_aggregation': 'mean'}, {'hyper_behavior_weight': .2},
    {'hyper_degree_power': .5}, {'n_hyper_layer': 0}, {'alpha': 0},
    {'score_mode': 'cross'}, {'train_context': 'target_mask'},
    {'feat_mod_mode': 'id_cond'}, {'directed_align_mode': 'cross_ilda_dt'},
    {'joint_fusion_mode': 'smore_cross'},
])
def test_sparse_rejects_unsupported_combinations(tmp_path, options):
    with pytest.raises(ValueError):
        LGMRecOpt(sparse_config(tmp_path, **options), Dataset())
