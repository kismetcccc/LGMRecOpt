"""Contracts for the isolated TRAIN-only MSCA behavior structural view."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.models.msca_behavior import (
    MSCA_MIN_COOCCURRENCE,
    MSCA_STRUCT_TOPK,
    MSCABehaviorView,
    behavior_graph_fingerprint,
    build_structural_adjacency,
)
from lgmrec.utils.configurator import Config


class Dataset:
    calls = 0

    @property
    def dataset(self):
        # The production model receives a train loader exposing its dataset.
        return self

    def get_user_num(self): return 4
    def get_item_num(self): return 6

    def inter_matrix(self, form='coo'):
        self.calls += 1
        matrix = sp.coo_matrix(
            (np.ones(12, dtype=np.float32),
             ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
              [0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 1, 3])),
            shape=(4, 6),
        )
        return matrix.asformat(form)


def test_identity_control_forward_gradient_and_rng():
    torch.manual_seed(17)
    before = torch.random.get_rng_state().clone()
    view = MSCABehaviorView(Dataset().inter_matrix(), 'cpu', graph_mode='identity')
    assert torch.equal(before, torch.random.get_rng_state())
    embeddings = torch.randn(6, 8, requires_grad=True)
    actual = view(embeddings)
    expected = torch.cat((view.normalized_interactions.to_dense() @ embeddings, embeddings))
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual.square().sum(), embeddings)
    expected_grad, = torch.autograd.grad(expected.square().sum(), embeddings)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert view.metadata['graph_mode'] == 'identity'
    assert view.metadata['normalized_nonzero_edges'] == 6
    assert not view.state_dict()
    real = MSCABehaviorView(Dataset().inter_matrix(), 'cpu')
    assert real.cache_fingerprint != view.cache_fingerprint
    with pytest.raises(ValueError, match='requires self_loop'):
        MSCABehaviorView(Dataset().inter_matrix(), 'cpu', graph_mode='identity', self_loop=False)


def tiny_config(tmp_path, **overrides):
    directory = tmp_path / 'baby'
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    np.save(directory / 'image_feat.npy',
            rng.normal(size=(6, 6)).astype('float32'))
    np.save(directory / 'text_feat.npy',
            rng.normal(size=(6, 4)).astype('float32'))
    values = dict(
        data_path=str(tmp_path) + os.sep,
        use_gpu=False,
        seed=999,
        embedding_size=8,
        feat_embed_dim=8,
        n_ui_layers=1,
        n_mm_layers=1,
        n_hyper_layer=1,
        hyper_num=2,
        keep_rate=1.,
        alpha=.1,
        cl_weight=1e-4,
        reg_weight=1e-6,
    )
    values.update(overrides)
    return Config('LGMRecOpt', 'baby', values)


def test_author_structural_rules_and_semantic_encode_are_exact():
    interactions = Dataset().inter_matrix()
    view = MSCABehaviorView(interactions, torch.device('cpu'), topk=1)
    assert MSCA_STRUCT_TOPK == 10
    assert MSCA_MIN_COOCCURRENCE == 2
    assert list(view.parameters()) == []
    dense = view.structural_adjacency.to_dense()
    expected = torch.zeros(6, 6)
    # With top-1, each active item selects its strongest neighbor; ties use
    # ascending item id. Every item also has the required self-loop.
    for row, column in [(0, 1), (1, 0), (2, 0), (3, 0)]:
        expected[row, column] = .5
    expected[0, 0] = expected[1, 1] = .5
    expected[2, 2] = expected[3, 3] = .5
    expected[4, 4] = expected[5, 5] = 1.
    torch.testing.assert_close(dense, expected)

    embeddings = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    actual = view(embeddings)
    item_expected = expected @ embeddings
    user_expected = view.normalized_interactions.to_dense() @ item_expected
    torch.testing.assert_close(
        actual, torch.cat((user_expected, item_expected), dim=0)
    )


def test_cache_identity_binds_train_matrix_and_graph_parameters():
    interactions = Dataset().inter_matrix()
    base = behavior_graph_fingerprint(interactions, topk=10, minimum=2)
    changed_matrix = interactions.tolil()
    changed_matrix[0, 5] = 1
    changed_matrix = changed_matrix.tocoo()
    assert base != behavior_graph_fingerprint(
        changed_matrix, topk=10, minimum=2
    )
    assert base != behavior_graph_fingerprint(
        interactions, topk=9, minimum=2
    )
    assert base != behavior_graph_fingerprint(
        interactions, topk=10, minimum=3
    )


def test_disabled_mode_preserves_state_and_rng(tmp_path):
    torch.manual_seed(19)
    first = LGMRecOpt(tiny_config(tmp_path), Dataset())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    second = LGMRecOpt(tiny_config(
        tmp_path, behavior_view_mode='off'
    ), Dataset())
    assert first.behavior_view is None
    assert set(first.state_dict()) == set(second.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_enabled_view_adds_exact_unnormalized_residual_and_no_state(tmp_path):
    torch.manual_seed(23)
    base = LGMRecOpt(tiny_config(tmp_path), Dataset()).eval()
    torch.manual_seed(23)
    model = LGMRecOpt(tiny_config(
        tmp_path, behavior_view_mode='msca_struct', behavior_eta=.4
    ), Dataset()).eval()
    assert set(base.state_dict()) == set(model.state_dict())
    torch.manual_seed(29)
    base_users, base_items, _ = base.forward()
    torch.manual_seed(29)
    users, items, _ = model.forward()
    behavior = model.behavior_view(model.item_id_embedding.weight)
    behavior_users, behavior_items = torch.split(
        behavior, [model.n_users, model.n_items], dim=0
    )
    torch.testing.assert_close(users, base_users + .4 * behavior_users)
    torch.testing.assert_close(items, base_items + .4 * behavior_items)


def test_train_graph_is_read_once_and_gradient_reaches_existing_item_ids(
        tmp_path):
    dataset = Dataset()
    model = LGMRecOpt(tiny_config(
        tmp_path, behavior_view_mode='msca_struct'
    ), dataset).train()
    assert dataset.calls == 1
    parameter_names = set(dict(model.named_parameters()))
    control_names = set(dict(LGMRecOpt(
        tiny_config(tmp_path), Dataset()
    ).named_parameters()))
    assert parameter_names == control_names
    loss = model.calculate_loss(
        torch.tensor([[0, 1, 2, 3], [0, 2, 3, 3], [5, 5, 5, 5]])
    )
    loss.backward()
    gradient = model.item_id_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert 'behavior' not in model.post_epoch_processing()


@pytest.mark.parametrize('overrides', [
    {'behavior_view_mode': 'unknown'},
    {'behavior_eta': -1},
    {'behavior_eta': float('nan')},
    {'behavior_view_mode': 'msca_struct',
     'joint_fusion_mode': 'smore_cross'},
    {'behavior_view_mode': 'msca_struct',
     'directed_align_mode': 'cross_ilda_dt'},
    {'behavior_view_mode': 'msca_struct', 'train_context': 'target_mask'},
    {'behavior_view_mode': 'msca_struct', 'score_mode': 'separate'},
    {'behavior_view_mode': 'msca_struct', 'feat_mod_mode': 'id_cond'},
])
def test_invalid_or_combined_settings_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, **overrides), Dataset())


@pytest.mark.parametrize('retired_key', [
    'behavior_align_mode',
    'behavior_align_weight',
    'behavior_align_tau',
])
def test_csa_configuration_is_retired(retired_key):
    with pytest.raises(ValueError):
        Config('LGMRecOpt', 'baby', {retired_key: 'off'})


def test_blockwise_matches_dense_reference_and_binarizes_duplicates():
    r = Dataset().inter_matrix().tocsr()
    doubled = r * 2
    expected = MSCABehaviorView(r, 'cpu').structural_adjacency.to_dense()
    for block_size in (1, 3, 100):
        actual = build_structural_adjacency(doubled, topk=10, minimum=2,
                                            device='cpu', block_size=block_size)
        torch.testing.assert_close(actual.to_dense(), expected)
    counts = (r.T @ r).toarray()
    np.fill_diagonal(counts, 0)
    raw = (counts >= 2).astype(np.float32) + np.eye(6, dtype=np.float32)
    scale = raw.sum(1) ** -.5
    torch.testing.assert_close(expected, torch.from_numpy(raw * scale[:, None] * scale[None, :]))


def test_random_relabel_preserves_topology_without_consuming_global_rng():
    r = Dataset().inter_matrix()
    before = np.random.get_state()
    graph = build_structural_adjacency(r, topk=1, minimum=2, device='cpu').to_dense()
    random_graph = build_structural_adjacency(r, topk=1, minimum=2, device='cpu',
                                             graph_mode='random_relabel', graph_seed=31).to_dense()
    permutation = np.random.default_rng(31).permutation(6)
    torch.testing.assert_close(random_graph[permutation][:, permutation], graph)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize('target', ['item', 'user', 'both'])
def test_residual_target_and_zero_eta(tmp_path, target):
    base = LGMRecOpt(tiny_config(tmp_path), Dataset()).eval()
    model = LGMRecOpt(tiny_config(tmp_path, behavior_view_mode='msca_struct',
        behavior_topk=1, behavior_minimum=3, behavior_residual_target=target), Dataset()).eval()
    model.load_state_dict(base.state_dict())
    bu, bi, _ = base.forward()
    u, i, _ = model.forward()
    residual = model.behavior_view(model.item_id_embedding.weight)
    torch.testing.assert_close(u, bu + (.2 * residual[:4] if target != 'item' else 0))
    torch.testing.assert_close(i, bi + (.2 * residual[4:] if target != 'user' else 0))
    model.behavior_eta = 0
    u, i, _ = model.forward()
    torch.testing.assert_close(u, bu)
    torch.testing.assert_close(i, bi)


def test_empty_no_loop_graph_is_finite_and_invalid_parameters_fail():
    r = sp.csr_matrix((3, 4), dtype=np.float32)
    graph = build_structural_adjacency(r, topk=1, minimum=1, device='cpu', self_loop=False)
    assert torch.isfinite(graph.to_dense()).all() and graph._nnz() == 0
    for args in ({'topk': 1.5}, {'minimum': 0}, {'block_size': 0}, {'graph_seed': -1}):
        options = dict(topk=1, minimum=2, device='cpu')
        options.update(args)
        with pytest.raises(ValueError):
            build_structural_adjacency(r, **options)
