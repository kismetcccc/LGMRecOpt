"""Contracts for conservative batch target-edge context masking."""
import os
from types import MethodType

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.utils.configurator import Config


class Dataset:
    def get_user_num(self): return 3
    def get_item_num(self): return 5


class Loader:
    dataset = Dataset()

    def inter_matrix(self, form='coo'):
        return sp.coo_matrix(
            (np.ones(5), ([0, 0, 1, 1, 2], [0, 1, 1, 2, 3])),
            shape=(3, 5),
        ).asformat(form)


def tiny_config(tmp_path, train_context='full', **overrides):
    directory = tmp_path / 'baby'
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for name, width in [('image_feat.npy', 6), ('text_feat.npy', 4)]:
        np.save(directory / name, rng.normal(size=(5, width)).astype('float32'))
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
        train_context=train_context,
    )
    values.update(overrides)
    return Config('LGMRecOpt', 'baby', values)


def test_full_default_preserves_c0_parameters_loss_gradient_and_rng(tmp_path):
    config = tiny_config(tmp_path)
    torch.manual_seed(19)
    base = LGMRec(config, Loader())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    model = LGMRecOpt(config, Loader())
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert set(base.state_dict()) == set(model.state_dict())
    batch = torch.tensor([[0, 1], [0, 2], [3, 4]])
    torch.manual_seed(23)
    expected = base.calculate_loss(batch)
    expected.backward()
    loss_rng = torch.get_rng_state()
    torch.manual_seed(23)
    actual = model.calculate_loss(batch)
    actual.backward()
    assert torch.equal(expected, actual)
    assert torch.equal(loss_rng, torch.get_rng_state())
    for name, parameter in base.named_parameters():
        if parameter.grad is not None:
            assert torch.equal(
                parameter.grad, dict(model.named_parameters())[name].grad
            ), name


def test_mask_is_stable_deduplicated_and_preserves_both_endpoint_degrees(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, 'target_mask'), Loader())
    graph = model._build_target_mask_graph(
        torch.tensor([0, 0, 1]), torch.tensor([1, 1, 1])
    )
    adj, norm_adj, num_inters, candidates, masked = graph
    assert (candidates, masked) == (2, 1)
    dense = adj.to_dense()
    assert dense[0, 1] == 0
    assert dense[1, 1] == 1
    assert torch.all(dense.sum(dim=1) >= 1)
    assert torch.all(dense.sum(dim=0)[[0, 1, 2, 3]] >= 1)
    symmetric = norm_adj.to_dense()
    assert torch.equal(symmetric, symmetric.T)
    assert torch.allclose(
        num_inters[:model.n_users, 0], 1 / (dense.sum(dim=1) + 1e-7)
    )


def test_cge_mge_and_ghe_observe_one_temporary_graph_then_restore(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, 'target_mask'), Loader()).train()
    full_graph = model.adj, model.norm_adj, model.num_inters
    observations = []
    original_cge = model.cge
    original_mge = model.mge
    original_distribution = model._hyperedge_distribution

    def cge(self):
        observations.append(('cge', id(self.adj), id(self.norm_adj)))
        return original_cge()

    def mge(self, kind='v'):
        observations.append((f'mge-{kind}', id(self.adj), id(self.norm_adj)))
        return original_mge(kind)

    def distribution(self, logits):
        observations.append(('ghe', id(self.adj), id(self.norm_adj)))
        return original_distribution(logits)

    model.cge = MethodType(cge, model)
    model.mge = MethodType(mge, model)
    model._hyperedge_distribution = MethodType(distribution, model)
    model.calculate_loss(torch.tensor([[0], [1], [4]]))
    assert {entry[1:] for entry in observations} == {
        (observations[0][1], observations[0][2])
    }
    assert observations[0][1] != id(full_graph[0])
    assert (model.adj, model.norm_adj, model.num_inters) == full_graph
    diagnostics = model.post_epoch_processing()
    assert 'target_mask_ratio=1.000000' in diagnostics
    assert 'target_masked=1' in diagnostics


def test_graph_restores_after_forward_exception_and_eval_uses_full_graph(
        tmp_path, monkeypatch):
    model = LGMRecOpt(tiny_config(tmp_path, 'target_mask'), Loader()).train()
    full_graph = model.adj, model.norm_adj, model.num_inters

    def fail():
        assert model.adj is not full_graph[0]
        raise RuntimeError('forward failed')

    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='forward failed'):
        model.calculate_loss(torch.tensor([[0], [1], [4]]))
    assert (model.adj, model.norm_adj, model.num_inters) == full_graph

    seen = []

    def eval_fail():
        seen.append(model.adj)
        assert model.adj is full_graph[0]
        raise RuntimeError('eval used full graph')

    monkeypatch.setattr(model, 'forward', eval_fail)
    model.eval()
    with pytest.raises(RuntimeError, match='eval used full graph'):
        model.calculate_loss(torch.tensor([[0], [1], [4]]))
    assert seen == [full_graph[0]]


@pytest.mark.parametrize('overrides', [
    {'train_context': 'unknown'},
    {'train_context': 'target_mask', 'score_mode': 'separate'},
    {'train_context': 'target_mask', 'feat_mod_mode': 'id_cond'},
])
def test_invalid_or_confounded_target_mask_settings_are_rejected(
        tmp_path, overrides):
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, **overrides), Loader())


def test_non_train_target_is_rejected_and_adds_no_parameters(tmp_path):
    full = LGMRecOpt(tiny_config(tmp_path), Loader())
    masked = LGMRecOpt(tiny_config(tmp_path, 'target_mask'), Loader())
    assert set(dict(full.named_parameters())) == set(dict(masked.named_parameters()))
    with pytest.raises(ValueError, match='TRAIN graph'):
        masked._build_target_mask_graph(torch.tensor([0]), torch.tensor([4]))
