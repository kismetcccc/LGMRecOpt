"""Contracts for the isolated CROSS ILDA+DT adaptation."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from lgmrec.models.cross_ilda import ILALoss, UPSTREAM_COMMIT
from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.utils.configurator import Config


class Dataset:
    def get_user_num(self): return 4
    def get_item_num(self): return 6


class Loader:
    dataset = Dataset()

    def inter_matrix(self, form='coo'):
        return sp.coo_matrix(
            (np.ones(8),
             ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 4])),
            shape=(4, 6),
        ).asformat(form)


def tiny_config(tmp_path, **overrides):
    directory = tmp_path / 'baby'
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for name, width in [('image_feat.npy', 6), ('text_feat.npy', 4)]:
        np.save(directory / name, rng.normal(size=(6, width)).astype('float32'))
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


def test_module_parameter_count_finite_loss_and_duplicate_aggregation():
    torch.manual_seed(3)
    module = ILALoss(dim=8, gamma=.007, leaky_bi=True)
    assert UPSTREAM_COMMIT == '125928a921bcb7cc395086ea9c87543914d008b7'
    assert sum(parameter.numel() for parameter in module.parameters()) == 6 * 8 * 8 + 1
    embeddings = [torch.randn(rows, 8, requires_grad=True)
                  for rows in (3, 5, 5, 5)]
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    loss = module(*embeddings, users, items)
    assert loss.ndim == 0 and torch.isfinite(loss)
    duplicated = module(
        *embeddings,
        users.repeat_interleave(2),
        items.repeat_interleave(2),
    )
    torch.testing.assert_close(duplicated, loss)
    loss.backward()
    gradients = [parameter.grad for parameter in module.parameters()
                 if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)

    single = module(
        *(value.detach() for value in embeddings),
        torch.tensor([0]), torch.tensor([0]),
    )
    assert torch.isfinite(single)


def test_disabled_mode_preserves_c0_state_loss_gradient_and_rng(tmp_path):
    config = tiny_config(tmp_path)
    torch.manual_seed(19)
    base = LGMRec(config, Loader())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    model = LGMRecOpt(config, Loader())
    assert model.directed_align is None
    assert set(base.state_dict()) == set(model.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())
    batch = torch.tensor([[0, 1, 2], [0, 2, 3], [5, 5, 5]])
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


def test_enabled_mode_reuses_forward_logs_loss_and_trains_module_and_encoder(
        tmp_path, monkeypatch):
    model = LGMRecOpt(tiny_config(
        tmp_path, directed_align_mode='cross_ilda_dt', align_weight=.01
    ), Loader()).train()
    calls = 0
    original_forward = model.forward

    def counted_forward():
        nonlocal calls
        calls += 1
        return original_forward()

    monkeypatch.setattr(model, 'forward', counted_forward)
    batch = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4], [5, 5, 5, 5]])
    loss = model.calculate_loss(batch)
    assert calls == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None
               for parameter in model.directed_align.parameters())
    assert model.item_id_embedding.weight.grad is not None
    assert model.item_image_trs.grad is not None
    assert model.item_text_trs.grad is not None
    diagnostics = model.post_epoch_processing()
    assert 'loss_ilda_dt_raw=' in diagnostics
    assert 'loss_ilda_dt_weighted=' in diagnostics


def test_inference_never_runs_alignment_module(tmp_path, monkeypatch):
    model = LGMRecOpt(tiny_config(
        tmp_path, directed_align_mode='cross_ilda_dt'
    ), Loader()).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError('alignment must not run during inference')

    monkeypatch.setattr(model.directed_align, 'forward', forbidden)
    scores = model.full_sort_predict(torch.tensor([[0, 1]]))
    assert scores.shape == (2, model.n_items)


@pytest.mark.parametrize('overrides', [
    {'directed_align_mode': 'unknown'},
    {'directed_align_mode': 'cross_ilda_dt', 'train_context': 'target_mask'},
    {'directed_align_mode': 'cross_ilda_dt', 'score_mode': 'separate'},
    {'directed_align_mode': 'cross_ilda_dt', 'feat_mod_mode': 'id_cond'},
    {'align_weight': -1},
    {'align_weight': float('nan')},
])
def test_invalid_or_confounded_settings_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, **overrides), Loader())
