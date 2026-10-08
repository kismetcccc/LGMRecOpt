"""Contracts for the three LGMRecOpt scoring modes."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import CROSS_BRANCH_PAIRS, LGMRecOpt
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


def tiny_config(tmp_path, score_mode='original', **overrides):
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
        score_mode=score_mode,
    )
    values.update(overrides)
    return Config('LGMRecOpt', 'baby', values)


def test_original_preserves_parameters_loss_prediction_gradient_and_rng(tmp_path):
    cfg = tiny_config(tmp_path)
    torch.manual_seed(19)
    base = LGMRec(cfg, Loader())
    base_rng = torch.get_rng_state()
    torch.manual_seed(19)
    model = LGMRecOpt(cfg, Loader())
    assert torch.equal(base_rng, torch.get_rng_state())
    assert set(dict(base.named_parameters())) == set(dict(model.named_parameters()))
    assert set(base.state_dict()) == set(model.state_dict())

    batch = torch.tensor([[0, 1], [0, 2], [3, 4]])
    base.eval(); model.eval()
    assert torch.equal(
        base.full_sort_predict(batch[:1]),
        model.full_sort_predict(batch[:1]),
    )
    base.train(); model.train()
    torch.manual_seed(23)
    left = base.calculate_loss(batch)
    left.backward()
    loss_rng = torch.get_rng_state()
    torch.manual_seed(23)
    right = model.calculate_loss(batch)
    right.backward()
    assert torch.equal(left, right)
    assert torch.equal(loss_rng, torch.get_rng_state())
    for name, parameter in base.named_parameters():
        if parameter.grad is not None:
            assert torch.equal(
                parameter.grad, dict(model.named_parameters())[name].grad
            ), name


def test_cross_initial_score_matches_original_with_six_bounded_parameters(tmp_path):
    original = LGMRecOpt(tiny_config(tmp_path, 'original'), Loader()).eval()
    cross = LGMRecOpt(tiny_config(tmp_path, 'cross'), Loader()).eval()
    cross.load_state_dict(original.state_dict(), strict=False)
    assert len(cross.cross_logits) == len(CROSS_BRANCH_PAIRS) == 6
    assert sum(parameter.numel() for parameter in cross.cross_logits.values()) == 6
    coefficients = torch.stack([
        2 * torch.sigmoid(parameter)
        for parameter in cross.cross_logits.values()
    ])
    assert torch.equal(coefficients, torch.ones(6))
    batch = torch.tensor([[0, 1]])
    torch.testing.assert_close(
        cross.full_sort_predict(batch),
        original.full_sort_predict(batch),
        rtol=1e-5,
        atol=1e-6,
    )


def test_cross_gradients_and_frozen_encoder_update_only_six_parameters(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, 'cross'), Loader())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('cross_logits.'))
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss = model.calculate_loss(torch.tensor([[0, 1], [0, 2], [3, 4]]))
    optimizer.zero_grad()
    loss.backward()
    assert all(parameter.grad is not None
               for parameter in model.cross_logits.values())
    optimizer.step()
    changed = [name for name, parameter in model.named_parameters()
               if not torch.equal(before[name], parameter.detach())]
    assert changed
    assert set(changed).issubset({
        f'cross_logits.{left}_{right}' for left, right in CROSS_BRANCH_PAIRS
    })


@pytest.mark.parametrize('mode', ['original', 'separate', 'cross'])
def test_training_pair_scores_are_entries_of_full_sort_scores(tmp_path, mode):
    model = LGMRecOpt(tiny_config(tmp_path, mode), Loader()).eval()
    users = torch.tensor([0, 1])
    positives = torch.tensor([1, 2])
    negatives = torch.tensor([3, 4])
    combined_users, combined_items, _ = model.forward()
    user_branches = tuple(branch[users] for branch in model._score_user_branches)
    positive_branches = tuple(
        branch[positives] for branch in model._score_item_branches
    )
    negative_branches = tuple(
        branch[negatives] for branch in model._score_item_branches
    )
    positive_scores = model.score(
        user_branches,
        positive_branches,
        combined=(combined_users[users], combined_items[positives]),
    )
    negative_scores = model.score(
        user_branches,
        negative_branches,
        combined=(combined_users[users], combined_items[negatives]),
    )
    full_scores = model.full_sort_predict(torch.stack((users, positives)))
    rows = torch.arange(users.numel())
    torch.testing.assert_close(positive_scores, full_scores[rows, positives])
    torch.testing.assert_close(negative_scores, full_scores[rows, negatives])


@pytest.mark.parametrize('mode', ['original', 'separate', 'cross'])
def test_scoring_modes_keep_train_only_guard(tmp_path, mode):
    with pytest.raises(ValueError, match='train_only'):
        LGMRecOpt(tiny_config(tmp_path, mode, protocol='legacy'), Loader())


def test_unknown_score_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='score_mode'):
        LGMRecOpt(tiny_config(tmp_path, 'unknown'), Loader())
