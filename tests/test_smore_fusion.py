"""Contracts for the isolated SMORE cross-modality fusion branch."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.models.smore_fusion import (
    CrossModalSpectrumFusion,
    UPSTREAM_COMMIT,
)
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


def test_spectral_module_matches_smore_formula_and_parameter_count():
    torch.manual_seed(3)
    module = CrossModalSpectrumFusion(8)
    assert UPSTREAM_COMMIT == 'c745bb92932981b371e8480ae9376ec3e71aaf24'
    assert module.fusion_complex_weight.shape == (1, 5, 2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 10
    visual = torch.randn(6, 8, requires_grad=True)
    text = torch.randn(6, 8, requires_grad=True)
    actual = module(visual, text)
    expected = torch.fft.irfft(
        torch.fft.rfft(visual, dim=1, norm='ortho')
        * torch.fft.rfft(text, dim=1, norm='ortho')
        * torch.view_as_complex(module.fusion_complex_weight),
        n=8,
        dim=1,
        norm='ortho',
    )
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    assert module.fusion_complex_weight.grad is not None
    assert torch.isfinite(module.fusion_complex_weight.grad).all()
    assert visual.grad is not None and text.grad is not None


def test_disabled_mode_preserves_c0_state_loss_gradient_and_rng(tmp_path):
    config = tiny_config(tmp_path)
    torch.manual_seed(19)
    base = LGMRec(config, Loader())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    model = LGMRecOpt(config, Loader())
    assert model.smore_fusion is None
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


def test_enabled_initialization_is_rng_isolated_and_adds_only_filter(tmp_path):
    torch.manual_seed(29)
    base = LGMRecOpt(tiny_config(tmp_path), Loader())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(29)
    fused = LGMRecOpt(tiny_config(
        tmp_path, joint_fusion_mode='smore_cross'
    ), Loader())
    assert torch.equal(expected_rng, torch.get_rng_state())
    base_state = base.state_dict()
    fused_state = fused.state_dict()
    assert set(fused_state) - set(base_state) == {
        'smore_fusion.fusion_complex_weight'
    }
    for name, value in base_state.items():
        assert torch.equal(value, fused_state[name]), name
    assert fused.smore_fusion.fusion_complex_weight.numel() == 10


def test_preference_gate_initialization_is_isolated_and_has_expected_size(
        tmp_path):
    torch.manual_seed(37)
    ungated = LGMRecOpt(tiny_config(
        tmp_path, joint_fusion_mode='smore_cross'
    ), Loader())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    gated = LGMRecOpt(tiny_config(
        tmp_path,
        joint_fusion_mode='smore_cross',
        joint_fusion_gate='smore_prefer',
    ), Loader())
    assert torch.equal(expected_rng, torch.get_rng_state())
    added = set(gated.state_dict()) - set(ungated.state_dict())
    assert added == {
        'smore_preference_gate.0.weight',
        'smore_preference_gate.0.bias',
    }
    assert sum(
        parameter.numel()
        for parameter in gated.smore_preference_gate.parameters()
    ) == 8 * 8 + 8
    for name, value in ungated.state_dict().items():
        assert torch.equal(value, gated.state_dict()[name]), name


def test_author_dimension_preference_gate_adds_4160_parameters():
    gate = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Sigmoid())
    assert sum(parameter.numel() for parameter in gate.parameters()) == 4160


def test_forward_adds_normalized_joint_mge_residual(tmp_path):
    base = LGMRecOpt(tiny_config(tmp_path), Loader()).eval()
    fused = LGMRecOpt(tiny_config(
        tmp_path, joint_fusion_mode='smore_cross', fusion_beta=.1
    ), Loader()).eval()
    fused.load_state_dict(base.state_dict(), strict=False)
    torch.manual_seed(31)
    base_users, base_items, _ = base.forward()
    torch.manual_seed(31)
    fused_users, fused_items, _ = fused.forward()
    joint = F.normalize(fused._smore_joint_mge())
    joint_users, joint_items = torch.split(
        joint, [fused.n_users, fused.n_items], dim=0
    )
    torch.testing.assert_close(fused_users, base_users + .1 * joint_users)
    torch.testing.assert_close(fused_items, base_items + .1 * joint_items)


def test_preference_gate_uses_shared_cge_after_joint_normalization(tmp_path):
    ungated = LGMRecOpt(tiny_config(
        tmp_path, joint_fusion_mode='smore_cross', fusion_beta=.1
    ), Loader()).eval()
    gated = LGMRecOpt(tiny_config(
        tmp_path,
        joint_fusion_mode='smore_cross',
        joint_fusion_gate='smore_prefer',
        fusion_beta=.1,
    ), Loader()).eval()
    gated.load_state_dict(ungated.state_dict(), strict=False)
    torch.nn.init.zeros_(gated.smore_preference_gate[0].weight)
    torch.nn.init.zeros_(gated.smore_preference_gate[0].bias)

    torch.manual_seed(41)
    ungated_users, ungated_items, _ = ungated.forward()
    ungated_joint = F.normalize(ungated._smore_joint_mge())
    torch.manual_seed(41)
    gated_users, gated_items, _ = gated.forward()
    gated_joint = F.normalize(gated._smore_joint_mge())
    torch.testing.assert_close(gated_joint, ungated_joint)
    joint_users, joint_items = torch.split(
        ungated_joint, [gated.n_users, gated.n_items], dim=0
    )
    torch.testing.assert_close(
        gated_users, ungated_users - .05 * joint_users
    )
    torch.testing.assert_close(
        gated_items, ungated_items - .05 * joint_items
    )


def test_training_updates_filter_and_existing_projections_without_new_loss(
        tmp_path):
    model = LGMRecOpt(tiny_config(
        tmp_path, joint_fusion_mode='smore_cross', fusion_beta=.1
    ), Loader()).train()
    loss = model.calculate_loss(
        torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4], [5, 5, 5, 5]])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.smore_fusion.fusion_complex_weight.grad is not None
    assert torch.isfinite(
        model.smore_fusion.fusion_complex_weight.grad
    ).all()
    assert model.item_image_trs.grad is not None
    assert model.item_text_trs.grad is not None
    assert model.image_embedding.weight.requires_grad is False
    assert model.text_embedding.weight.requires_grad is False
    diagnostics = model.post_epoch_processing()
    assert 'ilda' not in diagnostics
    assert 'fusion' not in diagnostics


def test_preference_gate_receives_gradient_without_auxiliary_loss(tmp_path):
    model = LGMRecOpt(tiny_config(
        tmp_path,
        joint_fusion_mode='smore_cross',
        joint_fusion_gate='smore_prefer',
        fusion_beta=.1,
    ), Loader()).train()
    loss = model.calculate_loss(
        torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4], [5, 5, 5, 5]])
    )
    loss.backward()
    for parameter in model.smore_preference_gate.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.smore_fusion.fusion_complex_weight.grad is not None
    assert model.item_id_embedding.weight.grad is not None
    diagnostics = model.post_epoch_processing()
    assert 'ilda' not in diagnostics
    assert 'fusion' not in diagnostics


@pytest.mark.parametrize('overrides', [
    {'joint_fusion_mode': 'unknown'},
    {'joint_fusion_mode': 'smore_cross',
     'directed_align_mode': 'cross_ilda_dt'},
    {'joint_fusion_mode': 'smore_cross', 'train_context': 'target_mask'},
    {'joint_fusion_mode': 'smore_cross', 'score_mode': 'separate'},
    {'joint_fusion_mode': 'smore_cross', 'feat_mod_mode': 'id_cond'},
    {'fusion_beta': -1},
    {'fusion_beta': float('nan')},
    {'joint_fusion_gate': 'unknown'},
    {'joint_fusion_gate': 'smore_prefer'},
])
def test_invalid_or_confounded_settings_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        LGMRecOpt(tiny_config(tmp_path, **overrides), Loader())
