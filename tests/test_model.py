"""Numerical contracts for the maintained C0 and optional v11 candidate."""
import os
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
        return sp.coo_matrix((np.ones(5), ([0, 0, 1, 1, 2], [0, 1, 1, 2, 3])),
                             shape=(3, 5)).asformat(form)


def tiny_config(tmp_path, **overrides):
    directory = tmp_path / 'baby'
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for name, width in [('image_feat.npy', 6), ('text_feat.npy', 4)]:
        np.save(directory / name, rng.normal(size=(5, width)).astype('float32'))
    values = dict(data_path=str(tmp_path) + os.sep, use_gpu=False, seed=999,
                  embedding_size=8, feat_embed_dim=8, n_ui_layers=1,
                  n_mm_layers=1, n_hyper_layer=1, hyper_num=2, keep_rate=.5,
                  alpha=.1, cl_weight=1e-4, reg_weight=1e-6)
    values.update(overrides)
    return Config('LGMRecOpt', 'baby', values)


@pytest.mark.parametrize('mode', ['off', 'id_cond'])
def test_initial_state_loss_gradient_and_rng(tmp_path, mode):
    cfg = tiny_config(tmp_path, feat_mod_mode=mode)
    torch.manual_seed(19)
    base = LGMRec(cfg, Loader())
    rng = torch.get_rng_state()
    torch.manual_seed(19)
    candidate = LGMRecOpt(cfg, Loader())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    batch = torch.tensor([[0, 1], [0, 2], [3, 4]])
    torch.manual_seed(23)
    left = base.calculate_loss(batch)
    left.backward()
    rng = torch.get_rng_state()
    torch.manual_seed(23)
    right = candidate.calculate_loss(batch)
    right.backward()
    assert torch.equal(left, right)
    assert torch.equal(rng, torch.get_rng_state())
    for key, p in base.named_parameters():
        other = dict(candidate.named_parameters())[key]
        if p.grad is not None:
            assert torch.equal(p.grad, other.grad), key
    if mode == 'id_cond':
        assert any(p.grad.abs().sum() > 0 for p in candidate.feat_mod.parameters())


def test_eval_is_deterministic(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path), Loader()).eval()
    rng = torch.get_rng_state()
    a = model.full_sort_predict(torch.tensor([[0, 1]]))
    b = model.full_sort_predict(torch.tensor([[0, 1]]))
    assert torch.equal(a, b)
    assert torch.equal(rng, torch.get_rng_state())


def test_hcl_zero_still_computes_raw_loss(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, lambda_hcl=0), Loader())
    loss = model.calculate_loss(torch.tensor([[0, 1], [0, 2], [3, 4]]))
    assert torch.isfinite(loss)
    assert model._sums['loss_hcl_raw'] > 0
    assert model._sums['loss_hcl_weighted'] == 0


@pytest.mark.parametrize('override', [dict(protocol='legacy'), dict(modal_graph_mode='raw'),
    dict(hcl_deweight=True), dict(seed=[999, 2026]), dict(ablation='no_selection')])
def test_retired_and_unsafe_settings_rejected(override):
    with pytest.raises(ValueError):
        Config('LGMRecOpt', 'baby', override)


def test_defaults_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config('LGMRecOpt', 'baby')
    assert cfg['n_ui_layers'] == [4]
    assert cfg['feat_mod_mode'] == 'off'
    assert cfg['report_test'] is False
    assert LGMRecOpt.__bases__ == (LGMRec,)


@pytest.mark.parametrize('mode', ['off', 'id_cond'])
def test_archived_v11_weights_and_forward_remain_equivalent(tmp_path, monkeypatch, mode):
    """Compare against preserved original implementation, including nonzero gates."""
    from pathlib import Path
    import importlib
    legacy = Path(__file__).resolve().parents[1] / 'references/legacy/v11_workspace'
    if not legacy.is_dir():
        pytest.skip('Optional local archive is not shipped in the clean distribution')
    monkeypatch.syspath_prepend(str(legacy))
    original = importlib.import_module('models.lgmrecopt')
    cfg = tiny_config(tmp_path, feat_mod_mode=mode)
    archived_cfg = dict(cfg.final_config_dict,
                        implementation_version=original.IMPLEMENTATION_VERSION,
                        enhancement_mode='modal_item_graph', ablation='no_enhancement',
                        modal_graph_mode='off', mge_layer_agg='last')
    torch.manual_seed(29)
    old = original.LGMRecOpt(archived_cfg, Loader())
    if mode == 'id_cond':
        with torch.no_grad():
            for p in old.feat_mod.parameters():
                p.uniform_(-.2, .2)
    new = LGMRecOpt(cfg, Loader())
    new.load_state_dict(old.state_dict(), strict=True)
    batch = torch.tensor([[0,1], [0,2], [3,4]])
    old.eval(); new.eval()
    assert torch.equal(old.full_sort_predict(batch[:1]), new.full_sort_predict(batch[:1]))
    old.train(); new.train()
    torch.manual_seed(31)
    left = old.calculate_loss(batch)
    left.backward()
    torch.manual_seed(31)
    right = new.calculate_loss(batch)
    right.backward()
    assert torch.equal(left, right)
    for key, value in old.named_parameters():
        if value.grad is not None:
            assert torch.equal(value.grad, dict(new.named_parameters())[key].grad), key
