"""TRAIN-only audit contracts, including RNG and parameter preservation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.models.lgmrec import LGMRec
from test_msca_behavior import Dataset, tiny_config

spec = importlib.util.spec_from_file_location('audit_model', Path(__file__).parents[1] / 'scripts/audit_model.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_reject_incomplete_or_mismatched_source():
    config = dict(model='LGMRecOpt', dataset='baby', seed=999, protocol='train_only',
                  implementation_version='test', report_test=False, data_signature='abc')
    checkpoint = dict(config=config, **{k: config[k] for k in
                      ('model', 'dataset', 'seed', 'protocol', 'implementation_version')},
                      epoch=4, best_valid_result={'recall@20': .1}, model_state_dict={})
    status = dict(config, finished=True, returncode=0, test_evaluated=False,
                  best_epoch=4, valid_result={'recall@20': .1}, checkpoint='/run/model.pth')
    audit.validate_source(checkpoint, status, Path('model.pth'))
    with pytest.raises(ValueError, match='completed'):
        audit.validate_source(checkpoint, dict(status, finished=False), Path('model.pth'))
    with pytest.raises(ValueError, match='selection'):
        audit.validate_source(checkpoint, dict(status, best_epoch=5), Path('model.pth'))
    with pytest.raises(ValueError, match='validation-only'):
        audit.validate_source(checkpoint, dict(status, test_evaluated=True), Path('model.pth'))


def test_assignment_audit_preserves_rng_and_reveals_underflow(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, keep_rate=.5), Dataset()).eval()
    with torch.no_grad():
        model.image_embedding.weight.fill_(1.)
        model.v_hyper[:, 0].fill_(100.)
        model.v_hyper[:, 1].fill_(-100.)
    before = torch.get_rng_state().clone()
    report = audit.assignment_diagnostics(model, draws=64)
    assert report['kind'] == 'prototype'
    assert report['monte_carlo_draws'] == 64
    assert torch.equal(before, torch.get_rng_state())
    assert model.training is False
    visual = report['visual']['item']
    assert visual['eval_nonzero_support_mean'] == 1
    assert visual['eval_argmax_occupancy'] == [6, 0]
    # In finite precision, dropping the only nonzero entry erases a row.
    assert .35 < visual['train_post_dropout_exact_zero_row_fraction'] < .65
    assert audit.assignment_diagnostics(model, draws=64) == report


@pytest.mark.parametrize('uniform', [False, True])
def test_sparse_assignment_audit_uses_real_structure_and_preserves_state(tmp_path, monkeypatch, uniform):
    options = dict(hypergraph_mode='sparse_group', behavior_view_mode='msca_struct', lambda_hcl=0.)
    if uniform:
        options['hyper_group_weighting'] = 'uniform'
    model = LGMRecOpt(tiny_config(tmp_path, **options), Dataset()).eval()
    # Audit must not switch a deliberately mixed module mode permanently.
    model.drop.train()
    modes = [module.training for module in model.modules()]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    layer = model.sparse_hypergraph
    members, mask = layer.members.clone(), layer.mask.clone()
    assert not any(key.startswith('sparse_hypergraph.') for key in state)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unused prototype machinery must not be audited')

    monkeypatch.setattr(model, 'hyperedge_logits', forbidden)
    monkeypatch.setattr(model, '_hyperedge_distribution', forbidden)
    rng = torch.get_rng_state().clone()
    report = audit.assignment_diagnostics(model, draws=1)
    assert report['kind'] == 'sparse_group'
    assert report['monte_carlo_draws'] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    assert report['structure']['metadata'] == layer.metadata
    counts = mask.sum(1)
    assert report['structure']['nnz'] == int(mask.sum())
    assert report['structure']['n_hyperedges'] == len(counts)
    assert report['structure']['singleton_hyperedges'] == 2  # Catalog-only items 4 and 5.
    with torch.no_grad():
        items = model.cge()[model.n_users:]
        weights = layer.weights(items, model.image_embedding.weight @ model.item_image_trs)
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)
    visual = report['visual']
    assert visual['all_hyperedges']['entropy_mean'] == pytest.approx(float(entropy.mean()))
    for modality in ('visual', 'text'):
        data = report[modality]
        assert data['train_eval_max_abs_difference'] == 0
        assert data['train_eval_l1_mean'] == 0
        assert data['eval_padding_weight_sum'] == 0
        assert data['eval_row_sum_max_error'] < 1e-6
        assert data['non_singleton_hyperedges']['n_hyperedges'] == 4
        if uniform:
            assert data['eval_l1_from_uniform_mean'] == pytest.approx(0., abs=1e-7)
            assert data['non_singleton_hyperedges']['normalized_entropy_mean'] == pytest.approx(1.)
            assert data['all_hyperedges']['entropy_mean'] == pytest.approx(data['all_hyperedges']['uniform_entropy_mean'])
            assert data['all_hyperedges']['max_weight_mean'] == pytest.approx(data['all_hyperedges']['uniform_max_weight_mean'])
    assert torch.equal(members, layer.members)
    assert torch.equal(mask, layer.mask)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert audit.assignment_diagnostics(model) == report


def test_sparse_audit_checks_rebuilt_structure_metadata(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hypergraph_mode='sparse_group',
                                 behavior_view_mode='msca_struct', lambda_hcl=0.), Dataset())
    actual = model.sparse_hypergraph.metadata
    assert audit.sparse_structure_metadata(model, {}) == actual
    assert audit.sparse_structure_metadata(model, {'sparse_hypergraph': dict(actual)}) == actual
    changed = dict(actual, nnz=actual['nnz'] + 1)
    with pytest.raises(ValueError, match='sparse hypergraph metadata mismatch'):
        audit.sparse_structure_metadata(model, {'sparse_hypergraph': changed})


def test_sampling_excludes_all_train_positives_only():
    train = SimpleNamespace(df=pd.DataFrame({'u': [0, 0, 1], 'i': [0, 1, 2]}),
                            uid_field='u', iid_field='i', item_num=4)
    batch = audit.sample_train_batch(train, 200, 9, 'cpu')
    history = {0: {0, 1}, 1: {2}}
    for u, p, n in batch.T.tolist():
        assert p in history[u]
        assert n not in history[u]
    assert 3 in batch[2].tolist()  # Catalog items absent from TRAIN remain eligible.
    assert torch.equal(batch, audit.sample_train_batch(train, 200, 9, 'cpu'))


def test_gradient_audit_no_update_or_rng_change(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path), Dataset()).eval()
    train = SimpleNamespace(df=pd.DataFrame({'u': [0, 0, 1, 2], 'i': [0, 1, 2, 3]}),
                            uid_field='u', iid_field='i', item_num=6)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = audit.gradient_diagnostics(model, train, 8, repeats=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    assert len(report['repeats']) == 2
    assert report['repeats'][0]['parameters']['v_hyper']['weighted_hcl_norm'] >= 0
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sparse_gradient_audit_handles_zero_hcl_and_frozen_prototypes(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hypergraph_mode='sparse_group',
                                 behavior_view_mode='msca_struct', lambda_hcl=0.), Dataset()).eval()
    train = SimpleNamespace(df=pd.DataFrame({'u': [0, 0, 1, 2], 'i': [0, 1, 2, 3]}),
                            uid_field='u', iid_field='i', item_num=6)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = audit.gradient_diagnostics(model, train, 8, repeats=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    repeat = report['repeats'][0]
    assert repeat['loss_hcl_weighted'] == 0
    assert 'v_hyper' not in repeat['parameters']
    assert 't_hyper' not in repeat['parameters']
    assert repeat['parameters']['item_id_embedding.weight']['bpr_norm'] > 0
    assert all(entry['weighted_hcl_norm'] == 0 for entry in repeat['parameters'].values())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fully_dropped_anchor_has_constant_hcl_and_zero_gradient():
    # This tests the hypergraph route only: BPR still has other active branches.
    incidence = torch.randn(1, 2, requires_grad=True)
    latent = torch.randn(2, 4, requires_grad=True)
    targets = torch.randn(3, 4, requires_grad=True)
    dropped_anchor = (incidence * torch.zeros_like(incidence)) @ latent
    loss = LGMRec.ssl_triple_loss(SimpleNamespace(tau=.2), dropped_anchor,
                                 targets[:1], targets)
    torch.testing.assert_close(loss, torch.tensor(3.).log())
    gradients = torch.autograd.grad(loss, (incidence, latent, targets))
    assert all(torch.count_nonzero(value) == 0 for value in gradients)
