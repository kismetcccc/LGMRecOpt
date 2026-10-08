"""Safety and numerical contracts for scripts/score_probe.py."""
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from lgmrec.models.lgmrec import LGMRec
from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.utils.configurator import Config
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'score_probe', ROOT / 'scripts' / 'score_probe.py'
)
score_probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_probe)

CROSS_STATE_KEYS = score_probe.CROSS_STATE_KEYS
_cache_representations = score_probe._cache_representations
_cached_score = score_probe._cached_score
_freeze_for_cross = score_probe._freeze_for_cross
_load_c0_state = score_probe._load_c0_state
_reproduction_check = score_probe._reproduction_check
_validate_source = score_probe._validate_source


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


def tiny_config(tmp_path, model='LGMRecOpt', **overrides):
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
    )
    values.update(overrides)
    return Config(model, 'baby', values)


def models(tmp_path):
    cfg = tiny_config(tmp_path, score_mode='cross')
    torch.manual_seed(11)
    source = LGMRec(tiny_config(tmp_path, model='LGMRec'), Loader())
    torch.manual_seed(11)
    cross = LGMRecOpt(cfg, Loader())
    return source, cross


def test_checkpoint_load_allows_only_six_missing_cross_logits(tmp_path):
    source, cross = models(tmp_path)
    state = source.state_dict()
    _load_c0_state(cross, state)
    assert CROSS_STATE_KEYS == {
        name for name in cross.state_dict() if name.startswith('cross_logits.')
    }

    missing_encoder = dict(state)
    missing_encoder.pop('user_embedding.weight')
    with pytest.raises(ValueError, match='exactly the six'):
        _load_c0_state(cross, missing_encoder)

    unexpected = dict(state, unexpected=torch.zeros(1))
    with pytest.raises(ValueError, match='Unexpected'):
        _load_c0_state(cross, unexpected)

    wrong_shape = dict(state)
    wrong_shape['user_embedding.weight'] = state['user_embedding.weight'][:1]
    with pytest.raises(ValueError, match='shape mismatch'):
        _load_c0_state(cross, wrong_shape)


def test_frozen_cached_training_updates_only_cross_logits_and_stays_eval(tmp_path):
    source, model = models(tmp_path)
    _load_c0_state(model, source.state_dict())
    trainable = _freeze_for_cross(model)
    cache = _cache_representations(model)
    before = {name: value.detach().clone()
              for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(
        list(model.cross_logits.values()), lr=.1
    )
    users = torch.tensor([0, 1])
    positive = torch.tensor([0, 2])
    negative = torch.tensor([3, 4])
    optimizer.zero_grad(set_to_none=True)
    loss = -F.logsigmoid(
        _cached_score(model, cache, users, positive)
        - _cached_score(model, cache, users, negative)
    ).mean()
    loss.backward()
    optimizer.step()
    changed = {
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    }
    assert not model.training
    assert sorted(trainable) == sorted(CROSS_STATE_KEYS)
    assert changed
    assert changed <= CROSS_STATE_KEYS
    assert all(not tensor.requires_grad
               for key, value in cache.items()
               for tensor in (value if isinstance(value, tuple) else (value,)))


def test_source_validation_rejects_scoring_smoke_and_metadata_mismatch(tmp_path):
    checkpoint_path = tmp_path / 'run' / 'checkpoints' / 'c0.pth'
    checkpoint_path.parent.mkdir(parents=True)
    config = {
        'model': 'LGMRecOpt', 'dataset': 'baby', 'seed': 999,
        'protocol': 'train_only', 'implementation_version': 'v1',
        'feat_mod_mode': 'off', 'ablation': None, 'report_test': False,
        'data_signature': 'abc',
    }
    checkpoint = {
        **{key: config[key] for key in
           ('model', 'dataset', 'seed', 'protocol', 'implementation_version')},
        'epoch': 36,
        'config': config,
        'test_result': None,
        'best_valid_result': {'recall@20': .1},
    }
    status = {
        'finished': True, 'model': 'LGMRecOpt', 'dataset': 'baby',
        'seed': 999, 'best_epoch': 36, 'implementation_version': 'v1',
        'test_evaluated': False, 'variant': 'c0',
        'checkpoint': str(checkpoint_path), 'data_signature': 'abc',
        'valid_result': {'recall@20': .1}, 'options': {},
    }
    _validate_source(checkpoint, checkpoint_path, status)
    status['options']['score_mode'] = 'original'
    with pytest.raises(ValueError, match='smoke/scoring'):
        _validate_source(checkpoint, checkpoint_path, status)
    status['options'].clear()
    checkpoint['seed'] = 7
    with pytest.raises(ValueError, match='mismatch'):
        _validate_source(checkpoint, checkpoint_path, status)


def test_reproduction_check_is_strict_by_default():
    assert _reproduction_check({'recall@20': .1}, {'recall@20': .1}, 0)['passed']
    result = _reproduction_check(
        {'recall@20': .1001}, {'recall@20': .1}, 0
    )
    assert not result['passed']
    assert result['max_absolute_delta'] == pytest.approx(.0001)


def test_probe_end_to_end_is_valid_only_and_refuses_overwrite(tmp_path):
    data_dir = tmp_path / 'data'
    dataset_dir = data_dir / 'baby'
    dataset_dir.mkdir(parents=True)
    rng = np.random.default_rng(17)
    np.save(dataset_dir / 'image_feat.npy', rng.normal(size=(60, 6)).astype('float32'))
    np.save(dataset_dir / 'text_feat.npy', rng.normal(size=(60, 4)).astype('float32'))
    rows = []
    timestamp = 0
    for item in range(58):
        rows.append((item % 2, item, timestamp, 0))
        timestamp += 1
    rows.extend([(0, 58, timestamp, 1), (1, 59, timestamp + 1, 1),
                 (0, 59, timestamp + 2, 2), (1, 58, timestamp + 3, 2)])
    pd.DataFrame(
        rows, columns=['userID', 'itemID', 'timestamp', 'x_label']
    ).to_csv(dataset_dir / 'baby.inter', sep='\t', index=False)

    source_run = tmp_path / 'complete-c0'
    checkpoint_dir = source_run / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    config = Config('LGMRecOpt', 'baby', {
        'data_path': str(data_dir) + os.sep,
        'use_gpu': False,
        'seed': 999,
        'embedding_size': 8,
        'feat_embed_dim': 8,
        'n_ui_layers': 1,
        'n_mm_layers': 1,
        'n_hyper_layer': 1,
        'hyper_num': 2,
        'keep_rate': 1.,
        'alpha': .1,
        'cl_weight': 1e-4,
        'reg_weight': 1e-6,
        'lambda_hcl': None,
        'topk': [20, 50],
        'valid_metric': 'Recall@20',
        'train_batch_size': 16,
        'eval_batch_size': 2,
        'score_mode': 'original',
        'save_recommended_topk': False,
    })
    input_hashes, signature = score_probe.fingerprint_inputs(config)
    config['input_hashes'] = input_hashes
    config['data_signature'] = signature
    train_data, valid_data = score_probe._make_loaders(config)
    torch.manual_seed(23)
    source_model = LGMRecOpt(config, train_data).eval()
    source_cache = _cache_representations(source_model)
    recorded = score_probe._evaluate_valid(
        source_model, source_cache, valid_data,
        score_probe.TopKEvaluator(config), 'original'
    )

    source_config = score_probe._json_value(config.final_config_dict)
    source_config.pop('score_mode')
    checkpoint_path = checkpoint_dir / 'complete-c0.pth'
    checkpoint = {
        'model_state_dict': source_model.state_dict(),
        'epoch': 0,
        'best_valid_score': recorded['recall@20'],
        'best_valid_result': recorded,
        'test_result': None,
        'config': source_config,
        'model': 'LGMRecOpt',
        'dataset': 'baby',
        'seed': 999,
        'protocol': 'train_only',
        'evaluation_version': source_config['evaluation_version'],
        'search_version': source_config['search_version'],
        'implementation_version': source_config['implementation_version'],
        'ablation': None,
    }
    torch.save(score_probe._checkpoint_safe_values(checkpoint), checkpoint_path)
    (source_run / 'status.json').write_text(score_probe.json.dumps({
        'finished': True,
        'model': 'LGMRecOpt',
        'dataset': 'baby',
        'seed': 999,
        'variant': 'c0',
        'options': {},
        'best_epoch': 0,
        'implementation_version': source_config['implementation_version'],
        'test_evaluated': False,
        'checkpoint': str(checkpoint_path),
        'data_signature': signature,
        'valid_result': recorded,
    }), encoding='utf-8')

    output = tmp_path / 'probe-output'
    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        source_status=None,
        data_dir=data_dir,
        run_dir=output,
        gpu=0,
        cpu=True,
        epochs=1,
        learning_rate=.01,
        patience=1,
        reproduction_tolerance=0.,
    )
    result = score_probe.run(args)
    assert result['finished']
    assert result['test_evaluated'] is False
    assert result['original_reproduction']['passed']
    assert result['freeze_check']['trainable_scalar_count'] == 6
    assert (output / 'status.json').is_file()
    assert (output / 'best_cross_logits.pth').is_file()
    with pytest.raises(ValueError, match='already exists'):
        score_probe.run(args)
