import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'tune_msca', ROOT / 'scripts' / 'tune_msca.py'
)
TUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TUNE)
SEARCH_SPACE = TUNE.SEARCH_SPACE
TOTAL_COMBINATIONS = TUNE.TOTAL_COMBINATIONS
build_train_command = TUNE.build_train_command
load_finished_status = TUNE.load_finished_status


def test_search_space_is_complete_36_point_grid():
    assert TOTAL_COMBINATIONS == 36
    assert 0.001 in SEARCH_SPACE['learning_rate']
    assert 2048 in SEARCH_SPACE['train_batch_size']
    assert 0.0 in SEARCH_SPACE['weight_decay']


def test_train_command_fixes_msca_baseline_and_disables_checkpoints(tmp_path):
    args = argparse.Namespace(
        dataset='baby', gpu=0, seed=999, data_dir=Path('data'),
        epochs=50, stopping_step=10,
    )
    params = {
        'learning_rate': 0.001,
        'train_batch_size': 2048,
        'weight_decay': 0.0,
    }
    command = build_train_command(args, params, tmp_path / 'trial')
    joined = ' '.join(map(str, command))
    assert '--no-checkpoint' in command
    assert 'behavior_view_mode=msca_struct' in joined
    assert 'behavior_eta=0.2' in joined
    assert 'joint_fusion_mode=off' in joined
    assert 'directed_align_mode=off' in joined
    assert 'score_mode=original' in joined
    assert 'learning_rate=0.001' in joined
    assert 'train_batch_size=2048' in joined
    assert 'weight_decay=0.0' in joined


def test_status_reader_requires_finished_valid_only_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    status = {
        'finished': True,
        'returncode': 0,
        'test_evaluated': False,
        'valid_result': {'recall@20': 0.1},
    }
    (run / 'status.json').write_text(json.dumps(status), encoding='utf-8')
    assert load_finished_status(run)['valid_result']['recall@20'] == 0.1
    status['test_evaluated'] = True
    (run / 'status.json').write_text(json.dumps(status), encoding='utf-8')
    with pytest.raises(RuntimeError, match='VALID-only'):
        load_finished_status(run)
