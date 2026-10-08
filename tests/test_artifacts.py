"""Public CLI artifact contracts, without using real recommendation data."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('behavior,hyper_weight', [('off', 0), ('msca_struct', 0), ('msca_struct', .2)])
def test_cli_writes_result_and_refuses_existing_directory(tmp_path, behavior, hyper_weight):
    data = tmp_path / 'data' / 'baby'
    data.mkdir(parents=True)
    for name in ('image_feat.npy', 'text_feat.npy'):
        np.save(data / name, np.random.default_rng(5).normal(size=(6,4)).astype('float32'))
    pd.DataFrame({'userID':[0,0,1,1,0,1,0,1], 'itemID':[0,1,1,2,3,4,5,5],
                  'timestamp':[1,2,1,2,3,3,4,4], 'x_label':[0,0,0,0,1,1,2,2]}).to_csv(
        data / 'baby.inter', index=False, sep='\t')
    run = tmp_path / 'run'
    command = [sys.executable, '-B', str(ROOT/'scripts/train.py'), '--data-dir', str(data.parent),
               '--run-dir', str(run), '--set', 'epochs=1', '--set', 'use_gpu=false',
               '--set', 'topk=[1,2]', '--set', 'valid_metric=Recall@2',
               '--set', 'save_recommended_topk=false',
               '--set', f'behavior_view_mode={behavior}',
               '--set', f'hyper_behavior_weight={hyper_weight}',
               '--set', 'behavior_minimum=1']
    first = subprocess.run(command, capture_output=True)
    assert first.returncode == 0, first.stderr.decode(errors='replace')
    status_bytes = (run/'status.json').read_bytes()
    status = json.loads(status_bytes)
    assert status['finished'] and not status['test_evaluated']
    # YAML 1.1 parses an unquoted CLI 'off' as False; the model accepts both.
    assert status['effective_config']['behavior_view_mode'] == (False if behavior == 'off' else behavior)
    assert status['effective_config']['protocol'] == 'train_only'
    if hyper_weight:
        assert status['hyper_behavior_graph']['weight'] == hyper_weight
        assert status['hyper_behavior_graph']['items_with_neighbors'] > 0
    else:
        assert status['hyper_behavior_graph'] is None
    if behavior == 'msca_struct':
        assert len(status['behavior_graph']['fingerprint']) == 64
    else:
        assert status['behavior_graph'] is None
    assert Path(status['checkpoint']).is_file()
    assert list((run/'log').glob('*.log'))
    second = subprocess.run(command, capture_output=True)
    assert second.returncode != 0
    assert (run/'status.json').read_bytes() == status_bytes
    summary = tmp_path/'summary.json'
    result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/summarize.py'), str(run),
                             '--out', str(summary)], capture_output=True)
    assert result.returncode == 0
    assert len(json.loads(summary.read_text())) == 1

    log_only_run = tmp_path / 'log-only-run'
    log_only_command = command.copy()
    log_only_command[log_only_command.index(str(run))] = str(log_only_run)
    log_only_command.append('--no-checkpoint')
    log_only = subprocess.run(log_only_command, capture_output=True)
    assert log_only.returncode == 0, log_only.stderr.decode(errors='replace')
    log_only_status = json.loads(
        (log_only_run / 'status.json').read_text(encoding='utf-8')
    )
    assert log_only_status['finished']
    assert log_only_status['checkpoint_enabled'] is False
    assert log_only_status['checkpoint'] is None
    assert not (log_only_run / 'checkpoints').exists()
    assert list((log_only_run / 'log').glob('*.log'))
