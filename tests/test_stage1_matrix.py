"""Experiment planning stays VALID-only and does not rerun unselected variants."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def script():
    path = Path(__file__).resolve().parents[1] / 'scripts/stage1_matrix.py'
    spec = importlib.util.spec_from_file_location('stage1_matrix_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(root):
    return ['stage1_matrix.py', '--output', str(root), '--learning-rate', '.0005',
            '--batch-size', '512', '--weight-decay', '1e-5']


def test_selected_matrix_and_manifest(tmp_path, monkeypatch):
    module = script()
    root = tmp_path / 'new'
    names = ['B', 'hyper20', 'hyper20_random', 'hyper20_only']
    monkeypatch.setattr(sys, 'argv', arguments(root) + ['--variants', *names, '--execute'])
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kw: commands.append(cmd))
    module.main()
    assert len(commands) == 12
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['variants'] == names
    assert manifest['protocol'] == 'train_only' and manifest['report_test'] is False
    assert manifest['commands'] == commands
    for command in commands:
        options = dict(command[i+1].split('=', 1) for i, arg in enumerate(command) if arg == '--set')
        run = Path(command[command.index('--run-dir')+1]).name
        assert options['learning_rate'] == '0.0005'
        assert options['train_batch_size'] == '512'
        assert 'report_test' not in options
        if run.startswith('B-'):
            assert 'hyper_behavior_weight' not in options
        else:
            assert options['hyper_behavior_weight'] == '0.2'
        if run.startswith('hyper20_random-'):
            assert options['hyper_behavior_graph_mode'] == 'random_relabel'
            assert 'behavior_graph_mode' not in options
        if run.startswith('hyper20_only-'):
            assert options['behavior_eta'] == '0'
    before = (root / 'manifest.json').read_bytes()
    with pytest.raises(SystemExit):
        module.main()
    assert before == (root / 'manifest.json').read_bytes()


def test_default_dry_run_and_legacy_ablations(tmp_path, monkeypatch, capsys):
    module = script()
    root = tmp_path / 'dry-run'
    monkeypatch.setattr(sys, 'argv', arguments(root))
    module.main()
    assert len(capsys.readouterr().out.splitlines()) == 6
    assert not root.exists()
    monkeypatch.setattr(sys, 'argv', arguments(root) + ['--ablations'])
    module.main()
    assert len(capsys.readouterr().out.splitlines()) == 18
    assert not root.exists()


def test_v4_variants_keep_v3_disabled(tmp_path, monkeypatch, capsys):
    module = script()
    monkeypatch.setattr(sys, 'argv', arguments(tmp_path / 'v4') +
                        ['--variants', 'B', 'edge05', 'edge10'])
    module.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 9
    for line in lines:
        if 'edge05-seed' in line:
            assert 'hyper_degree_power=0.5' in line and 'hyper_behavior_weight=0.0' in line
        elif 'edge10-seed' in line:
            assert 'hyper_degree_power=1.0' in line and 'hyper_behavior_weight=0.0' in line


def test_mechanism_matrix(tmp_path, monkeypatch):
    module = script()
    root = tmp_path / 'mechanism'
    monkeypatch.setattr(sys, 'argv', arguments(root) + ['--mechanism', '--execute'])
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kw: commands.append(cmd))
    module.main()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['variants'] == ['A', 'B', 'identity', 'random', 'item', 'user', 'no_self']
    assert len(commands) == 21
    for cmd in commands:
        options = dict(cmd[i+1].split('=', 1) for i, arg in enumerate(cmd) if arg == '--set')
        run = Path(cmd[cmd.index('--run-dir')+1]).name
        assert 'hyper_degree_power' not in options and 'hyper_behavior_weight' not in options
        if run.startswith('identity-'):
            assert options['behavior_graph_mode'] == 'identity'
        if run.startswith('no_self-'):
            assert options['behavior_self_loop'] == 'False'


@pytest.mark.parametrize('extra', [
    ['--variants', 'B', 'B'], ['--seeds', '999', '999'], ['--seeds', '-1'],
    ['--variants', 'unknown'], ['--ablations', '--variants', 'B'],
    ['--mechanism', '--ablations'], ['--mechanism', '--variants', 'B'],
])
def test_invalid_plan_rejected(tmp_path, monkeypatch, extra):
    module = script()
    root = tmp_path / 'invalid'
    monkeypatch.setattr(sys, 'argv', arguments(root) + extra)
    with pytest.raises(SystemExit):
        module.main()
    assert not root.exists()
