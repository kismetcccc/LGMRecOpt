"""Reproducible VALID-only grid search for the MSCA eta=0.2 baseline."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEARCH_SPACE = {
    'learning_rate': [5e-4, 1e-3, 2e-3],
    'train_batch_size': [512, 1024, 2048],
    'weight_decay': [0.0, 1e-6, 1e-5, 1e-4],
}
TOTAL_COMBINATIONS = math.prod(len(values) for values in SEARCH_SPACE.values())


def build_train_command(args, params, run_dir):
    """Build one isolated run using the unchanged project training entry."""
    settings = {
        'behavior_view_mode': 'msca_struct',
        'behavior_eta': 0.2,
        'joint_fusion_mode': 'off',
        'joint_fusion_gate': 'off',
        'directed_align_mode': 'off',
        'train_context': 'full',
        'score_mode': 'original',
        'epochs': args.epochs,
        'stopping_step': args.stopping_step,
        'eval_step': 1,
        'save_recommended_topk': False,
        **params,
    }
    command = [
        sys.executable, '-B', str(ROOT / 'scripts' / 'train.py'),
        '-m', 'LGMRecOpt', '-d', args.dataset, '-g', str(args.gpu),
        '--seed', str(args.seed), '--variant', 'c0',
        '--data-dir', str(args.data_dir), '--run-dir', str(run_dir),
        '--no-checkpoint',
    ]
    for key, value in settings.items():
        if isinstance(value, bool):
            value = str(value).lower()
        command.extend(('--set', f'{key}={value}'))
    return command


def load_finished_status(run_dir):
    status_path = run_dir / 'status.json'
    if not status_path.is_file():
        raise RuntimeError(f'trial did not create status.json: {run_dir}')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if not status.get('finished') or status.get('returncode') != 0:
        raise RuntimeError(f'trial did not finish successfully: {run_dir}')
    if status.get('test_evaluated') is not False:
        raise RuntimeError(f'trial violated VALID-only protocol: {run_dir}')
    result = status.get('valid_result') or {}
    if 'recall@20' not in result:
        raise RuntimeError(f'trial has no VALID recall@20: {run_dir}')
    return status


def trial_record(trial):
    return {
        'number': trial.number,
        'state': trial.state.name,
        'value': trial.value,
        'params': trial.params,
        'run_dir': trial.user_attrs.get('run_dir'),
        'best_epoch': trial.user_attrs.get('best_epoch'),
        'valid_result': trial.user_attrs.get('valid_result'),
        'data_signature': trial.user_attrs.get('data_signature'),
    }


def write_summary(study, sweep_dir, manifest):
    completed = [
        trial for trial in study.trials
        if trial.state.name == 'COMPLETE' and trial.value is not None
    ]
    ranked = sorted(completed, key=lambda trial: trial.value, reverse=True)
    top_count = max(1, math.ceil(len(ranked) * 0.2)) if ranked else 0
    top = ranked[:top_count]
    best_region = {
        key: sorted({trial.params[key] for trial in top})
        for key in SEARCH_SPACE
    } if top else {}
    output = {
        'schema_version': 1,
        'updated_at': datetime.now().isoformat(),
        'objective': 'VALID Recall@20',
        'test_evaluated': False,
        'manifest': manifest,
        'trial_count': len(study.trials),
        'completed_count': len(completed),
        'best_trial': trial_record(ranked[0]) if ranked else None,
        'best_region_top_20_percent': best_region,
        'trials': [trial_record(trial) for trial in study.trials],
    }
    temp = sweep_dir / 'summary.json.tmp'
    temp.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(sweep_dir / 'summary.json')
    study.trials_dataframe().to_csv(sweep_dir / 'trials.csv', index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='36-point VALID-only grid search for LGMRecOpt + MSCA eta=0.2'
    )
    parser.add_argument('-d', '--dataset', choices=['baby', 'sports', 'clothing'], default='baby')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--sweep-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--stopping-step', type=int, default=10)
    parser.add_argument(
        '--max-trials', type=int, default=TOTAL_COMBINATIONS,
        help='Total trials desired in this resumable study (default: full 36-point grid)',
    )
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.stopping_step < 1:
        parser.error('epochs and stopping-step must be positive')
    if not 1 <= args.max_trials <= TOTAL_COMBINATIONS:
        parser.error(f'max-trials must be between 1 and {TOTAL_COMBINATIONS}')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            'Optuna is required. Install project dependencies with: '
            'pip install -r requirements.txt'
        ) from exc

    sweep_dir = args.sweep_dir.resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = sweep_dir / 'trials'
    trials_dir.mkdir(exist_ok=True)
    manifest = {
        'dataset': args.dataset,
        'seed': args.seed,
        'data_dir': str(args.data_dir.resolve()),
        'epochs': args.epochs,
        'stopping_step': args.stopping_step,
        'search_space': SEARCH_SPACE,
        'fixed_model': {
            'model': 'LGMRecOpt',
            'variant': 'c0',
            'behavior_view_mode': 'msca_struct',
            'behavior_eta': 0.2,
            'score_mode': 'original',
            'train_context': 'full',
            'directed_align_mode': 'off',
            'joint_fusion_mode': 'off',
            'joint_fusion_gate': 'off',
        },
        'selection': 'VALID Recall@20',
        'report_test': False,
        'save_checkpoint': False,
    }
    manifest_path = sweep_dir / 'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise SystemExit(
                f'sweep directory contains a different manifest: {manifest_path}'
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
        )

    sampler = optuna.samplers.GridSampler(SEARCH_SPACE, seed=args.seed)
    database = (sweep_dir / 'study.sqlite3').as_posix()
    study = optuna.create_study(
        study_name='lgmrecopt-msca-eta02-grid',
        direction='maximize',
        sampler=sampler,
        storage=f'sqlite:///{database}',
        load_if_exists=True,
    )

    def objective(trial):
        params = {
            key: trial.suggest_categorical(key, values)
            for key, values in SEARCH_SPACE.items()
        }
        run_dir = trials_dir / f'trial-{trial.number:04d}'
        trial.set_user_attr('run_dir', str(run_dir))
        command = build_train_command(args, params, run_dir)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(
                f'training failed with exit code {result.returncode}: {run_dir}'
            )
        status = load_finished_status(run_dir)
        if status.get('model') != 'LGMRecOpt' or status.get('seed') != args.seed:
            raise RuntimeError(f'trial status identity mismatch: {run_dir}')
        signature = status.get('data_signature')
        expected = study.user_attrs.get('data_signature')
        if expected is None:
            study.set_user_attr('data_signature', signature)
        elif signature != expected:
            raise RuntimeError(f'data signature mismatch: {run_dir}')
        trial.set_user_attr('data_signature', signature)
        trial.set_user_attr('best_epoch', status.get('best_epoch'))
        trial.set_user_attr('valid_result', status['valid_result'])
        return float(status['valid_result']['recall@20'])

    def persist(study, _trial):
        write_summary(study, sweep_dir, manifest)

    remaining = args.max_trials - len(study.trials)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[persist],
            catch=(RuntimeError,),
        )
    write_summary(study, sweep_dir, manifest)
    if not any(trial.state.name == 'COMPLETE' for trial in study.trials):
        raise SystemExit('No trial completed successfully; inspect trial status files')
    best = study.best_trial
    print(f'Best VALID Recall@20: {best.value:.6f}')
    print(f'Best parameters: {best.params}')
    print(f'Summary: {sweep_dir / "summary.json"}')


if __name__ == '__main__':
    main()
