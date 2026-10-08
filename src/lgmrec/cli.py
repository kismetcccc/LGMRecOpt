"""One training entry point with isolated artifacts and machine-readable status."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import traceback
from uuid import uuid4
import yaml

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '48')


def parse_overrides(entries):
    result = {}
    for entry in entries:
        key, sep, value = entry.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not value.strip():
            raise ValueError(f'Expected KEY=VALUE: {entry}')
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str) and re.fullmatch(r'[-+]?[\d.]+[eE][-+]?\d+', parsed):
            parsed = float(parsed)
        result[key] = parsed
    return result


def main():
    parser = argparse.ArgumentParser(description='LGMRec: TRAIN-only learning, VALID selection')
    parser.add_argument('-m', '--model', choices=['LGMRec', 'LGMRecOpt'], default='LGMRecOpt')
    parser.add_argument('-d', '--dataset', choices=['baby', 'sports', 'clothing'], default='baby')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--variant', choices=['c0', 'id_cond'], default='c0')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--run-dir', type=Path, help='Must not exist; defaults to a unique data/runs directory')
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help='Keep the best VALID state in memory but do not write a .pth file',
    )
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    if args.model == 'LGMRec' and args.variant != 'c0':
        parser.error('id_cond requires LGMRecOpt')
    try:
        options = parse_overrides(args.set)
    except (ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    reserved = {'model', 'dataset', 'seed', 'gpu_id', 'data_path', 'protocol',
                'report_test', 'checkpoint_dir', 'log_dir', 'recommend_topk',
                'feat_mod_mode', 'ablation', 'implementation_version'}
    if reserved.intersection(options):
        parser.error('Use named CLI options; protected overrides: ' + ', '.join(sorted(reserved.intersection(options))))
    run = args.run_dir or Path('data/runs') / (
        f'{datetime.now():%Y%m%d-%H%M%S}-{args.variant}-seed{args.seed}-{uuid4().hex[:8]}')
    run = run.resolve()
    options.update(seed=args.seed, gpu_id=args.gpu, protocol='train_only', report_test=False,
                   data_path=str(args.data_dir.resolve()) + os.sep,
                   checkpoint_dir=str(run / 'checkpoints'), log_dir=str(run / 'log'),
                   recommend_topk=str(run / 'recommendations') + os.sep)
    if args.model == 'LGMRecOpt':
        options['feat_mod_mode'] = 'off' if args.variant == 'c0' else 'id_cond'
    from lgmrec.utils.configurator import Config
    from lgmrec.utils.quick_start import quick_start
    # Reject misspelled/retired options before making an output directory.
    try:
        Config(args.model, args.dataset, options)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        run.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Run directory already exists; preserved unchanged: {run}')
    status = dict(schema_version=1, model=args.model, dataset=args.dataset,
                  seed=args.seed, variant=args.variant, options=options,
                  checkpoint_enabled=not args.no_checkpoint,
                  started_at=datetime.now().isoformat(), finished=False)
    def save_status():
        temp = run / 'status.json.tmp'
        temp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(run / 'status.json')
    save_status()
    print(f'Artifacts: {run}', flush=True)
    try:
        result = quick_start(
            args.model, args.dataset, options,
            save_model=not args.no_checkpoint,
        )
        status.update(result, finished=True, returncode=0)
    except Exception:
        status.update(returncode=1, error=traceback.format_exc())
        raise
    finally:
        status['ended_at'] = datetime.now().isoformat()
        save_status()


if __name__ == '__main__':
    main()
