"""Print or execute matched VALID-only stage-one runs; never overwrite outputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', choices=['baby', 'sports', 'clothing'], default='baby')
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--seeds', nargs='+', type=int, default=[999, 2026, 2027])
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--weight-decay', type=float, required=True)
    p.add_argument('--eta', type=float, default=.2)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--minimum', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--stopping-step', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ablations', action='store_true')
    p.add_argument('--execute', action='store_true', help='Default only prints commands')
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Seeds must be unique')
    if a.output.exists():
        p.error('Output already exists; choose a fresh experiment directory')
    variants = {'A': ('LGMRec', {}), 'B': ('LGMRecOpt', {})}
    if a.ablations:
        variants.update(item=('LGMRecOpt', {'behavior_residual_target': 'item'}),
                        user=('LGMRecOpt', {'behavior_residual_target': 'user'}),
                        random=('LGMRecOpt', {'behavior_graph_mode': 'random_relabel'}),
                        zero=('LGMRecOpt', {'behavior_eta': 0}))
    commands = []
    train = Path(__file__).resolve().with_name('train.py')
    for seed in a.seeds:
        for name, (model, overrides) in variants.items():
            options = dict(learning_rate=a.learning_rate, train_batch_size=a.batch_size,
                           weight_decay=a.weight_decay, epochs=a.epochs, stopping_step=a.stopping_step)
            if model == 'LGMRecOpt':
                options.update(behavior_view_mode='msca_struct', behavior_eta=a.eta,
                               behavior_topk=a.topk, behavior_minimum=a.minimum,
                               behavior_graph_seed=seed)
            options.update(overrides)
            cmd = [sys.executable, '-B', str(train), '-m', model, '-d', a.dataset,
                   '-g', str(a.gpu), '--seed', str(seed), '--data-dir', str(a.data_dir.resolve()),
                   '--run-dir', str(a.output.resolve() / f'{name}-seed{seed}')]
            for key, value in options.items():
                cmd.extend(['--set', f'{key}={value}'])
            commands.append(cmd)
            print(shlex.join(cmd), flush=True)
    if a.execute:
        a.output.mkdir(parents=True, exist_ok=False)
        with (a.output / 'manifest.json').open('x', encoding='utf-8') as f:
            json.dump(dict(protocol='train_only', report_test=False, commands=commands), f, indent=2)
        for cmd in commands:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
