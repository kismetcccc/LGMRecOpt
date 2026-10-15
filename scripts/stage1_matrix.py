"""Print or execute matched VALID-only stage-one runs; never overwrite outputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


VARIANTS = {
    'edge05': ('LGMRecOpt', {'hyper_degree_power': .5, 'hyper_behavior_weight': 0.0}),
    'edge10': ('LGMRecOpt', {'hyper_degree_power': 1.0, 'hyper_behavior_weight': 0.0}),
    'A': ('LGMRec', {}),
    'B': ('LGMRecOpt', {}),
    'collab_hcl': ('LGMRecOpt', {'hcl_view': 'collaborative'}),
    'user_mean': ('LGMRecOpt', {'user_hyper_aggregation': 'mean'}),
    'hyper_l2': ('LGMRecOpt', {'hyper_input_norm': 'l2'}),
    'hyper_cosine': ('LGMRecOpt', {'hyper_input_norm': 'cosine'}),
    'hyper_centered': ('LGMRecOpt', {'hyper_input_norm': 'centered_l2'}),
    'centered_nohcl': ('LGMRecOpt', {'hyper_input_norm': 'centered_l2', 'lambda_hcl': 0.0}),
    'sparse_hyper': ('LGMRecOpt', {'hypergraph_mode': 'sparse_group'}),
    'sparse_nohcl': ('LGMRecOpt', {'hypergraph_mode': 'sparse_group', 'lambda_hcl': 0.0}),
    'sparse_uniform': ('LGMRecOpt', {'hypergraph_mode': 'sparse_group', 'hyper_group_weighting': 'uniform'}),
    'cosine': ('LGMRecOpt', {'behavior_neighbor_score': 'cosine'}),
    'item': ('LGMRecOpt', {'behavior_residual_target': 'item'}),
    'user': ('LGMRecOpt', {'behavior_residual_target': 'user'}),
    'random': ('LGMRecOpt', {'behavior_graph_mode': 'random_relabel'}),
    'identity': ('LGMRecOpt', {'behavior_graph_mode': 'identity'}),
    'no_self': ('LGMRecOpt', {'behavior_self_loop': False}),
    'zero': ('LGMRecOpt', {'behavior_eta': 0}),
    'hyper10': ('LGMRecOpt', {'hyper_behavior_weight': .1}),
    'hyper20': ('LGMRecOpt', {'hyper_behavior_weight': .2}),
    'hyper40': ('LGMRecOpt', {'hyper_behavior_weight': .4}),
    'hyper20_random': ('LGMRecOpt', {'hyper_behavior_weight': .2,
                                  'hyper_behavior_graph_mode': 'random_relabel'}),
    'hyper20_only': ('LGMRecOpt', {'hyper_behavior_weight': .2, 'behavior_eta': 0}),
}


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
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--ablations', action='store_true')
    selection.add_argument('--mechanism', action='store_true',
                           help='Matched A/B, identity, random, single-side and no-self-loop controls')
    selection.add_argument('--variants', nargs='+', choices=list(VARIANTS),
                           help='Run only these variants; default A B')
    p.add_argument('--execute', action='store_true', help='Default only prints commands')
    p.add_argument('--audit', action='store_true',
                   help='Export TRAIN-only LGMRecOpt diagnostics after each run; skip baseline A')
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Seeds must be unique')
    if any(seed < 0 for seed in a.seeds):
        p.error('Seeds must be non-negative')
    if a.output.exists():
        p.error('Output already exists; choose a fresh experiment directory')
    names = a.variants or (
        ['A', 'B', 'identity', 'random', 'item', 'user', 'no_self'] if a.mechanism else
        ['A', 'B', 'item', 'user', 'random', 'zero'] if a.ablations else ['A', 'B'])
    if len(set(names)) != len(names):
        p.error('Variants must be unique')
    variants = {name: VARIANTS[name] for name in names}
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
            json.dump(dict(protocol='train_only', report_test=False,
                           variants=names, seeds=a.seeds, commands=commands,
                           train_audit=a.audit,
                           train_audit_models=['LGMRecOpt'] if a.audit else []), f, indent=2)
        for cmd in commands:
            subprocess.run(cmd, check=True)
            if a.audit:
                if cmd[cmd.index('-m') + 1] != 'LGMRecOpt':
                    print('TRAIN audit skipped for baseline A (LGMRec); continuing matrix.', flush=True)
                    continue
                run = Path(cmd[cmd.index('--run-dir') + 1])
                status_path = run / 'status.json'
                status = json.loads(status_path.read_text(encoding='utf-8'))
                subprocess.run([
                    sys.executable, '-B', str(train.with_name('audit_model.py')),
                    '--checkpoint', status['checkpoint'], '--status', str(status_path),
                    '--output', str(run / 'train_audit.json'),
                    '--data-dir', str(a.data_dir.resolve()),
                ], check=True)


if __name__ == '__main__':
    main()
