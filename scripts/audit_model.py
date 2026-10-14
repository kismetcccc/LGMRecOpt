"""Read-only TRAIN diagnostics for an explicitly identified, completed checkpoint.

No ranking evaluation, parameter update, or validation/test loader is created.
Historical checkpoints remain historical evidence; this does not establish gains.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lgmrec.models.lgmrecopt import LGMRecOpt
from lgmrec.utils.configurator import Config
from lgmrec.utils.dataloader import TrainDataLoader
from lgmrec.utils.dataset import RecDataset
from lgmrec.utils.experiment_state import fingerprint_inputs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source(checkpoint, status, path):
    cfg = checkpoint['config']
    effective = status.get('effective_config', status.get('options', {}))
    for key in ('model', 'dataset', 'seed', 'protocol', 'implementation_version'):
        recorded = status.get(key, effective.get(key))
        if checkpoint.get(key) != cfg.get(key) or recorded != cfg.get(key):
            raise ValueError(f'Source metadata mismatch: {key}')
    for key, value in effective.items():
        if key in cfg and cfg[key] != value:
            raise ValueError(f'Source effective configuration mismatch: {key}')
    if cfg['model'] != 'LGMRecOpt' or cfg['protocol'] != 'train_only':
        raise ValueError('Expected TRAIN-only LGMRecOpt checkpoint')
    if (cfg.get('report_test') is not False or checkpoint.get('test_result') is not None
            or status.get('test_evaluated') is not False):
        raise ValueError('Expected validation-only source')
    if status.get('finished') is not True or status.get('returncode') != 0:
        raise ValueError('Source run must be completed successfully')
    if (status.get('best_epoch') != checkpoint.get('epoch')
            or status.get('valid_result') != checkpoint.get('best_valid_result')):
        raise ValueError('Source checkpoint does not match recorded selection')
    if Path(status.get('checkpoint', '')).name != Path(path).name:
        raise ValueError('Source checkpoint filename mismatch')
    if status.get('data_signature') != cfg.get('data_signature'):
        raise ValueError('Source data signature mismatch')
    # The direct BPR diagnostic below is the original dot-product scoring rule.
    if cfg.get('score_mode', 'original') != 'original' or cfg.get('train_context', 'full') != 'full':
        raise ValueError('Audit requires original scoring and full TRAIN context')
    for key, value in checkpoint['model_state_dict'].items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint tensor: {key}')


def load_source(checkpoint_path, status_path, data_dir, use_gpu=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    status = json.loads(Path(status_path).read_text(encoding='utf-8'))
    validate_source(checkpoint, status, checkpoint_path)
    saved = checkpoint['config']
    defaults = Config('LGMRecOpt', saved['dataset']).final_config_dict
    metadata = {'input_hashes', 'data_signature'}
    unknown = set(saved) - set(defaults) - metadata
    if unknown:
        raise ValueError(f'Unknown source configuration keys: {sorted(unknown)}')
    values = {key: value for key, value in saved.items()
              if key not in metadata | {'device', 'valid_metric_bigger'}}
    values.update(data_path=str(Path(data_dir).resolve()) + os.sep,
                  use_gpu=use_gpu, gpu_id=0)
    config = Config('LGMRecOpt', saved['dataset'], values)
    # This tool never evaluates rankings. The runtime may use a newer metric
    # precision version; preserve the source version as provenance instead.
    runtime = {'data_path', 'use_gpu', 'gpu_id', 'device', 'evaluation_version'}
    for key in set(saved) - metadata - runtime:
        if config[key] != saved[key]:
            raise ValueError(f'Source configuration changed: {key}')
    hashes, signature = fingerprint_inputs(config)
    if hashes != saved.get('input_hashes') or signature != saved.get('data_signature'):
        raise ValueError('Input fingerprints do not match checkpoint')
    dataset = RecDataset(config)
    # Only the split indicator is used to select TRAIN. No held-out pairs are
    # passed to sampling, graph construction, model losses, or diagnostics.
    frame = dataset.df[dataset.df[dataset.splitting_label] == 0].copy()
    frame.drop(columns=[dataset.splitting_label], inplace=True)
    train = dataset.copy(frame)
    loader = TrainDataLoader(config, train, batch_size=config['train_batch_size'])
    devices = [config['device'].index or 0] if config['device'].type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model = LGMRecOpt(config, loader).to(config['device'])
    state = checkpoint['model_state_dict']
    for key in ('image_embedding.weight', 'text_embedding.weight'):
        if not torch.equal(model.state_dict()[key].cpu(), state[key]):
            raise ValueError(f'Frozen features differ from source data: {key}')
    model.load_state_dict(state, strict=True)
    provenance = {
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'checkpoint_sha256': sha256(checkpoint_path),
        'status': str(Path(status_path).resolve()),
        'status_sha256': sha256(status_path),
        'implementation_version': saved['implementation_version'],
        'source_evaluation_version': saved['evaluation_version'],
        'data_signature': signature, 'seed': saved['seed'], 'epoch': checkpoint['epoch'],
        'source_valid_result': checkpoint['best_valid_result'],
        'source_training': {k: saved[k] for k in
                            ('learning_rate', 'train_batch_size', 'weight_decay')},
        'compatibility_defaults': {k: v for k, v in defaults.items() if k not in saved},
    }
    return model, train, provenance


def quantiles(values):
    points = torch.tensor([0., .5, .9, .99, 1.], device=values.device)
    result = torch.quantile(values.float().flatten(), points).cpu().tolist()
    return dict(zip(('min', 'median', 'p90', 'p99', 'max'), result))


@torch.no_grad()
def assignment_diagnostics(model, draws=128, seed=731):
    if draws < 2:
        raise ValueError('draws must be at least 2')
    device = model.v_hyper.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    was_training = model.training
    report = {}
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for modality, features, projection in (
                ('visual', model.image_embedding.weight, model.v_hyper),
                ('text', model.text_embedding.weight, model.t_hyper),
            ):
                item_logits = model.hyperedge_logits(features, projection)
                report[modality] = {'feature_norm': quantiles(features.norm(dim=1)),
                                    'projection_norm': float(projection.norm())}
                for side, logits in (('item', item_logits),
                                     ('user', model.user_hyperedge_logits(item_logits))):
                    model.eval()
                    expected_eval = model._hyperedge_distribution(logits)
                    counts = torch.bincount(expected_eval.argmax(1), minlength=logits.shape[1])
                    total = torch.zeros_like(logits)
                    half = torch.zeros_like(logits)
                    mismatch = zero_rows = 0.
                    model.train()
                    for draw in range(draws):
                        sample = model._hyperedge_distribution(logits)
                        total += sample
                        if draw < draws // 2:
                            half += sample
                        mismatch += float((sample.argmax(1) != expected_eval.argmax(1)).float().mean())
                        zero_rows += float((model.drop(sample).abs().sum(1) == 0).float().mean())
                    entropy = -(expected_eval * expected_eval.clamp_min(1e-30).log()).sum(1)
                    report[modality][side] = {
                        'logit_span': quantiles(logits.max(1).values - logits.min(1).values),
                        'eval_entropy_mean': float(entropy.mean()),
                        'eval_max_probability_mean': float(expected_eval.max(1).values.mean()),
                        'eval_max_probability_above_099_fraction': float((expected_eval.max(1).values > .99).float().mean()),
                        'eval_argmax_occupancy': counts.cpu().tolist(),
                        'eval_nonzero_support_mean': float((expected_eval > 0).float().sum(1).mean()),
                        'train_mc_mean_l1_from_eval': float((total / draws - expected_eval).abs().sum(1).mean()),
                        'train_mc_split_half_l1': float((half / (draws // 2) - (total - half) / (draws - draws // 2)).abs().sum(1).mean()),
                        'train_argmax_mismatch_fraction': mismatch / draws,
                        'train_post_dropout_exact_zero_row_fraction': zero_rows / draws,
                    }
    finally:
        model.train(was_training)
    return report


def sample_train_batch(train, size, seed, device):
    rng = random.Random(seed)
    pairs = list(train.df[[train.uid_field, train.iid_field]].itertuples(index=False, name=None))
    history = {}
    for user, item in pairs:
        history.setdefault(user, set()).add(item)
    if not pairs or any(len(items) >= train.item_num for items in history.values()):
        raise ValueError('No valid TRAIN-only sampling distribution')
    rows = []
    for _ in range(size):
        user, positive = pairs[rng.randrange(len(pairs))]
        negative = rng.randrange(train.item_num)
        while negative in history[user]:
            negative = rng.randrange(train.item_num)
        rows.append((user, positive, negative))
    return torch.tensor(rows, dtype=torch.long, device=device).T


def gradient_diagnostics(model, train, batch_size, repeats=3, seed=731):
    """Multiple stochastic forwards, same fixed TRAIN triplets, no optimizer."""
    device = model.v_hyper.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    batch = sample_train_batch(train, batch_size, seed, device)
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named]
    output = []
    was_training = model.training
    try:
        model.train()
        with torch.random.fork_rng(devices=devices):
            for repeat in range(repeats):
                torch.manual_seed(seed + repeat)
                users, items, hyper = model.forward()
                uid, pos, neg = batch
                bpr = model.bpr_loss(users[uid], items[pos], items[neg])
                hcl = model.lambda_hcl * model.hypergraph_contrastive_loss(hyper, uid, pos)
                left = torch.autograd.grad(bpr, parameters, retain_graph=True, allow_unused=True)
                right = torch.autograd.grad(hcl, parameters, allow_unused=True)
                gradients = {}
                for (name, parameter), a, b in zip(named, left, right):
                    a = torch.zeros_like(parameter) if a is None else a
                    b = torch.zeros_like(parameter) if b is None else b
                    na, nb = float(a.norm()), float(b.norm())
                    gradients[name] = {
                        'bpr_norm': na, 'weighted_hcl_norm': nb,
                        'weighted_hcl_to_bpr_norm_ratio': nb / na if na > 0 else None,
                        'cosine': float((a * b).sum()) / (na * nb) if na * nb > 0 else None,
                    }
                output.append({'repeat': repeat, 'loss_bpr': float(bpr.detach()),
                               'loss_hcl_weighted': float(hcl.detach()), 'parameters': gradients})
    finally:
        model.train(was_training)
    return {'fixed_train_batch_size': batch_size, 'seed': seed, 'repeats': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--gradient-repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.gradient_repeats < 1:
        parser.error('--gradient-repeats must be positive')
    torch.set_num_threads(2)
    model, train, source = load_source(args.checkpoint, args.status, args.data_dir, args.gpu)
    result = {
        'source': source, 'test_evaluated': False, 'ranking_evaluated': False,
        'interpretation': 'Frozen source-checkpoint TRAIN diagnostics; not proof of a performance defect or improvement.',
        'monte_carlo_draws': args.draws,
        'assignments': assignment_diagnostics(model, draws=args.draws),
        'gradients': gradient_diagnostics(model, train, source['source_training']['train_batch_size'],
                                          repeats=args.gradient_repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
