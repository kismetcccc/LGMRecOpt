"""Frozen-representation scoring probe for a completed C0 checkpoint.

This is deliberately separate from the normal training entry point. It loads
one completed, pre-scoring C0 run, evaluates ORIGINAL and SEPARATE on VALID,
then trains only the six CROSS logits with TRAIN pairs and selects on VALID.
TEST is never wrapped in an evaluation loader or evaluated here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lgmrec.common.trainer import _checkpoint_safe_values
from lgmrec.models.lgmrecopt import CROSS_BRANCH_PAIRS, LGMRecOpt
from lgmrec.utils.configurator import Config
from lgmrec.utils.dataloader import EvalDataLoader, TrainDataLoader
from lgmrec.utils.dataset import RecDataset
from lgmrec.utils.experiment_state import fingerprint_inputs
from lgmrec.utils.quick_start import _prepare_train_only_protocol
from lgmrec.utils.topk_evaluator import TopKEvaluator
from lgmrec.utils.utils import init_seed


CROSS_STATE_KEYS = {
    f'cross_logits.{left}_{right}' for left, right in CROSS_BRANCH_PAIRS
}
RUNTIME_CONFIG_KEYS = {
    'data_path', 'device', 'gpu_id', 'use_gpu', 'checkpoint_dir', 'log_dir',
    'recommend_topk', 'save_recommended_topk', 'score_mode',
}
CHECKPOINT_METADATA_KEYS = {'input_hashes', 'data_signature'}
COMPUTED_CONFIG_KEYS = {'valid_metric_bigger'}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    return _checkpoint_safe_values(value)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    temporary.replace(path)


def _load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError as exc:  # pragma: no cover - supported torch is >=2.2
        raise RuntimeError('PyTorch with weights_only checkpoint loading is required') from exc
    required = {
        'model_state_dict', 'epoch', 'best_valid_result', 'config', 'model',
        'dataset', 'seed', 'protocol', 'implementation_version',
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f'Checkpoint metadata missing: {sorted(missing)}')
    if not isinstance(checkpoint['model_state_dict'], dict):
        raise ValueError('Checkpoint model_state_dict must be a mapping')
    if not isinstance(checkpoint['config'], dict):
        raise ValueError('Checkpoint config must be a mapping')
    return checkpoint


def _source_status_path(checkpoint_path, explicit=None):
    path = Path(explicit).resolve() if explicit else (
        checkpoint_path.parent.parent / 'status.json'
    )
    if not path.is_file():
        raise ValueError(f'Completed C0 status.json not found: {path}')
    return path


def _validate_source(checkpoint, checkpoint_path, status):
    config = checkpoint['config']
    failures = []
    for key in ('model', 'dataset', 'seed', 'protocol', 'implementation_version'):
        if checkpoint.get(key) != config.get(key):
            failures.append(f'checkpoint.{key} != checkpoint.config.{key}')
    if checkpoint['model'] != 'LGMRecOpt':
        failures.append('model must be LGMRecOpt')
    if checkpoint['protocol'] != 'train_only' or config.get('protocol') != 'train_only':
        failures.append('protocol must be train_only')
    if config.get('feat_mod_mode', 'off') != 'off':
        failures.append('feat_mod_mode must be off for C0')
    if config.get('ablation') not in (None, 'no_enhancement'):
        failures.append('checkpoint is not a C0 ablation')
    if config.get('report_test') is not False or checkpoint.get('test_result') is not None:
        failures.append('source checkpoint must be validation-only')
    if 'score_mode' in config:
        failures.append('source checkpoint already contains a scoring mode')

    expected_status = {
        'finished': True,
        'model': checkpoint['model'],
        'dataset': checkpoint['dataset'],
        'seed': checkpoint['seed'],
        'best_epoch': checkpoint['epoch'],
        'implementation_version': checkpoint['implementation_version'],
        'test_evaluated': False,
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            failures.append(f'status.{key} mismatch')
    if status.get('variant') != 'c0':
        failures.append('source status variant must be c0')
    if 'score_mode' in status.get('options', {}):
        failures.append('smoke/scoring checkpoint is not an allowed source')
    recorded_checkpoint = status.get('checkpoint')
    if not recorded_checkpoint or Path(recorded_checkpoint).name != checkpoint_path.name:
        failures.append('status checkpoint filename mismatch')
    if status.get('valid_result') != checkpoint.get('best_valid_result'):
        failures.append('status/checkpoint best_valid_result mismatch')
    if status.get('data_signature') != config.get('data_signature'):
        failures.append('status/checkpoint data_signature mismatch')
    if failures:
        raise ValueError('C0 source validation failed: ' + '; '.join(failures))


def _build_config(checkpoint, data_dir, gpu, cpu, run_dir):
    checkpoint_config = checkpoint['config']
    defaults = Config('LGMRecOpt', checkpoint['dataset'])
    known = set(defaults.final_config_dict)
    unknown = set(checkpoint_config) - known - CHECKPOINT_METADATA_KEYS
    if unknown:
        raise ValueError(f'Unknown checkpoint configuration keys: {sorted(unknown)}')
    overrides = {
        key: value for key, value in checkpoint_config.items()
        if key in known
        and key not in RUNTIME_CONFIG_KEYS
        and key not in COMPUTED_CONFIG_KEYS
    }
    overrides.update(
        data_path=str(data_dir.resolve()) + os.sep,
        gpu_id=gpu,
        use_gpu=not cpu,
        checkpoint_dir=str(run_dir / 'checkpoints'),
        log_dir=str(run_dir / 'log'),
        recommend_topk=str(run_dir / 'recommendations') + os.sep,
        save_recommended_topk=False,
        score_mode='cross',
        report_test=False,
        protocol='train_only',
    )
    config = Config('LGMRecOpt', checkpoint['dataset'], overrides)
    for key, value in checkpoint_config.items():
        if key in RUNTIME_CONFIG_KEYS or key in CHECKPOINT_METADATA_KEYS:
            continue
        if _json_value(config[key]) != _json_value(value):
            raise ValueError(f'Configuration mismatch for {key}')
    if 20 not in config['topk'] or 50 not in config['topk']:
        raise ValueError('Checkpoint topk must contain both 20 and 50')
    metrics = {value.lower() for value in config['metrics']}
    if not {'recall', 'ndcg'}.issubset(metrics):
        raise ValueError('Checkpoint metrics must contain Recall and NDCG')
    return config


def _validate_data_fingerprint(config, checkpoint):
    input_hashes, signature = fingerprint_inputs(config)
    recorded_hashes = checkpoint['config'].get('input_hashes')
    recorded_signature = checkpoint['config'].get('data_signature')
    if recorded_hashes != input_hashes or recorded_signature != signature:
        raise ValueError('Dataset fingerprint does not match the C0 checkpoint')
    return input_hashes, signature


def _load_c0_state(model, source_state):
    expected = model.state_dict()
    source_keys = set(source_state)
    expected_keys = set(expected)
    missing = expected_keys - source_keys
    unexpected = source_keys - expected_keys
    if missing != CROSS_STATE_KEYS:
        raise ValueError(
            'Checkpoint missing keys must be exactly the six cross logits; '
            f'got {sorted(missing)}'
        )
    if unexpected:
        raise ValueError(f'Unexpected checkpoint state keys: {sorted(unexpected)}')
    wrong_shapes = {
        key: (tuple(source_state[key].shape), tuple(expected[key].shape))
        for key in source_keys
        if tuple(source_state[key].shape) != tuple(expected[key].shape)
    }
    if wrong_shapes:
        raise ValueError(f'Checkpoint tensor shape mismatch: {wrong_shapes}')
    result = model.load_state_dict(source_state, strict=False)
    if set(result.missing_keys) != CROSS_STATE_KEYS or result.unexpected_keys:
        raise RuntimeError('Internal strict-load contract was not satisfied')


def _freeze_for_cross(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.cross_logits.values():
        parameter.requires_grad_(True)
    model.eval()
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    expected = sorted(CROSS_STATE_KEYS)
    if sorted(trainable) != expected:
        raise RuntimeError(f'Frozen-parameter check failed: {trainable}')
    return trainable


@torch.no_grad()
def _cache_representations(model):
    model.eval()
    users, items, _ = model.forward()
    return {
        'combined_users': users.detach(),
        'combined_items': items.detach(),
        'user_branches': tuple(
            value.detach() for value in model._score_user_branches
        ),
        'item_branches': tuple(
            value.detach() for value in model._score_item_branches
        ),
    }


def _cached_score(model, cache, users, items=None, full_sort=False):
    user_branches = tuple(value[users] for value in cache['user_branches'])
    combined_users = cache['combined_users'][users]
    if full_sort:
        return model.score(
            user_branches,
            cache['item_branches'],
            full_sort=True,
            combined=(combined_users, cache['combined_items']),
        )
    item_branches = tuple(value[items] for value in cache['item_branches'])
    return model.score(
        user_branches,
        item_branches,
        combined=(combined_users, cache['combined_items'][items]),
    )


@torch.no_grad()
def _evaluate_valid(model, cache, valid_data, evaluator, mode):
    model.eval()
    model.score_mode = mode
    topk_batches = []
    for batched_data in valid_data:
        users, masked_items = batched_data
        scores = _cached_score(model, cache, users, full_sort=True)
        scores[masked_items[0], masked_items[1]] = -1e10
        topk_batches.append(torch.topk(
            scores, max(evaluator.topk), dim=-1
        ).indices)
    return evaluator.evaluate(topk_batches, valid_data)


def _coefficient_values(model):
    return {
        name: float((2 * torch.sigmoid(parameter)).detach().cpu())
        for name, parameter in model.cross_logits.items()
    }


def _raw_logit_values(model):
    return {
        name: float(parameter.detach().cpu())
        for name, parameter in model.cross_logits.items()
    }


def _reproduction_check(actual, recorded, tolerance):
    missing = sorted(set(recorded) - set(actual))
    deltas = {
        key: abs(float(actual[key]) - float(recorded[key]))
        for key in recorded if key in actual
    }
    maximum = max(deltas.values(), default=math.inf)
    return {
        'passed': not missing and maximum <= tolerance,
        'tolerance': tolerance,
        'missing_metrics': missing,
        'max_absolute_delta': maximum,
        'deltas': deltas,
    }


def _make_loaders(config):
    dataset = RecDataset(config)
    train_dataset, valid_dataset, _test_dataset = dataset.split()
    train_data = TrainDataLoader(
        config, train_dataset,
        batch_size=config['train_batch_size'], shuffle=True,
    )
    _prepare_train_only_protocol(train_data, train_dataset, 1)
    valid_data = EvalDataLoader(
        config, valid_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'],
    )
    return train_data, valid_data


def _train_cross(model, cache, train_data, valid_data, evaluator, *,
                 epochs, learning_rate, patience, status, status_path):
    parameters = list(model.cross_logits.values())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    optimizer_ids = {id(value) for group in optimizer.param_groups
                     for value in group['params']}
    if optimizer_ids != {id(value) for value in parameters} or len(parameters) != 6:
        raise RuntimeError('Optimizer must contain exactly the six cross logits')

    model.score_mode = 'cross'
    initial = _evaluate_valid(model, cache, valid_data, evaluator, 'cross')
    status.setdefault('initial_results', {})['cross'] = initial
    _atomic_json(status_path, status)
    metric = model._probe_valid_metric
    best_result = initial
    best_epoch = -1
    best_score = float(initial[metric])
    best_logits = {
        name: value.detach().cpu().clone()
        for name, value in model.cross_logits.items()
    }
    stale = 0
    history = []
    train_data.pretrain_setup()
    for epoch in range(epochs):
        if model.training:
            raise RuntimeError('Model left eval mode during frozen scoring training')
        total_loss = 0.0
        batches = 0
        for interaction in train_data:
            if interaction.size(0) != 3:
                raise ValueError('Expected one TRAIN-only negative per positive')
            users, positive, negative = interaction.long()
            optimizer.zero_grad(set_to_none=True)
            positive_scores = _cached_score(model, cache, users, positive)
            negative_scores = _cached_score(model, cache, users, negative)
            loss = -torch.mean(F.logsigmoid(positive_scores - negative_scores))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite BPR loss at epoch {epoch}')
            loss.backward()
            encoder_grads = [
                name for name, parameter in model.named_parameters()
                if name not in CROSS_STATE_KEYS and parameter.grad is not None
            ]
            if encoder_grads:
                raise RuntimeError(f'Frozen encoder received gradients: {encoder_grads}')
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        valid_result = _evaluate_valid(
            model, cache, valid_data, evaluator, 'cross'
        )
        score = float(valid_result[metric])
        history.append({
            'epoch': epoch,
            'mean_train_bpr': total_loss / max(1, batches),
            'valid_result': valid_result,
            'coefficients': _coefficient_values(model),
        })
        if score > best_score:
            best_score = score
            best_result = valid_result
            best_epoch = epoch
            best_logits = {
                name: value.detach().cpu().clone()
                for name, value in model.cross_logits.items()
            }
            stale = 0
        else:
            stale += 1
        status['training_history'] = history
        status['last_completed_epoch'] = epoch
        _atomic_json(status_path, status)
        if stale >= patience:
            break

    with torch.no_grad():
        for name, value in model.cross_logits.items():
            value.copy_(best_logits[name].to(value.device))
    return {
        'epoch': best_epoch,
        'valid_metric': metric,
        'valid_score': best_score,
        'valid_result': best_result,
        'raw_logits': _raw_logit_values(model),
        'coefficients': _coefficient_values(model),
    }, initial


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Frozen C0 scoring probe: TRAIN BPR, VALID selection, no TEST'
    )
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-status', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--reproduction-tolerance', type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate must be finite and positive')
    if args.reproduction_tolerance < 0:
        parser.error('reproduction-tolerance must be non-negative')
    return args


def run(args):
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint_path}')
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise ValueError(f'Run directory already exists: {run_dir}')

    checkpoint = _load_checkpoint(checkpoint_path)
    status_path = _source_status_path(checkpoint_path, args.source_status)
    source_status = json.loads(status_path.read_text(encoding='utf-8'))
    _validate_source(checkpoint, checkpoint_path, source_status)
    config = _build_config(
        checkpoint, args.data_dir, args.gpu, args.cpu, run_dir
    )
    input_hashes, data_signature = _validate_data_fingerprint(
        config, checkpoint
    )

    init_seed(checkpoint['seed'])
    train_data, valid_data = _make_loaders(config)
    model = LGMRecOpt(config, train_data).to(config['device'])
    _load_c0_state(model, checkpoint['model_state_dict'])
    trainable = _freeze_for_cross(model)
    cache = _cache_representations(model)
    evaluator = TopKEvaluator(config)
    model._probe_valid_metric = config['valid_metric'].lower()

    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / 'status.json'
    status = {
        'schema_version': 1,
        'kind': 'frozen-scoring-probe',
        'finished': False,
        'protocol': 'train_only',
        'test_evaluated': False,
        'dataset': checkpoint['dataset'],
        'seed': checkpoint['seed'],
        'source_checkpoint': {
            'path': str(checkpoint_path),
            'sha256': _sha256(checkpoint_path),
            'size_bytes': checkpoint_path.stat().st_size,
            'epoch': checkpoint['epoch'],
            'status_path': str(status_path),
            'data_signature': data_signature,
            'input_hashes': input_hashes,
        },
        'probe_config': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'valid_metric': config['valid_metric'],
            'topk': config['topk'],
            'device': str(config['device']),
        },
        'freeze_check': {
            'model_training': model.training,
            'trainable_parameter_names': trainable,
            'trainable_scalar_count': sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            'representations_cached_and_detached': all(
                not tensor.requires_grad
                for key, value in cache.items()
                for tensor in (value if isinstance(value, tuple) else (value,))
            ),
        },
    }
    _atomic_json(output_path, status)

    try:
        original = _evaluate_valid(
            model, cache, valid_data, evaluator, 'original'
        )
        reproduction = _reproduction_check(
            original, checkpoint['best_valid_result'],
            args.reproduction_tolerance,
        )
        status['checkpoint_recorded_valid'] = checkpoint['best_valid_result']
        status['original_reproduction'] = reproduction
        status['initial_results'] = {'original': original}
        _atomic_json(output_path, status)
        if not reproduction['passed']:
            raise RuntimeError(
                'Original VALID result did not reproduce the checkpoint record: '
                f"max delta {reproduction['max_absolute_delta']}"
            )

        separate = _evaluate_valid(
            model, cache, valid_data, evaluator, 'separate'
        )
        status['initial_results']['separate'] = separate
        _atomic_json(output_path, status)
        best_cross, initial_cross = _train_cross(
            model, cache, train_data, valid_data, evaluator,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            status=status,
            status_path=output_path,
        )
        status['initial_results'] = {
            'original': original,
            'separate': separate,
            'cross': initial_cross,
        }
        status['best_cross'] = best_cross
        status['finished'] = True
        _atomic_json(output_path, status)
        torch.save({
            'cross_logits': {
                name: value.detach().cpu().clone()
                for name, value in model.cross_logits.items()
            },
            'coefficients': best_cross['coefficients'],
            'source_checkpoint_sha256': status['source_checkpoint']['sha256'],
            'data_signature': data_signature,
            'best_epoch': best_cross['epoch'],
            'best_valid_result': best_cross['valid_result'],
            'test_evaluated': False,
        }, run_dir / 'best_cross_logits.pth')
    except Exception as exc:
        status['error'] = f'{type(exc).__name__}: {exc}'
        _atomic_json(output_path, status)
        raise
    return status


def main(argv=None):
    args = parse_args(argv)
    status = run(args)
    print(json.dumps({
        'run_dir': str(args.run_dir.resolve()),
        'original': status['initial_results']['original'],
        'separate': status['initial_results']['separate'],
        'best_cross': status['best_cross'],
        'test_evaluated': status['test_evaluated'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
