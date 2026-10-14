"""Selection precision and durable artifacts under interruption/failure."""
from pathlib import Path

import numpy as np
import pytest
import torch

from lgmrec.common.trainer import Trainer
from lgmrec.utils.configurator import Config
from lgmrec.utils.topk_evaluator import TopKEvaluator


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def calculate_loss(self, _):
        return (self.weight - 1).square()

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass


def make_trainer(tmp_path, **overrides):
    options = dict(use_gpu=False, seed=999, epochs=3, topk=[1],
                   metrics=['Recall'], valid_metric='Recall@1',
                   save_recommended_topk=False, checkpoint_dir=str(tmp_path / 'saved'))
    options.update(overrides)
    return Trainer(Config('LGMRec', 'baby', options), TinyModel())


def load_checkpoint(trainer):
    return torch.load(trainer.saved_model_file, map_location='cpu', weights_only=True)


def test_evaluator_retains_precision_for_selection(tmp_path):
    class EvaluationData:
        def get_eval_items(self):
            return [np.array([1, 2, 3])]

        def get_eval_len_list(self):
            return np.array([3])

    evaluator = TopKEvaluator(make_trainer(tmp_path).config)
    result = evaluator.evaluate([torch.tensor([[1]])], EvaluationData())
    assert result['recall@1'] == pytest.approx(1 / 3)
    assert result['recall@1'] != round(1 / 3, 4)


def test_baseline_and_extension_share_new_evaluation_version():
    versions = [Config(model, 'baby')['evaluation_version']
                for model in ('LGMRec', 'LGMRecOpt')]
    assert versions == ['full-ranking-train-only-full-precision-v2'] * 2


def test_close_validation_improvement_selects_new_state_and_preserves_other_runs(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    directory = Path(trainer.config['checkpoint_dir'])
    directory.mkdir()
    prior = directory / 'prior-run.pth'
    prior.write_bytes(b'keep previous experiment')
    scores = iter([0.10001, 0.10004, 0.10002])
    states = []

    def validate(_):
        states.append(trainer.model.weight.detach().clone())
        score = next(scores)
        return score, {'recall@1': score}

    monkeypatch.setattr(trainer, '_valid_epoch', validate)
    score, _, _ = trainer.fit([None], valid_data=object(), saved=True, verbose=False)
    assert score == 0.10004
    assert trainer.best_epoch_idx == 1
    checkpoint = load_checkpoint(trainer)
    assert checkpoint['epoch'] == 1
    assert checkpoint['best_valid_score'] == score
    assert checkpoint['test_result'] is None
    assert torch.equal(checkpoint['model_state_dict']['weight'], states[1])
    assert torch.equal(trainer.model.weight, states[1])
    assert prior.read_bytes() == b'keep previous experiment'
    assert len(list(directory.glob('*.pth'))) == 2
    assert not list(directory.glob('*.tmp'))


def test_interruption_keeps_last_valid_best_on_disk(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    evaluations = 0

    def validate(_):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise KeyboardInterrupt('simulated interruption')
        return 0.1, {'recall@1': 0.1}

    monkeypatch.setattr(trainer, '_valid_epoch', validate)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit([None], valid_data=object(), saved=True, verbose=False)
    checkpoint = load_checkpoint(trainer)
    assert checkpoint['epoch'] == 0
    assert checkpoint['best_valid_score'] == 0.1
    assert torch.equal(checkpoint['model_state_dict']['weight'], trainer.best_model_state['weight'])


def test_failed_atomic_save_does_not_corrupt_last_checkpoint(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    trainer._save_checkpoint(trainer.model.state_dict(), 0)
    path = Path(trainer.saved_model_file)
    previous_bytes = path.read_bytes()

    def fail_save(_, handle):
        handle.write(b'incomplete replacement')
        raise OSError('simulated disk error')

    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='disk error'):
        trainer._save_checkpoint(trainer.model.state_dict(), 1)
    assert path.read_bytes() == previous_bytes
    assert not list(path.parent.glob('*.tmp'))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_loss_fails_after_good_epoch_and_preserves_best(tmp_path, monkeypatch, value):
    trainer = make_trainer(tmp_path)
    real_loss = trainer.model.calculate_loss
    count = 0

    def loss(batch):
        nonlocal count
        count += 1
        if count == 1:
            return real_loss(batch)
        return trainer.model.weight * value

    monkeypatch.setattr(trainer.model, 'calculate_loss', loss)
    monkeypatch.setattr(trainer, '_valid_epoch', lambda _: (0.1, {'recall@1': 0.1}))
    with pytest.raises(FloatingPointError, match='Non-finite loss at epoch 1'):
        trainer.fit([None], valid_data=object(), saved=True, verbose=False)
    assert load_checkpoint(trainer)['epoch'] == 0


def test_no_checkpoint_mode_still_does_not_write(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, epochs=1)
    monkeypatch.setattr(trainer, '_valid_epoch', lambda _: (0.1, {'recall@1': 0.1}))
    trainer.fit([None], valid_data=object(), saved=False, verbose=False)
    assert trainer.best_model_state is not None
    assert trainer.saved_model_file is None
    assert not Path(trainer.config['checkpoint_dir']).exists()


def test_final_test_evaluation_runs_once_after_valid_selection(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, report_test=True)
    test_data = object()
    test_calls = []
    scores = iter([0.1, 0.2, 0.15])

    def validate(data):
        if data is test_data:
            test_calls.append(trainer.model.weight.detach().clone())
            assert trainer.best_epoch_idx == 1
            assert torch.equal(trainer.model.weight, trainer.best_model_state['weight'])
            return 0.3, {'recall@1': 0.3}
        score = next(scores)
        return score, {'recall@1': score}

    monkeypatch.setattr(trainer, '_valid_epoch', validate)
    trainer.fit([None], valid_data=object(), test_data=test_data, saved=True, verbose=False)
    assert len(test_calls) == 1
    assert load_checkpoint(trainer)['test_result'] == {'recall@1': 0.3}
