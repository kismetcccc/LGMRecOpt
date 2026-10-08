import os
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from lgmrec.common.trainer import Trainer
from lgmrec.utils.quick_start import quick_start

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model,mode', [('LGMRec', 'off'), ('LGMRecOpt', 'off'), ('LGMRecOpt', 'id_cond')])
def test_train_only_end_to_end(tmp_path, monkeypatch, model, mode):
    directory = tmp_path / 'baby'
    directory.mkdir()
    rng = np.random.default_rng(8)
    for filename in ('image_feat.npy', 'text_feat.npy'):
        np.save(directory / filename, rng.normal(size=(6, 4)).astype('float32'))
    pd.DataFrame({'userID': [0,0,1,1,0,1,0,1], 'itemID': [0,1,1,2,3,4,5,5],
                  'timestamp': [1,2,1,2,3,3,4,4], 'x_label': [0,0,0,0,1,1,2,2]}).to_csv(
        directory / 'baby.inter', sep='\t', index=False)
    real_fit = Trainer.fit
    def fit(self, training, *, valid_data, test_data, saved):
        assert test_data is None
        assert training.history_items_per_u == {0:{0,1}, 1:{1,2}}
        assert set(valid_data.dataset.df.itemID) == {3,4}
        return real_fit(self, training, valid_data=valid_data, test_data=None, saved=saved, verbose=False)
    monkeypatch.setattr(Trainer, 'fit', fit)
    config = dict(data_path=str(tmp_path)+os.sep, use_gpu=False, seed=999,
                  epochs=2, train_batch_size=4, eval_batch_size=2,
                  topk=[1,2], valid_metric='Recall@2', embedding_size=8, feat_embed_dim=8,
                  n_ui_layers=1, n_mm_layers=1, n_hyper_layer=1, hyper_num=2,
                  keep_rate=1., alpha=.1, cl_weight=1e-4, reg_weight=1e-6,
                  log_dir=str(tmp_path/'log'), checkpoint_dir=str(tmp_path/'saved'),
                  save_recommended_topk=False)
    if model == 'LGMRecOpt':
        config['feat_mod_mode'] = mode
    result = quick_start(model, 'baby', config)
    assert result['test_evaluated'] is False
    assert Path(result['checkpoint']).is_file()
    assert result['best_epoch'] in (0,1)
    assert 'recall@2' in result['valid_result']


def test_cli_help_without_training():
    result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/train.py'), '--help'], capture_output=True)
    assert result.returncode == 0
    assert b'--variant' in result.stdout
    assert b'--no-checkpoint' in result.stdout


def test_cli_rejects_retired_and_test_overrides():
    for setting in ('hgnn_norm=true', 'report_test=true'):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/train.py'), '--set', setting], capture_output=True)
        assert result.returncode != 0
