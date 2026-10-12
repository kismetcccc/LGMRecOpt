"""Loss replacement preserves inference and uses only existing TRAIN views."""
import pytest
import torch
from torch.nn import functional as F

from lgmrec.models.lgmrecopt import LGMRecOpt
from test_msca_behavior import Dataset, tiny_config


def test_collaborative_hcl_inference_rng_and_training(tmp_path):
    models = []
    for mode in ('cross_modal', 'collaborative'):
        torch.manual_seed(39)
        model = LGMRecOpt(tiny_config(tmp_path, hcl_view=mode,
                         behavior_view_mode='msca_struct'), Dataset())
        models.append(model)
    baseline, candidate = models
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    for model in models:
        model.eval()
    for left, right in zip(baseline.forward()[:2], candidate.forward()[:2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    candidate.train()
    loss = candidate.calculate_loss(torch.tensor([[0, 1, 2], [0, 1, 3], [4, 5, 4]]))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (candidate.v_hyper, candidate.t_hyper,
                      candidate.user_embedding.weight, candidate.item_id_embedding.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_loss_reference_and_stop_gradient(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path, hcl_view='collaborative'), Dataset())
    torch.manual_seed(4)
    target = torch.randn(10, 8, requires_grad=True)
    model._score_collaborative = target
    hyper = tuple(torch.randn(n, 8, requires_grad=True) for n in (4, 6, 4, 6))
    uid, positive = torch.tensor([0, 0, 2]), torch.tensor([1, 3, 3])
    actual = model.hypergraph_contrastive_loss(hyper, uid, positive)
    expected = 0.
    for view, labels, bank in zip(hyper, (uid, positive, uid, positive),
                                 (target[:4], target[4:], target[:4], target[4:])):
        logits = F.normalize(view[labels]) @ F.normalize(bank.detach()).T / model.tau
        expected = expected + 0.5 * F.cross_entropy(logits, labels, reduction='sum')
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert target.grad is None
    assert all(view.grad is not None and view.grad.norm() > 0 for view in hyper)


def test_original_hcl_unchanged_and_invalid_mode(tmp_path):
    model = LGMRecOpt(tiny_config(tmp_path), Dataset())
    _, _, hyper = model.forward()
    uv, iv, ut, it = hyper
    uid, positive = torch.tensor([0, 2]), torch.tensor([1, 3])
    expected = model.ssl_triple_loss(uv[uid], ut[uid], ut) + model.ssl_triple_loss(iv[positive], it[positive], it)
    torch.testing.assert_close(model.hypergraph_contrastive_loss(hyper, uid, positive), expected)
    with pytest.raises(ValueError, match='hcl_view'):
        LGMRecOpt(tiny_config(tmp_path, hcl_view='unknown'), Dataset())
