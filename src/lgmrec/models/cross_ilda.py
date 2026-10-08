"""CROSS ILDA+DT directed-alignment loss, independently reimplemented.

Algorithm and interface adapted from XMUDM/FETTLE ``CROSS/loss_cross.py`` at
commit 125928a921bcb7cc395086ea9c87543914d008b7:
https://github.com/XMUDM/FETTLE/blob/125928a921bcb7cc395086ea9c87543914d008b7/CROSS/loss_cross.py

The referenced repository had no root LICENSE at that commit, so this file is
an attributed reimplementation rather than a verbatim copy. It preserves the
six residual directions, leaky bidirectional feedback masks, target-side stop
gradient, direction-tuning term, and learnable clamped temperature. Native
``index_add_`` replaces ``torch_scatter.scatter_add``. ``F.normalize`` and
population variance keep singleton batches finite without changing ordinary
multi-item semantics.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_COMMIT = '125928a921bcb7cc395086ea9c87543914d008b7'


class ILALoss(nn.Module):
    """Item-level dynamic alignment plus direction tuning from CROSS."""

    def __init__(self, dim=64, gamma=0.007, leaky_bi=False):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('gamma must be finite and positive')
        self.temp = nn.Parameter(torch.tensor(float(gamma)))
        self.i2t_map = nn.Linear(dim, dim, bias=False)
        self.t2i_map = nn.Linear(dim, dim, bias=False)
        self.i2d_map = nn.Linear(dim, dim, bias=False)
        self.d2i_map = nn.Linear(dim, dim, bias=False)
        self.t2d_map = nn.Linear(dim, dim, bias=False)
        self.d2t_map = nn.Linear(dim, dim, bias=False)
        self.leaky_bi = bool(leaky_bi)

    @staticmethod
    def _group_mean(values, inverse, counts):
        grouped = values.new_zeros(counts.numel())
        grouped.index_add_(0, inverse, values)
        return grouped / counts.to(values.dtype)

    @staticmethod
    def _direction_ce(source, target, mask, labels, temperature):
        if not mask.any():
            return temperature * 0.0
        logits = (source @ target.detach().T)[mask] / temperature
        return F.cross_entropy(logits, labels[mask], reduction='sum')

    def _feedback_masks(self, id_scores, image_scores, text_scores):
        if self.leaky_bi:
            def adjusted(scores):
                variance = torch.var(scores, unbiased=False)
                return scores + variance * (torch.exp(1 - scores) - 1)

            adjusted_id = adjusted(id_scores)
            adjusted_image = adjusted(image_scores)
            adjusted_text = adjusted(text_scores)
            return {
                't2i': adjusted_image > text_scores,
                'i2t': adjusted_text > image_scores,
                'i2d': adjusted_id > image_scores,
                'd2i': adjusted_image > id_scores,
                't2d': adjusted_id > text_scores,
                'd2t': adjusted_text > id_scores,
            }
        return {
            't2i': image_scores > text_scores,
            'i2t': text_scores > image_scores,
            'i2d': id_scores > image_scores,
            'd2i': image_scores > id_scores,
            't2d': id_scores > text_scores,
            'd2t': text_scores > id_scores,
        }

    def forward(self, user_embeddings, item_embeddings, image_embeddings,
                text_embeddings, user_id, item_id, epoch_idx=None):
        del epoch_idx
        embeddings = (
            user_embeddings, item_embeddings, image_embeddings, text_embeddings
        )
        if any(value.ndim != 2 for value in embeddings):
            raise ValueError('All embedding inputs must be rank-2 tensors')
        dim = self.i2t_map.in_features
        if any(value.shape[1] != dim for value in embeddings):
            raise ValueError(f'All embedding widths must equal {dim}')
        if user_id.ndim != 1 or item_id.ndim != 1 or user_id.shape != item_id.shape:
            raise ValueError('user_id and item_id must be aligned rank-1 tensors')
        if user_id.numel() == 0:
            raise ValueError('ILDA+DT requires a non-empty positive batch')

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # Use a per-forward value so a later forward's safety clamp cannot
        # invalidate an earlier graph before its backward pass.
        temperature = self.temp.clone()

        sorted_items, order = torch.sort(item_id.long())
        sorted_users = user_id.long()[order]
        unique_items, inverse, counts = torch.unique(
            sorted_items, sorted=True, return_inverse=True, return_counts=True
        )

        users = F.normalize(user_embeddings, dim=1)
        ids = F.normalize(item_embeddings, dim=1)
        images = F.normalize(image_embeddings, dim=1)
        texts = F.normalize(text_embeddings, dim=1)

        with torch.no_grad():
            id_scores = self._group_mean(
                (users[sorted_users] * ids[sorted_items]).sum(dim=1),
                inverse, counts,
            )
            image_scores = self._group_mean(
                (users[sorted_users] * images[sorted_items]).sum(dim=1),
                inverse, counts,
            )
            text_scores = self._group_mean(
                (users[sorted_users] * texts[sorted_items]).sum(dim=1),
                inverse, counts,
            )
            masks = self._feedback_masks(
                id_scores, image_scores, text_scores
            )

        ids = ids[unique_items]
        images = images[unique_items]
        texts = texts[unique_items]
        transformed = {
            'i2t': F.normalize(self.i2t_map(images) + images, dim=1),
            'i2d': F.normalize(self.i2d_map(images) + images, dim=1),
            't2i': F.normalize(self.t2i_map(texts) + texts, dim=1),
            't2d': F.normalize(self.t2d_map(texts) + texts, dim=1),
            'd2i': F.normalize(self.d2i_map(ids) + ids, dim=1),
            'd2t': F.normalize(self.d2t_map(ids) + ids, dim=1),
        }
        targets = {
            'i2t': texts,
            'i2d': ids,
            't2i': images,
            't2d': ids,
            'd2i': images,
            'd2t': texts,
        }
        baselines = {
            'i2t': image_scores,
            'i2d': image_scores,
            't2i': text_scores,
            't2d': text_scores,
            'd2i': id_scores,
            'd2t': id_scores,
        }
        labels = torch.arange(unique_items.numel(), device=ids.device)
        align_loss = temperature * 0.0
        for direction in ('t2i', 'i2t', 'i2d', 'd2i', 't2d', 'd2t'):
            align_loss = align_loss + self._direction_ce(
                transformed[direction], targets[direction],
                masks[direction], labels, temperature,
            )
        align_loss = align_loss / unique_items.numel()

        detached_users = users.detach()[sorted_users]
        direction_loss = temperature * 0.0
        for direction in ('i2t', 't2i', 'i2d', 'd2i', 't2d', 'd2t'):
            mask = masks[direction]
            if not mask.any():
                continue
            positive = self._group_mean(
                (detached_users * transformed[direction][inverse]).sum(dim=1),
                inverse, counts,
            )
            direction_loss = direction_loss - torch.mean(
                positive[mask] - baselines[direction][mask]
            )
        return align_loss + direction_loss / 6
