"""Experimental TRAIN-behavior refinement of LGMRec hyperedge logits.

An independent candidate, not a reproduction of MMHCL. Reuses the existing
behavior graph, without new learnable parameters or auxiliary losses.
"""
import hashlib
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn


class BehaviorHypergraphRefiner(nn.Module):
    """Convex neighbor correction before the original Gumbel/softmax step.

    Remove self loops, row-normalize the weighted graph, and give isolated
    items an identity row. Random relabeling affects ONLY this module; it
    does not alter the graph used by the final embedding residual.
    """

    def __init__(self, adjacency, weight, *, graph_mode='cooccurrence', seed=0):
        super().__init__()
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('hyper_behavior_weight must be finite and in [0, 1]')
        if graph_mode not in ('cooccurrence', 'random_relabel'):
            raise ValueError('Unknown hyper_behavior_graph_mode')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('Hypergraph control seed must be a non-negative integer')
        if adjacency.layout != torch.sparse_coo or adjacency.ndim != 2:
            raise ValueError('Expected sparse COO item adjacency')
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('Item adjacency must be square')
        graph = adjacency.detach().cpu().coalesce()
        indices, values = graph.indices(), graph.values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Graph weights must be finite and non-negative')
        n_items = graph.shape[0]
        keep = (indices[0] != indices[1]) & (values > 0)
        rows, cols = indices[:, keep]
        values = values[keep]
        if graph_mode == 'random_relabel':
            permutation = torch.from_numpy(np.random.default_rng(seed).permutation(n_items))
            rows, cols = permutation[rows], permutation[cols]
        mass = torch.zeros(n_items, dtype=values.dtype)
        mass.index_add_(0, rows, values)
        values = values / mass[rows]
        isolated = torch.where(mass == 0)[0]
        rows = torch.cat((rows, isolated))
        cols = torch.cat((cols, isolated))
        values = torch.cat((values, torch.ones(isolated.numel(), dtype=values.dtype)))
        transition = torch.sparse_coo_tensor(
            torch.stack((rows, cols)), values, graph.shape
        ).coalesce()
        digest = hashlib.sha256()
        digest.update(np.asarray(graph.shape, dtype=np.int64).tobytes())
        digest.update(transition.indices().numpy().tobytes())
        digest.update(transition.values().numpy().tobytes())
        self.weight = float(weight)
        self.metadata = dict(
            version='behavior-hyper-logits-v1', weight=self.weight,
            graph_mode=graph_mode, seed=seed if graph_mode == 'random_relabel' else None,
            fingerprint=digest.hexdigest(), n_items=n_items,
            items_with_neighbors=int((mass > 0).sum()),
            isolated_items=int(isolated.numel()), nonzero_edges=transition._nnz(),
        )
        # Never let an old checkpoint overwrite the current TRAIN graph.
        self.register_buffer('transition', transition.to(adjacency.device), persistent=False)

    def forward(self, logits):
        if self.weight == 0:
            return logits
        neighbors = torch.sparse.mm(self.transition, logits)
        # Difference form preserves isolated rows exactly, even at nonzero weight.
        return logits + self.weight * (neighbors - logits)
