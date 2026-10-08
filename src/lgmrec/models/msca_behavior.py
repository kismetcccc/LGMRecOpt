"""TRAIN-only MSCA item behavior co-occurrence structural view.

This is an independent, minimal adaptation of ``get_struct_adj_mat`` and
``semantic_encode`` from the authors' MSCA implementation:
https://github.com/recomall/MSCA/blob/main/src/models/msca.py

It intentionally excludes MSCA's modality views, adaptive fusion, and
contrastive objectives. The sparse construction below avoids materializing
the authors' dense item-by-item temporary matrix while preserving the stated
candidate, binary-edge, self-loop, and degree-normalization rules.
"""
import hashlib
import json
from numbers import Integral

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


MSCA_STRUCT_TOPK = 10
MSCA_MIN_COOCCURRENCE = 2  # equivalent to the authors' data > 1
UPSTREAM_SOURCE = 'https://github.com/recomall/MSCA/blob/main/src/models/msca.py'


def _canonical_interactions(interaction_matrix):
    matrix = interaction_matrix.astype(np.float32).tocsr(copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('TRAIN interactions must be finite and non-negative')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data[:] = 1  # Count distinct users, not duplicate event frequency.
    matrix.sort_indices()
    return matrix


def behavior_graph_fingerprint(interaction_matrix, *, topk, minimum,
                               graph_mode='cooccurrence', graph_seed=0,
                               self_loop=True, edge_weight='binary'):
    """Identify the exact TRAIN matrix and structural graph parameters."""
    matrix = _canonical_interactions(interaction_matrix)
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.data.astype(np.float32, copy=False).tobytes())
    digest.update(np.asarray([topk, minimum], dtype=np.int64).tobytes())
    digest.update(json.dumps(dict(version=2, graph_mode=graph_mode,
        graph_seed=graph_seed if graph_mode == 'random_relabel' else None,
        self_loop=self_loop, edge_weight=edge_weight), sort_keys=True).encode())
    return digest.hexdigest()


def _scipy_to_torch(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, device=device
    ).coalesce()


def build_normalized_interactions(interaction_matrix, device):
    """Build MSCA's D_u^-1/2 R D_i^-1/2 from TRAIN interactions."""
    matrix = _canonical_interactions(interaction_matrix)
    user_degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    item_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    user_scale = np.zeros_like(user_degree, dtype=np.float32)
    item_scale = np.zeros_like(item_degree, dtype=np.float32)
    np.power(user_degree, -0.5, out=user_scale, where=user_degree > 0)
    np.power(item_degree, -0.5, out=item_scale, where=item_degree > 0)
    normalized = sp.diags(user_scale).dot(matrix).dot(
        sp.diags(item_scale)
    )
    return _scipy_to_torch(normalized, device)


def build_structural_adjacency(interaction_matrix, *, topk, minimum,
                               device, graph_mode='cooccurrence', graph_seed=0,
                               self_loop=True, edge_weight='binary', block_size=1024):
    """Directed row-TopK graph; blockwise product bounds temporary row count.

    random_relabel permutes item identities, preserving topology and the
    degree distribution, NOT each item's degree. It is a correspondence control.
    """
    for name, value in [('topk', topk), ('minimum', minimum), ('block_size', block_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if graph_mode not in ('cooccurrence', 'random_relabel'):
        raise ValueError('Unknown behavior graph mode')
    if edge_weight not in ('binary', 'count') or not isinstance(self_loop, bool):
        raise ValueError('Invalid edge_weight or self_loop')
    if isinstance(graph_seed, bool) or not isinstance(graph_seed, Integral) or graph_seed < 0:
        raise ValueError('graph_seed must be a non-negative integer')
    interactions = _canonical_interactions(interaction_matrix)
    n_items = interactions.shape[1]
    rows, columns, weights = [], [], []
    for item in range(n_items):
        if item % block_size == 0:
            cooccurrence = (interactions[:, item:item + block_size].T @ interactions).tocsr()
        local = item % block_size
        start, end = cooccurrence.indptr[local:local + 2]
        candidates = cooccurrence.indices[start:end]
        counts = cooccurrence.data[start:end]
        eligible = (counts >= minimum) & (candidates != item)
        candidates, counts = candidates[eligible], counts[eligible]
        if candidates.size:
            # Descending count with item id as a deterministic tie-breaker.
            order = np.lexsort((candidates, -counts))[:topk]
            selected = candidates[order]
            rows.extend([item] * len(selected))
            columns.extend(selected.tolist())
            weights.extend(counts[order].tolist() if edge_weight == 'count' else [1.] * len(selected))

    if self_loop:
        rows.extend(range(n_items))
        columns.extend(range(n_items))
        weights.extend([1.] * n_items)

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if graph_mode == 'random_relabel':
        permutation = np.random.default_rng(graph_seed).permutation(n_items)
        rows, columns = permutation[rows], permutation[columns]
    weights = np.asarray(weights, dtype=np.float32)
    degree = np.bincount(rows, weights=weights, minlength=n_items).astype(np.float32)
    scale = np.zeros_like(degree)
    np.power(degree, -.5, out=scale, where=degree > 0)
    values = weights * scale[rows] * scale[columns]
    adjacency = sp.coo_matrix(
        (values, (rows, columns)), shape=(n_items, n_items)
    )
    adjacency.eliminate_zeros()
    return _scipy_to_torch(adjacency, device)


class MSCABehaviorView(nn.Module):
    """Parameter-free MSCA structural encoder cached from one TRAIN graph."""

    def __init__(self, interaction_matrix, device, *,
                 topk=MSCA_STRUCT_TOPK,
                 minimum=MSCA_MIN_COOCCURRENCE, graph_mode='cooccurrence',
                 graph_seed=0, self_loop=True, edge_weight='binary', block_size=1024):
        super().__init__()
        self.topk = topk
        self.minimum = minimum
        options = dict(graph_mode=graph_mode, graph_seed=graph_seed,
                       self_loop=self_loop, edge_weight=edge_weight)
        self.cache_fingerprint = behavior_graph_fingerprint(
            interaction_matrix, topk=self.topk, minimum=self.minimum, **options
        )
        # Rebuild from the current TRAIN matrix on every model construction.
        # Non-persistent buffers prevent a checkpoint from replacing this
        # graph with one produced by a different dataset or split.
        self.register_buffer(
            'structural_adjacency',
            build_structural_adjacency(
                interaction_matrix, topk=self.topk,
                minimum=self.minimum, device=device, block_size=block_size, **options
            ),
            persistent=False,
        )
        self.metadata = dict(fingerprint=self.cache_fingerprint, topk=topk,
                             minimum=minimum, **options,
                             normalized_nonzero_edges=self.structural_adjacency._nnz())
        self.register_buffer(
            'normalized_interactions',
            build_normalized_interactions(interaction_matrix, device),
            persistent=False,
        )

    def forward(self, item_id_embeddings):
        item_structural = torch.sparse.mm(
            self.structural_adjacency, item_id_embeddings
        )
        user_structural = torch.sparse.mm(
            self.normalized_interactions, item_structural
        )
        return torch.cat((user_structural, item_structural), dim=0)
