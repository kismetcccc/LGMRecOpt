"""Parameter-free, TRAIN-supported item -> group -> item propagation.

Each directed structural-graph row defines one anchored hyperedge. Graph
values are ignored: membership comes from the positive, unique support.
Attention is deterministic and reused across propagation layers. No graph
or membership tensor is persisted in a checkpoint.
"""
import hashlib
import json
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class SparseGroupHypergraph(nn.Module):
    def __init__(self, structural_adjacency, *, weighting='attention', layers=1):
        super().__init__()
        if weighting not in ('attention', 'uniform'):
            raise ValueError('weighting must be attention or uniform')
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError('layers must be a positive integer')
        if structural_adjacency.layout != torch.sparse_coo or structural_adjacency.ndim != 2:
            raise ValueError('Expected sparse COO item adjacency')
        n_items = structural_adjacency.shape[0]
        if n_items < 1 or structural_adjacency.shape[1] != n_items:
            raise ValueError('Item adjacency must be nonempty and square')
        graph = structural_adjacency.detach().cpu().coalesce()
        values = graph.values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Graph weights must be finite and non-negative')
        rows, columns = graph.indices()[:, values > 0]
        if int((rows == columns).sum()) != n_items:
            raise ValueError('Each hyperedge must include its anchor self membership')
        sizes = torch.bincount(rows, minlength=n_items)
        width = int(sizes.max())
        # Padding is the row's anchor, never an unrelated item such as item 0.
        members = torch.arange(n_items)[:, None].expand(n_items, width).clone()
        mask = torch.zeros((n_items, width), dtype=torch.bool)
        offsets = torch.cumsum(sizes, dim=0) - sizes
        slots = torch.arange(rows.numel()) - offsets[rows]
        members[rows, slots] = columns
        mask[rows, slots] = True
        device = structural_adjacency.device
        self.register_buffer('members', members.to(device), persistent=False)
        self.register_buffer('mask', mask.to(device), persistent=False)
        self.register_buffer('_edge_ids', rows.to(device), persistent=False)
        self.register_buffer('_item_ids', columns.to(device), persistent=False)
        self.weighting = weighting
        self.layers = int(layers)
        self.n_items = n_items
        support = {'shape': [n_items, n_items], 'rows': rows.tolist(),
                   'columns': columns.tolist()}
        fingerprint = hashlib.sha256(json.dumps(
            support, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.metadata = dict(
            version='sparse-group-hypergraph-v1', weighting=weighting, layers=self.layers,
            n_hyperedges=n_items, nnz=int(rows.numel()), min_size=int(sizes.min()),
            max_size=width, singleton_count=int((sizes == 1).sum()),
            fingerprint=fingerprint,
        )

    def weights(self, item_embeddings, projected_features):
        """Return [anchor, padded member] coefficients; padded entries are zero."""
        for name, value in (('item_embeddings', item_embeddings),
                            ('projected_features', projected_features)):
            if value.ndim != 2 or value.shape[0] != self.n_items or value.shape[1] < 1:
                raise ValueError(f'{name} must be a nonempty item feature matrix')
            if value.device != self.members.device or not value.is_floating_point():
                raise ValueError(f'{name} must be floating point on the graph device')
        if item_embeddings.dtype != projected_features.dtype:
            raise ValueError('Item embeddings and projected features must share dtype')
        mask = self.mask.to(dtype=item_embeddings.dtype)
        counts = mask.sum(dim=1, keepdim=True)
        if self.weighting == 'uniform':
            return mask / counts
        features = F.normalize(projected_features, dim=1)
        gathered = features[self.members]
        query = F.normalize((gathered * mask[..., None]).sum(dim=1) / counts, dim=1)
        collaborative = F.normalize(item_embeddings, dim=1)
        scores = (gathered * query[:, None, :]).sum(dim=-1)
        scores = scores + (collaborative[self.members] * collaborative[:, None, :]).sum(dim=-1)
        return scores.masked_fill(~self.mask, -torch.inf).softmax(dim=1)

    def forward(self, item_embeddings, projected_features):
        # The memberships depend on initial representations. Repeated layers
        # propagate with these same weights rather than rebuilding a new graph.
        weights = self.weights(item_embeddings, projected_features)
        real_weights = weights[self.mask]
        mass = item_embeddings.new_zeros(self.n_items)
        mass.index_add_(0, self._item_ids, real_weights)
        current = item_embeddings
        for _ in range(self.layers):
            edges = (current[self.members] * weights[..., None]).sum(dim=1)
            messages = edges[self._edge_ids] * real_weights[:, None]
            current = torch.zeros_like(current).index_add_(0, self._item_ids, messages)
            current = current / mass.clamp_min(torch.finfo(current.dtype).tiny)[:, None]
        return current
