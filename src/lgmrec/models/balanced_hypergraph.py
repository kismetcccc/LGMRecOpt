"""Opt-in hyperedge-mass correction, not an empirically proven upgrade."""
import math

import torch

from .lgmrec import HGNNLayer


class BalancedHGNNLayer(HGNNLayer):
    """Replace hyperedge sums by degree-corrected sums using dropped incidence.

    power=1 is hyperedge mean pooling, power=.5 a partial correction.
    power=0 executes the original implementation exactly. No new parameters.
    This is only hyperedge normalization, not full symmetric HGNN normalization.
    """

    def __init__(self, n_hyper_layer, power):
        super().__init__(n_hyper_layer)
        if not math.isfinite(power) or not 0 <= power <= 1:
            raise ValueError('hyper_degree_power must be finite and in [0, 1]')
        if n_hyper_layer < 1:
            raise ValueError('Balanced hypergraph requires at least one layer')
        self.power = float(power)

    def forward(self, i_hyper, u_hyper, embeds):
        if self.power == 0:
            return super().forward(i_hyper, u_hyper, embeds)
        # Same TRAIN-derived (and training-dropout) incidences as the numerator.
        # Empty hyperedges have zero sums; clamp avoids division by zero.
        mass = i_hyper.sum(dim=0).clamp_min(torch.finfo(i_hyper.dtype).eps)
        divisor = mass.pow(self.power).unsqueeze(1)
        items = embeds
        for _ in range(self.h_layer):
            latent = (i_hyper.T @ items) / divisor
            items = i_hyper @ latent
            users = u_hyper @ latent
        return users, items
