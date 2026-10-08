"""Isolated SMORE cross-modality spectral fusion branch.

Algorithm adapted from ``spectrum_convolution`` in kennethorq/SMORE at commit
c745bb92932981b371e8480ae9376ec3e71aaf24:
https://github.com/kennethorq/SMORE/blob/c745bb92932981b371e8480ae9376ec3e71aaf24/src/models/smore.py

The referenced repository had no root LICENSE at that commit, so this is an
attributed independent reimplementation rather than a verbatim copy. Only the
cross-modality fusion calculation is implemented here; SMORE's unimodal
denoising, KNN graphs, preference gates, and contrastive loss are not included.
"""
import torch
from torch import nn


UPSTREAM_COMMIT = 'c745bb92932981b371e8480ae9376ec3e71aaf24'


class CrossModalSpectrumFusion(nn.Module):
    """Fuse aligned visual/text item features in the frequency domain."""

    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        self.dim = dim
        self.fusion_complex_weight = nn.Parameter(
            torch.randn(1, dim // 2 + 1, 2, dtype=torch.float32)
        )

    def forward(self, visual_items, text_items):
        if visual_items.ndim != 2 or text_items.ndim != 2:
            raise ValueError('visual_items and text_items must be rank-2')
        if visual_items.shape != text_items.shape:
            raise ValueError('visual_items and text_items must have equal shapes')
        if visual_items.shape[1] != self.dim:
            raise ValueError(f'feature width must equal {self.dim}')
        visual_spectrum = torch.fft.rfft(
            visual_items, dim=1, norm='ortho'
        )
        text_spectrum = torch.fft.rfft(
            text_items, dim=1, norm='ortho'
        )
        complex_weight = torch.view_as_complex(
            self.fusion_complex_weight
        )
        return torch.fft.irfft(
            visual_spectrum * text_spectrum * complex_weight,
            n=self.dim,
            dim=1,
            norm='ortho',
        )
