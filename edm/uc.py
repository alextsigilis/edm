"""UC: one precision per cluster, shared across time points.

    log f_g(s) = -λ_g·d_H(s, θ_g) - T·log((v - 1)e^{-λ_g} + 1)

Closed-form CM-step (Table B2), used here only for initialisation:
    λ_g = max(0, log(v - 1) + log(T·Σ_i z_ig w_i / Σ_i z_ig w_i d_H(s_i, θ_g) - 1))
With G = 1 this is identical to CC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEDM
from .types import EDMType
from .utils import closed_form_precision, inv_softplus


class UCModel(BaseEDM):
    edm_type = EDMType.UC

    def _build_precision(self, init: float) -> None:
        self.raw_precision = nn.Parameter(inv_softplus(torch.full((self.K,), float(init))))

    def precision(self) -> torch.Tensor:
        return F.softplus(self.raw_precision).unsqueeze(1).expand(self.K, self.T)

    def n_precision_params(self) -> int:
        return self.K

    @torch.no_grad()
    def reset_precision(self, N: torch.Tensor, M: torch.Tensor) -> None:
        lam = closed_form_precision(N.sum(1), M.sum(1), self.v)
        self.raw_precision.copy_(inv_softplus(lam))