"""CC: one precision shared by every cluster and every time point.

    log f_g(s) = -λ·d_H(s, θ_g) - T·log((v - 1)e^{-λ} + 1)

Closed-form CM-step (Table B2), used here only for initialisation:
    λ = max(0, log(v - 1) + log(T·n / Σ_i Σ_g z_ig w_i d_H(s_i, θ_g) - 1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEDM
from .types import EDMType
from .utils import closed_form_precision, inv_softplus


class CCModel(BaseEDM):
    edm_type = EDMType.CC

    def _build_precision(self, init: float) -> None:
        self.raw_precision = nn.Parameter(inv_softplus(torch.tensor(float(init))))

    def precision(self) -> torch.Tensor:
        return F.softplus(self.raw_precision).expand(self.K, self.T)

    def n_precision_params(self) -> int:
        return 1

    @torch.no_grad()
    def reset_precision(self, N: torch.Tensor, M: torch.Tensor) -> None:
        lam = closed_form_precision(N.sum(), M.sum(), self.v)
        self.raw_precision.copy_(inv_softplus(lam))