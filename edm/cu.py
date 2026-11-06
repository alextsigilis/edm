"""CU: one precision per time point, shared across clusters (a weighted Hamming distance).

    log f_g(s) = -Σ_t λ_t·1(s_t ≠ θ_{g,t}) - Σ_t log((v - 1)e^{-λ_t} + 1)

Closed-form CM-step (Table B2), used here only for initialisation:
    λ_t = max(0, log(v - 1) + log(n / Σ_i Σ_g z_ig w_i 1(s_it ≠ θ_gt) - 1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEDM
from .types import EDMType
from .utils import closed_form_precision, inv_softplus


class CUModel(BaseEDM):
    edm_type = EDMType.CU

    def _build_precision(self, init: float) -> None:
        self.raw_precision = nn.Parameter(inv_softplus(torch.full((self.T,), float(init))))

    def precision(self) -> torch.Tensor:
        return F.softplus(self.raw_precision).unsqueeze(0).expand(self.K, self.T)

    def n_precision_params(self) -> int:
        return self.T

    @torch.no_grad()
    def reset_precision(self, N: torch.Tensor, M: torch.Tensor) -> None:
        lam = closed_form_precision(N.sum(0), M.sum(0), self.v)
        self.raw_precision.copy_(inv_softplus(lam))