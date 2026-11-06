"""UU: a separate precision for every (cluster, time point) pair.

    log f_g(s) = -Σ_t λ_{g,t}·1(s_t ≠ θ_{g,t}) - Σ_t log((v - 1)e^{-λ_{g,t}} + 1)

Closed-form CM-step (Table B2), used here only for initialisation:
    λ_gt = max(0, log(v - 1) + log(Σ_i z_ig w_i / Σ_i z_ig w_i 1(s_it ≠ θ_gt) - 1))
If every member of g agrees at t, the MLE is λ_gt = ∞; under softplus the parameter just
keeps growing slowly, which is harmless. With G = 1 this is identical to CU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEDM
from .types import EDMType
from .utils import closed_form_precision, inv_softplus


class UUModel(BaseEDM):
    edm_type = EDMType.UU

    def _build_precision(self, init: float) -> None:
        self.raw_precision = nn.Parameter(inv_softplus(torch.full((self.K, self.T), float(init))))

    def precision(self) -> torch.Tensor:
        return F.softplus(self.raw_precision)

    def n_precision_params(self) -> int:
        return self.K * self.T

    @torch.no_grad()
    def reset_precision(self, N: torch.Tensor, M: torch.Tensor) -> None:
        self.raw_precision.copy_(inv_softplus(closed_form_precision(N, M, self.v)))