"""Gating network τ_g(x): mixing proportions, optionally driven by covariates (Section 3.4)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Returns log mixing proportions of shape (n, G); a noise component is the last column.

    * no covariates : constant τ, G - 1 free parameters
    * GN            : multinomial logit over all G components (reference = component 0)
    * NGN           : constant τ_0 for the noise component; multinomial logit over the
                      G - 1 non-noise components, scaled by (1 - τ_0)
    """

    def __init__(self, n_components: int, n_covariates: int = 0, *,
                 noise: bool = False, gated_noise: bool = False):
        super().__init__()
        self.G, self.r = n_components, n_covariates
        self.ngn = noise and n_covariates > 0 and not gated_noise
        if n_covariates == 0:
            self.logits = nn.Parameter(torch.zeros(n_components))
        elif self.ngn:
            if n_components - 1 < 2:
                raise ValueError("NGN gating covariates need >= 2 non-noise components (G >= 3)")
            self.linear = nn.Linear(n_covariates, n_components - 2)
            self.noise_logit = nn.Parameter(torch.tensor(-3.0))
        else:
            if n_components < 2:
                raise ValueError("gating covariates need G >= 2")
            self.linear = nn.Linear(n_covariates, n_components - 1)

    def _with_reference(self, x: torch.Tensor) -> torch.Tensor:
        eta = self.linear(x)
        return torch.cat([eta.new_zeros(eta.shape[0], 1), eta], dim=1).log_softmax(dim=1)

    def forward(self, covariates: torch.Tensor | None = None, n: int | None = None) -> torch.Tensor:
        if self.r == 0:
            return self.logits.log_softmax(0).expand(n, -1)
        if covariates is None:
            raise ValueError("this model was built with gating covariates; pass them")
        if not self.ngn:
            return self._with_reference(covariates)
        log_rest = F.logsigmoid(-self.noise_logit) + self._with_reference(covariates)
        log_noise = F.logsigmoid(self.noise_logit).expand(covariates.shape[0], 1)
        return torch.cat([log_rest, log_noise], dim=1)

    def n_parameters(self) -> int:
        if self.r == 0:
            return self.G - 1
        if self.ngn:
            return (self.r + 1) * (self.G - 2) + 1
        return (self.r + 1) * (self.G - 1)

    @torch.no_grad()
    def initialize(self, log_props: torch.Tensor) -> None:
        """Start at the given log mixing proportions (noise last), with zero covariate effects."""
        if self.r == 0:
            self.logits.copy_(log_props)
            return
        self.linear.weight.zero_()
        if self.ngn:
            log_tau0 = log_props[-1]
            self.noise_logit.copy_(log_tau0 - torch.log1p(-log_tau0.exp()))
            lp = log_props[:-1]
            self.linear.bias.copy_(lp[1:] - lp[0])
        else:
            self.linear.bias.copy_(log_props[1:] - log_props[0])

    def coefficients(self) -> torch.Tensor | None:
        """Rows = non-reference components; columns = (intercept, covariates...)."""
        if self.r == 0:
            return None
        return torch.cat([self.linear.bias[:, None], self.linear.weight], dim=1).detach().clone()