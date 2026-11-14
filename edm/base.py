"""Shared machinery for the MEDseq family of exponential-distance mixtures, trained by autodiff.

Every non-noise component g has a central sequence θ_g and precisions λ_{g,t} >= 0:

    log f_g(s) = -Σ_t λ_{g,t}·1(s_t ≠ θ_{g,t}) - Σ_t log((v - 1)·exp(-λ_{g,t}) + 1)

θ_g is relaxed to soft one-hot rows θ̃_{g,t} = softmax(φ_{g,t} / τ), and the mismatch
indicator to 1 - <x_t, θ̃_{g,t}>, where x_t is the one-hot state.

normalizer="hamming": the paper's closed-form constant. For soft θ̃ the objective lower-bounds
    the log-likelihood of a proper distribution, is exact for one-hot θ̃, and (for fixed
    responsibilities and λ) is linear in θ̃, so its maximiser is the weighted modal sequence.
normalizer="exact": Σ_t log Σ_k exp(-λ_{g,t}(1 - θ̃_{g,t,k})), the exact constant for soft θ̃.
    Components become a richer product-of-categoricals family; annealing τ makes θ̃ one-hot.
normalizer="sampled": the same value as "exact", but the gradient of log Z_g is not taken
    through the closed form. It is estimated by Monte Carlo with the identity of
    Goodfellow, Bengio & Courville (2016), Deep Learning, eq. 18.13:

        ∇ log Z_g = E_{s ~ f_g}[∇ log f̃_g(s)],   log f̃_g(s) = -Σ_t λ_{g,t}(1 - <x_t, θ̃_{g,t}>)

    with mc_samples sequences drawn from each component on every forward pass that needs a
    gradient. The estimate is unbiased for the gradient of the "exact" normaliser.

All three coincide for one-hot θ̃, which is how log-likelihoods and BIC are reported.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import GatingNetwork
from .types import EDMType
from .utils import weighted_kmodes

NORMALIZERS = ("hamming", "exact", "sampled")


class BaseEDM(nn.Module):
    """Subclasses decide how λ is shared across clusters and time points by implementing
    ``_build_precision``, ``precision``, ``n_precision_params`` and ``reset_precision``.
    ``has_noise = True`` appends a uniform component (λ = 0) as the last component.
    """

    edm_type: EDMType
    has_noise: bool = False

    def __init__(
        self,
        n_components: int,
        seq_len: int,
        n_states: int,
        n_covariates: int = 0,
        *,
        gated_noise: bool = False,
        normalizer: str = "hamming",
        straight_through: bool = False,
        init_precision: float = 1.0,
        mc_samples: int = 32,
    ):
        super().__init__()
        if n_states < 2:
            raise ValueError("need at least 2 states")
        if normalizer not in NORMALIZERS:
            raise ValueError(f"normalizer must be one of {NORMALIZERS}")
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        self.G = n_components
        self.K = n_components - int(self.has_noise)          # non-noise components
        if self.K < 1:
            raise ValueError(f"{self.edm_type.name} needs G >= {1 + int(self.has_noise)}")
        self.T, self.v, self.r = seq_len, n_states, n_covariates
        self.normalizer = normalizer
        self.straight_through = straight_through
        self.mc_samples = mc_samples
        self.generator: torch.Generator | None = None    # RNG for the "sampled" normaliser

        self.theta_logits = nn.Parameter(0.01 * torch.randn(self.K, self.T, self.v))
        self._build_precision(init_precision)
        self.gating = GatingNetwork(n_components, n_covariates,
                                    noise=self.has_noise, gated_noise=gated_noise)
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("state_mask", torch.ones(self.T, self.v, dtype=torch.bool))

    # ---- hooks implemented by each model file --------------------------------------
    def _build_precision(self, init: float) -> None:
        raise NotImplementedError

    def precision(self) -> torch.Tensor:
        """λ broadcast to shape (K, T), all >= 0."""
        raise NotImplementedError

    def n_precision_params(self) -> int:
        raise NotImplementedError

    def reset_precision(self, N: torch.Tensor, M: torch.Tensor) -> None:
        """Closed-form λ from (K, T) weighted position counts N and mismatch counts M."""
        raise NotImplementedError

    # ---- central sequences ---------------------------------------------------------
    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(value)

    def _masked_logits(self) -> torch.Tensor:
        # States never observed at time t can never be the mode there (Section 4.1.2).
        return self.theta_logits.masked_fill(~self.state_mask, float("-inf"))

    def theta(self, hard: bool = False) -> torch.Tensor:
        """Soft (or hard) central sequences, shape (K, T, v)."""
        logits = self._masked_logits()
        if hard:
            return F.one_hot(logits.argmax(-1), self.v).to(logits.dtype)
        p = torch.softmax(logits / self.temperature, dim=-1)
        if self.straight_through:
            p = F.one_hot(p.argmax(-1), self.v).to(p.dtype) + p - p.detach()
        return p

    def central_sequences(self) -> torch.Tensor:
        """Hard central sequences θ̂_g, shape (K, T)."""
        return self._masked_logits().argmax(-1)

    # ---- likelihood ----------------------------------------------------------------
    @staticmethod
    def _state_log_weights(lam: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """-λ_{g,t}(1 - θ̃_{g,t,k}), shape (K, T, v): log f̃_g(s) = Σ_t of this at k = s_t."""
        return -lam.unsqueeze(-1) * (1.0 - p)

    def log_normalizer(self, lam: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """log Ψ per non-noise component, shape (K,)."""
        if self.normalizer == "hamming":
            # log((v-1)e^{-λ} + 1) = softplus(log(v-1) - λ), numerically stable for large λ
            return F.softplus(math.log(self.v - 1) - lam).sum(-1)
        log_w = self._state_log_weights(lam, p)                        # (K, T, v)
        log_z = torch.logsumexp(log_w, dim=-1).sum(-1)                 # exact log Z_g
        if self.normalizer == "exact" or not (torch.is_grad_enabled() and log_w.requires_grad):
            return log_z
        # Equation 18.13: ∇ log Z_g = E_{s ~ f_g}[∇ log f̃_g(s)]. Average log f̃_g over sequences
        # drawn from f_g (no gradient through the draw) so autograd returns the Monte Carlo
        # estimate of that expectation. Adding it as (x - x.detach()) leaves the forward value
        # at the exact log Z_g, which the responsibilities and the reported loss still need.
        samples = self._sample_from_log_weights(log_w.detach(), self.mc_samples, self.generator)
        x_neg = F.one_hot(samples, self.v).to(log_w.dtype)            # (K, S, T, v)
        neg_phase = (x_neg * log_w.unsqueeze(1)).sum((-1, -2)).mean(1)  # (1/S) Σ_j log f̃_g(s̃_j)
        return log_z.detach() + (neg_phase - neg_phase.detach())

    @staticmethod
    def _sample_from_log_weights(log_w: torch.Tensor, n: int,
                                 generator: torch.Generator | None = None) -> torch.Tensor:
        """n draws from each component, shape (K, n, T).

        A component is a product of independent categoricals over time points, so each s_t is
        drawn directly from softmax(log_w[g, t]); the samples are exact, with no Markov chain.
        """
        K, T, v = log_w.shape
        probs = torch.softmax(log_w, dim=-1).reshape(K * T, v)
        draws = torch.multinomial(probs, n, replacement=True, generator=generator)  # (K*T, n)
        return draws.reshape(K, T, n).transpose(1, 2)

    @torch.no_grad()
    def sample_components(self, n: int, hard: bool = True,
                          generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw n sequences from every non-noise component f_g, shape (K, n, T).

        hard=True samples the reported model (one-hot θ); hard=False samples the normalised
        soft-θ̃ component, the distribution the "sampled" normaliser draws from in training.
        """
        log_w = self._state_log_weights(self.precision(), self.theta(hard))
        return self._sample_from_log_weights(log_w, n, generator)

    def component_log_prob(self, X: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """log f_g(s_i) for the non-noise components, shape (n, K). X is one-hot (n, T, v)."""
        p = self.theta(hard)                                           # (K, T, v)
        lam = self.precision()                                         # (K, T)
        agree = X.flatten(1) @ (lam.unsqueeze(-1) * p).flatten(1).T    # Σ_t λ_kt <x_t, θ̃_kt>
        dist = lam.sum(-1) - agree                                     # weighted Hamming
        return -dist - self.log_normalizer(lam, p)

    def log_joint(self, X, covariates=None, hard: bool = False) -> torch.Tensor:
        """log τ_g(x_i) + log f_g(s_i), shape (n, G); noise component last."""
        n = X.shape[0]
        log_f = self.component_log_prob(X, hard)
        if self.has_noise:
            log_f = torch.cat([log_f, log_f.new_full((n, 1), -self.T * math.log(self.v))], dim=1)
        return self.gating(covariates, n) + log_f

    def forward(self, X, covariates=None, hard: bool = False) -> torch.Tensor:
        """Per-sequence log-likelihood, shape (n,)."""
        return torch.logsumexp(self.log_joint(X, covariates, hard), dim=1)

    @torch.no_grad()
    def posterior(self, X, covariates=None, hard: bool = True) -> torch.Tensor:
        """Responsibilities ẑ_ig (E-step, Equation 5); sampling weights do not enter."""
        return torch.softmax(self.log_joint(X, covariates, hard), dim=1)

    def n_parameters(self, states_per_time: torch.Tensor) -> int:
        """Free parameters for the BIC (Table A1 plus gating, Appendix A)."""
        k_theta = self.K * int((states_per_time - 1).sum())
        return k_theta + self.n_precision_params() + self.gating.n_parameters()

    # ---- initialisation ------------------------------------------------------------
    @torch.no_grad()
    def initialize(self, X, weights, generator=None, *,
                   logit_scale: float = 3.0, noise_init: float = 0.05) -> None:
        """Weighted k-modes start, then closed-form λ and mixing proportions from that partition."""
        self.state_mask.copy_(X.sum(0) > 0)
        modes, labels = weighted_kmodes(X, weights, self.K, generator=generator)
        H = F.one_hot(modes, self.v).to(X.dtype)                        # (K, T, v)
        self.theta_logits.copy_(logit_scale * H)

        Z = F.one_hot(labels, self.K).to(X.dtype) * weights[:, None]    # (n, K)
        counts = torch.einsum("nk,ntv->ktv", Z, X)                      # weighted state counts
        N = Z.sum(0)[:, None].expand(-1, self.T)                        # (K, T)
        M = N - (counts * H).sum(-1)                                    # weighted mismatches
        self.reset_precision(N, M)

        props = (Z.sum(0) / weights.sum()).clamp_min(1e-3)
        props = props / props.sum()
        if self.has_noise:
            props = torch.cat([(1 - noise_init) * props, props.new_tensor([noise_init])])
        self.gating.initialize(props.log())