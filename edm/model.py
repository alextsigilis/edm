"""EDModel: one entry point for the whole MEDseq family, selected by EDMType."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cc import CCModel
from .ccn import CCNModel
from .cu import CUModel
from .cun import CUNModel
from .types import EDMType
from .uc import UCModel
from .ucn import UCNModel
from .uu import UUModel
from .uun import UUNModel
from .utils import collapse_duplicates, normalize_weights, one_hot, states_per_time

MODEL_REGISTRY = {
    EDMType.CC: CCModel,
    EDMType.UC: UCModel,
    EDMType.CU: CUModel,
    EDMType.UU: UUModel,
    EDMType.CCN: CCNModel,
    EDMType.UCN: UCNModel,
    EDMType.CUN: CUNModel,
    EDMType.UUN: UUNModel,
}


class EDModel(nn.Module):
    """Mixture of exponential-distance models for equal-length categorical sequences.

    >>> m = EDModel(EDMType.UUN, n_components=11, seq_len=70, n_states=6, n_covariates=1)
    >>> m.fit(S, weights=w, covariates=gcse5eq, n_init=3)
    >>> m.bic(), m.predict(S, gcse5eq), m.central_sequences()

    Components are indexed 0..G-1; for noise types the noise component is the last one.
    Extra keyword arguments go to the underlying model:
    gated_noise, normalizer ("hamming", "exact" or "sampled"), mc_samples, straight_through,
    init_precision.
    Covariates must be numeric (n, r), e.g. 0/1 dummies with reference levels dropped.
    """

    def __init__(self, edm_type, n_components: int, seq_len: int, n_states: int,
                 n_covariates: int = 0, **model_kwargs):
        super().__init__()
        self.edm_type = EDMType[edm_type] if isinstance(edm_type, str) else EDMType(edm_type)
        self._kwargs = dict(n_components=n_components, seq_len=seq_len, n_states=n_states,
                            n_covariates=n_covariates, **model_kwargs)
        self.model = MODEL_REGISTRY[self.edm_type](**self._kwargs)
        self.log_likelihood_ = None
        self.n_obs_ = None
        self.states_per_time_ = None
        self.history_ = []

    def forward(self, X, covariates=None, hard: bool = False):
        """Per-sequence log-likelihood for one-hot X of shape (n, T, v)."""
        return self.model(X, covariates, hard)

    # ---- training ------------------------------------------------------------------
    def fit(
        self,
        sequences,
        weights=None,
        covariates=None,
        *,
        epochs: int = 600,
        lr: float = 0.05,
        temperature: tuple[float, float] = (1.0, 0.05),
        anneal_frac: float = 0.75,
        n_init: int = 3,
        batch_size: int | None = None,
        tol: float = 1e-7,
        patience: int = 50,
        noise_init: float = 0.05,
        collapse: bool = True,
        seed: int | None = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
    ) -> "EDModel":
        """Maximise (1/n) Σ_i w_i log f(s_i) with Adam.

        τ is annealed geometrically from temperature[0] to temperature[1] over the first
        anneal_frac of the epochs, then held while early stopping may trigger. The restart
        with the highest hard-θ log-likelihood is kept.
        """
        device = torch.device("cpu" if device is None else device)
        v, T, r = self._kwargs["n_states"], self._kwargs["seq_len"], self._kwargs["n_covariates"]
        S = torch.as_tensor(sequences, dtype=torch.long, device=device)
        if S.ndim != 2 or S.shape[1] != T:
            raise ValueError(f"sequences must have shape (n, {T})")
        if int(S.min()) < 0 or int(S.max()) >= v:
            raise ValueError(f"states must be integers in [0, {v})")
        n = S.shape[0]
        w = normalize_weights(weights, n, dtype, device)
        cov = None if covariates is None else torch.as_tensor(covariates, dtype=dtype, device=device)
        if (cov is None) != (r == 0):
            raise ValueError("pass covariates exactly when n_covariates > 0")
        if cov is not None and cov.shape != (n, r):
            raise ValueError(f"covariates must have shape (n, {r})")

        S_fit, cov_fit, w_fit = collapse_duplicates(S, cov, w) if collapse else (S, cov, w)
        X = one_hot(S_fit, v, dtype)

        best = None
        for run in range(n_init):
            gen = None if seed is None else torch.Generator(device=device).manual_seed(seed + run)
            model = MODEL_REGISTRY[self.edm_type](**self._kwargs).to(device=device, dtype=dtype)
            model.initialize(X, w_fit, generator=gen, noise_init=noise_init)
            model.generator = gen                  # Monte Carlo draws for normalizer="sampled"
            history = self._train(model, X, w_fit, cov_fit, epochs=epochs, lr=lr,
                                  temperature=temperature, anneal_frac=anneal_frac,
                                  batch_size=batch_size, tol=tol, patience=patience,
                                  verbose=verbose, generator=gen)
            model.generator = None
            with torch.no_grad():
                ll = (w_fit * model(X, cov_fit, hard=True)).sum().item()
            if verbose:
                print(f"[{self.edm_type.name} G={model.G}] run {run}: log-lik {ll:.2f} "
                      f"after {len(history)} epochs")
            if best is None or ll > best[0]:
                best = (ll, model, history)

        self.log_likelihood_, self.model, self.history_ = best
        self.n_obs_ = n
        self.states_per_time_ = states_per_time(S, v)
        return self

    @staticmethod
    def _train(model, X, w, cov, *, epochs, lr, temperature, anneal_frac, batch_size,
               tol, patience, verbose, generator):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        t_start, t_end = temperature
        n_anneal = max(1, int(anneal_frac * epochs))
        m = X.shape[0]
        history, best, stall = [], math.inf, 0
        for epoch in range(epochs):
            tau = t_start * (t_end / t_start) ** min(1.0, epoch / max(1, n_anneal - 1))
            model.set_temperature(tau)
            if batch_size is None:
                batches = [slice(None)]
            else:
                batches = torch.randperm(m, device=X.device, generator=generator).split(batch_size)
            total = 0.0
            for idx in batches:
                wb = w[idx]
                ll = model(X[idx], None if cov is None else cov[idx])
                loss = -(wb * ll).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * wb.sum().item()
            total /= w.sum().item()
            history.append(total)
            if verbose and epoch % 100 == 0:
                print(f"  epoch {epoch:4d}  loss {total:.5f}  tau {tau:.3f}")
            if epoch >= n_anneal:                      # only stop once τ is fixed
                if best == math.inf or total < best - tol * abs(best):
                    best, stall = total, 0
                else:
                    stall += 1
                    if stall >= patience:
                        break
        return history

    # ---- inference -----------------------------------------------------------------
    def _prep(self, sequences, covariates):
        p = next(self.model.parameters())
        S = torch.as_tensor(sequences, dtype=torch.long, device=p.device)
        X = one_hot(S, self.model.v, p.dtype)
        cov = None if covariates is None else torch.as_tensor(covariates, dtype=p.dtype, device=p.device)
        return X, cov

    @torch.no_grad()
    def predict_proba(self, sequences, covariates=None) -> torch.Tensor:
        X, cov = self._prep(sequences, covariates)
        return self.model.posterior(X, cov, hard=True)

    def predict(self, sequences, covariates=None) -> torch.Tensor:
        """MAP cluster labels."""
        return self.predict_proba(sequences, covariates).argmax(1)

    @torch.no_grad()
    def log_likelihood(self, sequences, weights=None, covariates=None) -> float:
        X, cov = self._prep(sequences, covariates)
        w = normalize_weights(weights, X.shape[0], X.dtype, X.device)
        return (w * self.model(X, cov, hard=True)).sum().item()

    def n_parameters(self) -> int:
        if self.states_per_time_ is None:
            raise RuntimeError("call fit() first")
        return self.model.n_parameters(self.states_per_time_)

    def bic(self) -> float:
        """2ℓ - k·log(n) on the pseudo log-likelihood (larger is better, as in the paper)."""
        if self.log_likelihood_ is None:
            raise RuntimeError("call fit() first")
        return 2 * self.log_likelihood_ - self.n_parameters() * math.log(self.n_obs_)

    def central_sequences(self) -> torch.Tensor:
        return self.model.central_sequences()

    def sample_components(self, n: int, generator=None) -> torch.Tensor:
        """n sequences drawn from each fitted non-noise component, shape (K, n, T)."""
        return self.model.sample_components(n, hard=True, generator=generator)

    def precision(self) -> torch.Tensor:
        return self.model.precision().detach()

    def gating_coefficients(self) -> torch.Tensor | None:
        return self.model.gating.coefficients()

    @property
    def noise_component(self) -> int | None:
        return self.model.G - 1 if self.model.has_noise else None


def _redundant(edm_type: EDMType, G: int) -> bool:
    # UC/UU at G = 1 equal CC/CU; UCN/UUN at G = 2 equal CCN/CUN (Section 3.3)
    return (edm_type.name in ("UC", "UU") and G == 1) or (edm_type.name in ("UCN", "UUN") and G == 2)


def select_by_bic(sequences, n_states: int, n_components, *, edm_types=tuple(EDMType),
                  weights=None, covariates=None, model_kwargs=None, **fit_kwargs):
    """Fit every allowable (type, G) pair; return (best_model, {(type_name, G): BIC})."""
    S = torch.as_tensor(sequences, dtype=torch.long)
    r = 0 if covariates is None else torch.as_tensor(covariates).shape[1]
    best, table = None, {}
    for edm_type in edm_types:
        for G in n_components:
            if _redundant(edm_type, G):
                continue
            try:
                m = EDModel(edm_type, G, S.shape[1], n_states, r, **(model_kwargs or {}))
            except ValueError:          # e.g. noise model with G = 1, or covariates with G < 2
                continue
            m.fit(S, weights, covariates, **fit_kwargs)
            table[(edm_type.name, G)] = m.bic()
            if best is None or table[(edm_type.name, G)] > best.bic():
                best = m
    return best, table