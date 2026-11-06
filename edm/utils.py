"""Data helpers, closed-form precision estimates and the k-modes initialiser."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def one_hot(sequences: torch.Tensor, n_states: int, dtype=torch.float32) -> torch.Tensor:
    """(n, T) integer states -> (n, T, v) one-hot."""
    return F.one_hot(sequences.long(), n_states).to(dtype)


def normalize_weights(weights, n: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Sampling weights rescaled to sum to n (Section 3.2); all ones if None."""
    if weights is None:
        return torch.ones(n, dtype=dtype, device=device)
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    if w.shape != (n,) or bool((w <= 0).any()):
        raise ValueError("weights must be positive with shape (n,)")
    return w * (n / w.sum())


def states_per_time(sequences: torch.Tensor, n_states: int) -> torch.Tensor:
    """v_t: number of distinct states observed at each time point (used in the BIC count)."""
    return (F.one_hot(sequences.long(), n_states).sum(0) > 0).sum(-1)


def collapse_duplicates(sequences, covariates, weights):
    """Merge identical (sequence, covariate) rows and sum their weights.

    The weighted (pseudo) likelihood is unchanged, but each epoch touches fewer rows.
    """
    T = sequences.shape[1]
    key = sequences.double()
    if covariates is not None:
        key = torch.cat([key, covariates.double()], dim=1)
    uniq, inverse = torch.unique(key, dim=0, return_inverse=True)
    w = weights.new_zeros(uniq.shape[0]).index_add_(0, inverse, weights)
    seq_u = uniq[:, :T].long()
    cov_u = None if covariates is None else uniq[:, T:].to(covariates.dtype)
    return seq_u, cov_u, w


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def closed_form_precision(N, M, n_states: int, min_value: float = 1e-3, max_value: float = 10.0):
    """MLE of λ given N weighted positions containing M weighted mismatches (Appendix B):

        λ = max(0, log(v - 1) + log(N / M - 1))

    clamped to [min_value, max_value] so it can seed a softplus parameter.
    """
    eps = 1e-12
    rate = M / N.clamp_min(eps)
    lam = torch.log((n_states - 1) * (1 - rate).clamp_min(eps)) - torch.log(rate.clamp_min(eps))
    lam = torch.where(N > 0, lam, torch.ones_like(lam))
    return lam.clamp(min_value, max_value)


def weighted_kmodes(X, weights, K: int, n_iter: int = 25, generator=None):
    """Weighted k-modes under the Hamming distance, with k-means++ seeding.

    This is a CC model fitted by classification EM with λ = 1 and equal mixing
    proportions: the hard counterpart the paper initialises from (Section 4.2).
    Returns modes (K, T) and labels (n,).
    """
    S = X.argmax(-1)                                            # (n, T)
    first = torch.multinomial(weights, 1, generator=generator)[0]
    centres = [S[first]]
    d = (S != S[first]).sum(-1).to(weights.dtype)
    for _ in range(1, K):
        probs = weights * d.square()
        if probs.sum() <= 0:
            probs = weights
        j = torch.multinomial(probs, 1, generator=generator)[0]
        centres.append(S[j])
        d = torch.minimum(d, (S != S[j]).sum(-1).to(weights.dtype))
    modes = torch.stack(centres)                                # (K, T)

    for _ in range(n_iter):
        labels = (S[:, None, :] != modes[None]).sum(-1).argmin(1)
        Z = F.one_hot(labels, K).to(X.dtype) * weights[:, None]
        new = torch.einsum("nk,ntv->ktv", Z, X).argmax(-1)    # weighted mode per cluster
        empty = Z.sum(0) == 0
        new[empty] = modes[empty]
        if torch.equal(new, modes):
            break
        modes = new
    labels = (S[:, None, :] != modes[None]).sum(-1).argmin(1)
    return modes, labels