"""Tests for normalizer="sampled": the Monte Carlo gradient of log Z (Deep Learning, eq. 18.13).

Run from the repository root with

    python -m unittest discover tests
"""
import math
import unittest

import torch
import torch.nn.functional as F

from edm import MODEL_REGISTRY, EDModel, EDMType
from edm.utils import one_hot

DTYPE = torch.float64
G, T, V, R = 4, 8, 4, 2        # components, sequence length, states, covariates
N = 120                        # observations


def make_model(edm_type, normalizer, mc_samples=8, seed=0):
    """A model with soft θ̃, varied λ and one masked state, identical for a given seed."""
    torch.manual_seed(seed)
    model = MODEL_REGISTRY[edm_type](G, T, V, R, normalizer=normalizer,
                                     mc_samples=mc_samples).to(DTYPE)
    with torch.no_grad():
        model.theta_logits.normal_(0.0, 2.0)
        model.raw_precision.normal_(0.5, 1.0)
        model.state_mask[0, -1] = False
    model.set_temperature(0.7)
    return model


def make_data(seed=0):
    gen = torch.Generator().manual_seed(seed)
    S = torch.randint(0, V, (N, T), generator=gen)
    cov = torch.randn(N, R, generator=gen, dtype=DTYPE)
    w = torch.rand(N, generator=gen, dtype=DTYPE) + 0.5
    return one_hot(S, V, DTYPE), cov, w


def flat_grad(model, loss):
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(params, grads)])


def normalizer_loss(model, data):
    del data
    coef = torch.linspace(0.5, 2.0, model.K, dtype=DTYPE)
    return (coef * model.log_normalizer(model.precision(), model.theta())).sum()


def mixture_loss(model, data):
    X, cov, w = data
    return -(w * model(X, cov)).sum() / w.sum()


class SampledGradientTest(unittest.TestCase):
    """The sampled gradient is an unbiased estimate of the closed-form gradient."""

    def test_unbiased(self):
        data = make_data()
        n_estimates = 300
        for edm_type in (EDMType.UUN, EDMType.CC):
            for loss_fn in (normalizer_loss, mixture_loss):
                with self.subTest(edm_type=edm_type.name, loss=loss_fn.__name__):
                    exact = make_model(edm_type, "exact")
                    target = flat_grad(exact, loss_fn(exact, data))
                    sampled = make_model(edm_type, "sampled", mc_samples=8)
                    sampled.generator = torch.Generator().manual_seed(1)
                    est = torch.stack([flat_grad(sampled, loss_fn(sampled, data))
                                       for _ in range(n_estimates)])

                    noisy = est.std(0) > 0
                    self.assertTrue(noisy.any(), "no gradient came from the samples")
                    # Parameters that do not reach log Z (gating, data term) are exact.
                    torch.testing.assert_close(est.mean(0)[~noisy], target[~noisy],
                                               rtol=0, atol=1e-10)
                    # The rest average to the closed form: z-scores look standard normal.
                    se = est.std(0)[noisy] / math.sqrt(n_estimates)
                    z = (est.mean(0)[noisy] - target[noisy]) / se
                    self.assertLess(z.abs().max().item(), 5.0)
                    self.assertGreater(z.abs().mean().item(), 0.5)   # E|z| = 0.80 if unbiased
                    self.assertLess(z.abs().mean().item(), 1.1)
                    self.assertLess(z.mean().abs().item(), 0.4)

    def test_error_shrinks_with_more_samples(self):
        data = make_data()
        exact = make_model(EDMType.UU, "exact")
        target = flat_grad(exact, normalizer_loss(exact, data))
        errors = []
        for mc_samples in (4, 4096):
            sampled = make_model(EDMType.UU, "sampled", mc_samples=mc_samples)
            sampled.generator = torch.Generator().manual_seed(2)
            grad = flat_grad(sampled, normalizer_loss(sampled, data))
            errors.append(((grad - target).norm() / target.norm()).item())
        # The error scales as 1/sqrt(S), so it should be ~32x smaller; require at least 8x.
        self.assertLess(errors[1], errors[0] / 8)
        self.assertLess(errors[1], 0.05)


class SampledValueTest(unittest.TestCase):
    """Only the gradient is estimated: values are those of the exact normaliser."""

    def test_forward_value_is_exact(self):
        X, cov, _ = make_data()
        exact = make_model(EDMType.UUN, "exact")
        sampled = make_model(EDMType.UUN, "sampled")
        self.assertTrue(torch.equal(sampled(X, cov), exact(X, cov)))
        with torch.no_grad():
            self.assertTrue(torch.equal(sampled(X, cov), exact(X, cov)))

    def test_no_samples_without_gradient(self):
        X, cov, _ = make_data()
        model = make_model(EDMType.UUN, "sampled")
        model.generator = torch.Generator().manual_seed(3)
        state = model.generator.get_state()
        with torch.no_grad():
            model(X, cov)
            model.posterior(X, cov)
        self.assertTrue(torch.equal(model.generator.get_state(), state))

    def test_samples_drawn_with_gradient(self):
        X, cov, _ = make_data()
        model = make_model(EDMType.UUN, "sampled")
        model.generator = torch.Generator().manual_seed(3)
        state = model.generator.get_state()
        model(X, cov).sum().backward()
        self.assertFalse(torch.equal(model.generator.get_state(), state))


class SampleComponentsTest(unittest.TestCase):
    """Draws follow each component's per-time-point categorical distribution."""
    n = 40_000

    @staticmethod
    def frequencies(draws):
        return torch.stack([(draws == k).to(DTYPE).mean(1) for k in range(V)], dim=-1)

    def test_soft_marginals(self):
        model = make_model(EDMType.UUN, "sampled")
        draws = model.sample_components(self.n, hard=False,
                                        generator=torch.Generator().manual_seed(4))
        self.assertEqual(tuple(draws.shape), (model.K, self.n, T))
        with torch.no_grad():
            lam, theta = model.precision(), model.theta()
            probs = torch.softmax(-lam.unsqueeze(-1) * (1 - theta), dim=-1)
        torch.testing.assert_close(self.frequencies(draws), probs, rtol=0, atol=0.015)

    def test_hard_marginals(self):
        model = make_model(EDMType.UU, "hamming")
        draws = model.sample_components(self.n, generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            lam = model.precision()
            p_match = 1 / ((V - 1) * torch.exp(-lam) + 1)        # P(s_t = θ_t)
            theta_hat = model.central_sequences()
        match = (draws == theta_hat.unsqueeze(1)).to(DTYPE).mean(1)
        torch.testing.assert_close(match, p_match, rtol=0, atol=0.015)


def synthetic_sequences(n=150, seq_len=10, n_states=4, seed=0):
    """Two noisy copies of random central sequences, 20% of positions resampled."""
    gen = torch.Generator().manual_seed(seed)
    centres = torch.randint(0, n_states, (2, seq_len), generator=gen)
    S = centres[torch.randint(0, 2, (n,), generator=gen)].clone()
    flip = torch.rand(n, seq_len, generator=gen) < 0.2
    S[flip] = torch.randint(0, n_states, (int(flip.sum()),), generator=gen)
    return S


class FitTest(unittest.TestCase):

    def fit(self, seed):
        model = EDModel(EDMType.CC, 2, 10, 4, normalizer="sampled", mc_samples=4)
        return model.fit(synthetic_sequences(), n_init=1, epochs=40, seed=seed)

    def test_reproducible_from_seed(self):
        a, b = self.fit(seed=7), self.fit(seed=7)
        self.assertEqual(a.log_likelihood_, b.log_likelihood_)
        self.assertTrue(torch.equal(a.central_sequences(), b.central_sequences()))
        self.assertIsNone(a.model.generator)
        self.assertEqual(tuple(a.sample_components(5).shape), (2, 5, 10))

    def test_validation(self):
        with self.assertRaises(ValueError):
            EDModel(EDMType.CC, 2, 10, 4, normalizer="mc")
        with self.assertRaises(ValueError):
            EDModel(EDMType.CC, 2, 10, 4, normalizer="sampled", mc_samples=0)


if __name__ == "__main__":
    unittest.main()
