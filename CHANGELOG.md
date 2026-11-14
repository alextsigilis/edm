# Changelog

## [Unreleased]

### Added

- **Monte Carlo gradient for the normalising constant: `normalizer="sampled"`.** The gradient of each component's log normaliser is now estimated from samples, using equation 18.13 of Goodfellow, Bengio & Courville, *Deep Learning*, [chapter 18](https://www.deeplearningbook.org/contents/partition.html):

  $$\nabla \log Z_g = \mathbb{E}_{s \sim f_g}\big[\nabla \log \tilde f_g(s)\big], \qquad \log \tilde f_g(s) = -\sum_t \lambda_{g,t}\big(1 - \langle x_t, \tilde\theta_{g,t}\rangle\big)$$

  - **Positive and negative phase.** The data term (positive phase) is unchanged. At every training step, `mc_samples` sequences are drawn from each non-noise component, and the mean of ∇ log f̃_g over them replaces ∇ log Z_g (negative phase).
  - **Exact sampling.** Each component is a product of independent categoricals over time points, so samples are drawn exactly. No Markov chain is needed; for this model a single Gibbs sweep of the book's Algorithm 18.1 would already be exact.
  - **Exact forward value.** Only the gradient is estimated. The value of log Z_g stays the closed form, via `log_z.detach() + (neg_phase - neg_phase.detach())`. Three things need that value: the mixture responsibilities, the reported loss and early stopping.
  - **Which gradient it targets.** The estimator is unbiased for the gradient of the `"exact"` normaliser, which is the true normaliser of f̃_g for soft θ̃. The `"hamming"` constant is not the normaliser of f̃_g when θ̃ is soft, so equation 18.13 does not apply to it.
  - **When samples are drawn.** Only when a gradient is needed. Under `torch.no_grad()` (prediction, posteriors, reporting), `"sampled"` returns the exact value and draws nothing.
- **`mc_samples` model option** (default 32): the number of sequences drawn per component per step. Pass it to `EDModel(...)` with the other model options.
- **Seeded draws.** `fit` passes each restart's seeded `torch.Generator` to the model, so a `"sampled"` fit is reproducible from `seed`. It detaches the generator afterwards.
- **`EDModel.sample_components(n)` and `BaseEDM.sample_components(n, hard=True)`.** These draw `n` sequences from each non-noise component and return a tensor of shape `(K, n, T)`. Set `hard=False` to sample the soft-θ̃ components used in training.
- **`main.py` options** `--normalizer {hamming,exact,sampled}` and `--mc-samples N`.
- **`edm.NORMALIZERS`**, the tuple of valid normaliser names.
- **`compare_normalizers.py`.** For one model type (`--type`, default UUN), it sweeps G = 5..15 (`--g-min`, `--g-max`) and fits each G with the `hamming`, `exact` and `sampled` normalisers, using the same seed. It saves `results/normalizers_<TYPE>.csv` and `.png`, prints the BIC table and each normaliser's best G, and opens the BIC-versus-G plot (`--no-show` turns that off). `--plot-only` re-plots a saved table.
- **Cluster plot in `compare_normalizers.py`, after Figure 7 of Murphy et al. (2021).** For each normaliser, the model at its BIC-optimal G is drawn as one panel. Each of the n = 712 observed sequences is one row, coloured by state, and rows are grouped by MAP cluster with the noise component last.
  - **Ordering.** Clusters are seriated by their central sequences. Rows within a cluster start from the member nearest the central sequence and follow a greedy nearest-neighbour chain on Hamming distance, which approximates the paper's TSP seriation.
  - **Labels.** Long SPS labels wrap after three spells and are spread apart with leader lines.
  - **Output.** The figure is saved as `results/normalizers_<TYPE>_clusters.png`. A table of cluster sizes and mean months per state is printed for each panel.
  - **Stored settings.** The sweep table now records the fit settings (covariates, weights, gate, `mc_samples`, restarts, epochs, lr, seed). `--plot-only` uses them to refit the three best models deterministically, without rerunning the sweep.
  - **Example figures.** The README shows `results/normalizers_UUN.png` and `results/normalizers_UUN_clusters.png` from a default UUN run, with notes on what they show.
  - **Colours.** SC is blue, EM green, TR magenta and JL yellow, as in the paper; FE is aqua and HE red. No set of six hues separates every pair. The closest pair (magenta and red) goes to TR and HE, which rarely border each other, and the SPS labels and the printed table repeat the information as text.
- **Tests: `tests/test_sampled_normalizer.py`.** They use the standard-library `unittest`, run in about a second, and are run with `python -m unittest discover tests`. They check that:
  - the sampled gradient is unbiased, for log Z alone and for the full mixture loss, on UUN and CC models. The mean of 300 estimates must match the closed-form gradient with standard-normal z-scores. Parameters that do not reach log Z must match exactly.
  - the error shrinks roughly as 1/√S as `mc_samples` grows.
  - forward values are bit-identical to `"exact"`, both with and without gradients.
  - no random numbers are drawn under `torch.no_grad()`, and they are drawn when a gradient is needed.
  - `sample_components` follows the soft-θ̃ categoricals, and the hard-θ match probability 1/((v−1)e^{−λ}+1).
  - a `"sampled"` fit is reproducible from `seed` and detaches its generator afterwards.
  - invalid `normalizer` and `mc_samples` values raise `ValueError`.

  Before committing, the tests were checked against deliberately wrong samplers: uniform draws, draws at the wrong temperature, and draws from hard θ. Each one fails the unbiasedness, error-shrinkage and marginal tests. A sampler that reuses a single draw for every sample is still unbiased, so it passes the unbiasedness test but fails the error-shrinkage test.

### Changed

- **New validation.** `BaseEDM` raises `ValueError` for `mc_samples < 1`. The message for an unknown normaliser now lists all three options.
- **Docs.** The `edm/base.py` docstring and the README (CLI table, Python API, normaliser options) describe the new option.
- **Unchanged defaults.** The default stays `normalizer="hamming"`, and the `"hamming"` and `"exact"` code paths behave exactly as before.

### Verification

Checks were run in float64 on random UUN and CC models with soft θ̃, one masked state and covariates:

- **Exact forward value.** `"sampled"` and `"exact"` give bit-identical log-likelihoods.
- **Formula check.** With the sampled one-hots replaced by their expectation, the negative-phase term reproduces the closed-form gradient to 1e-12.
- **Unbiasedness.** Over 4,000 independent estimates (S = 16), the mean matched the closed-form gradient of log Z and of the full mixture loss. Per-coordinate z-scores were calibrated: mean |z| was 0.68–0.85 across seeds against 0.80 expected, with mean z ≈ 0. Gradients that do not involve log Z, such as gating and the data term, match exactly.
- **Sampler.** Empirical state frequencies matched the component's categorical probabilities to within 0.003.
- **CLI smoke test.** `main.py --normalizer sampled` ran end to end.

### Notes: MVAD results

These fits used UUN with G = 11, the GCSE5eq covariate, sampling weights, one restart (seed 0), 600 epochs, τ annealed from 1 to 0.05 and no early stopping. The soft objective is the training objective, (1/n) Σ wᵢ log f(sᵢ) with soft θ̃, scaled to a total. The hard log-likelihood is the reported value used for BIC.

| Model | Normaliser | Soft objective | Hard log-lik |
|---|---|---|---|
| UUN, G = 11 | exact | −31,235 | −48,951 |
| UUN, G = 11 | sampled, S = 512 | −31,718 | −52,238 |
| UUN, G = 11 | sampled, S = 32 | −34,616 | −43,014 |
| UUN, G = 11 | sampled, S = 1 | −37,286 | −37,295 |
| UUN, G = 11 | hamming | −37,007 (a different objective: a lower bound) | −37,007 |

- **Sampling optimises the same objective as `"exact"`.** With many samples it reaches nearly the same soft objective. Fewer samples add gradient noise and stop further from that optimum.
- **The hard (reported) log-likelihood is poor for `"exact"`.** The exact objective rewards keeping θ̃ soft, and hardening θ̃ for reporting then loses a lot of likelihood. `"sampled"` inherits this. A noisier gradient (small S) exploits the softness less, which is why its hard log-likelihood looks better.
- **For BIC on MVAD, `"hamming"` is still the best choice.** Use `"sampled"` when you specifically want the score-function / negative-phase gradient. Examples are studying the estimator, or a starting point for variants where log Z has no closed form.
