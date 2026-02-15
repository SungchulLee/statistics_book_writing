#!/usr/bin/env python3
# ======================================================================
# 09_bayesian_01_conjugate_and_mcmc.py
# ======================================================================
# Demonstrate two Bayesian estimation approaches:
#   1. Conjugate prior (Beta–Binomial) — closed-form posterior.
#   2. Grid-based posterior for a Normal mean with known variance.
#
# Both examples show how the posterior concentrates as data accumulate,
# and how prior choice affects the result.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (14_BayesianStatistics.ipynb) and the course's Chapter 6
#          material on Bayesian inference.
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def demo_beta_binomial():
    """Beta–Binomial conjugate model for estimating a proportion p."""
    # Prior: Beta(α₀, β₀)  — weakly informative
    alpha_0, beta_0 = 2, 2

    # Observed data: k successes out of n trials
    n, k = 50, 32

    # Posterior: Beta(α₀ + k, β₀ + n - k)
    alpha_post = alpha_0 + k
    beta_post  = beta_0 + (n - k)

    p = np.linspace(0, 1, 500)
    prior = stats.beta(alpha_0, beta_0).pdf(p)
    likelihood = stats.binom(n, p).pmf(k)
    likelihood /= likelihood.max()                     # scale for plotting
    posterior = stats.beta(alpha_post, beta_post).pdf(p)

    # MAP estimate
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    # Posterior mean
    post_mean = alpha_post / (alpha_post + beta_post)

    print("1. Beta-Binomial Conjugate Model")
    print(f"   Prior:     Beta({alpha_0}, {beta_0})")
    print(f"   Data:      {k} successes in {n} trials")
    print(f"   Posterior:  Beta({alpha_post}, {beta_post})")
    print(f"   MAP estimate   = {map_est:.4f}")
    print(f"   Posterior mean = {post_mean:.4f}")
    print(f"   MLE            = {k/n:.4f}\n")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p, prior, '--', label=f'Prior: Beta({alpha_0},{beta_0})')
    ax.plot(p, likelihood, ':', label='Likelihood (scaled)')
    ax.plot(p, posterior, linewidth=2,
            label=f'Posterior: Beta({alpha_post},{beta_post})')
    ax.axvline(map_est, color='red', linestyle='-.', alpha=0.5,
               label=f'MAP = {map_est:.3f}')
    ax.set_xlabel('p')
    ax.set_ylabel('Density')
    ax.set_title('Beta-Binomial Conjugate Update')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def demo_normal_grid():
    """Grid-based posterior for the mean of a Normal with known variance."""
    sigma = 2.0         # known population std
    mu_true = 5.0       # true (unknown) mean

    # Prior: Normal(μ₀, τ₀²)
    mu_0, tau_0 = 0, 10

    # Observe n data points
    n = 25
    data = np.random.normal(mu_true, sigma, n)
    x_bar = data.mean()

    # Posterior (conjugate): Normal(μ_n, τ_n²)
    tau_n_sq = 1 / (1 / tau_0**2 + n / sigma**2)
    mu_n = tau_n_sq * (mu_0 / tau_0**2 + n * x_bar / sigma**2)
    tau_n = np.sqrt(tau_n_sq)

    grid = np.linspace(-5, 12, 600)
    prior_pdf = stats.norm(mu_0, tau_0).pdf(grid)
    post_pdf  = stats.norm(mu_n, tau_n).pdf(grid)

    print("2. Normal-Normal Conjugate Model")
    print(f"   Prior:      N({mu_0}, {tau_0}^2)")
    print(f"   Data:       n={n}, x_bar={x_bar:.3f}, sigma={sigma}")
    print(f"   Posterior:  N({mu_n:.3f}, {tau_n:.3f}^2)")
    print(f"   95% credible interval: [{mu_n - 1.96*tau_n:.3f}, "
          f"{mu_n + 1.96*tau_n:.3f}]")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, prior_pdf, '--', label=f'Prior N({mu_0}, {tau_0})')
    ax.plot(grid, post_pdf, linewidth=2,
            label=f'Posterior N({mu_n:.2f}, {tau_n:.2f})')
    ax.axvline(mu_true, color='green', linestyle=':', alpha=0.6,
               label=f'True mean = {mu_true}')
    ax.axvline(x_bar, color='red', linestyle='-.', alpha=0.6,
               label=f'Sample mean = {x_bar:.2f}')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel('Density')
    ax.set_title('Normal-Normal Conjugate Update')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("Bayesian Estimation Demonstrations")
    print("=" * 60 + "\n")
    demo_beta_binomial()
    print()
    demo_normal_grid()


if __name__ == "__main__":
    main()
