#!/usr/bin/env python3
# ======================================================================
# 25_bootstrap_01_ci_methods.py
# ======================================================================
# Demonstrate three bootstrap confidence-interval methods:
#   1. Percentile method
#   2. Basic (reverse percentile) method
#   3. BCa (bias-corrected and accelerated) method
#
# The example uses a Poisson sample — a non-normal distribution —
# to highlight when bootstrapping is most useful.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (11_bootstrapping.ipynb).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def bootstrap_percentile_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """Percentile bootstrap CI."""
    n = len(data)
    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi, boot_stats


def bootstrap_basic_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """Basic (reverse-percentile) bootstrap CI."""
    theta_hat = statistic(data)
    n = len(data)
    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    lo = 2 * theta_hat - np.percentile(boot_stats, 100 * (1 - alpha / 2))
    hi = 2 * theta_hat - np.percentile(boot_stats, 100 * alpha / 2)
    return lo, hi, boot_stats


def bootstrap_bca_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """BCa (bias-corrected and accelerated) bootstrap CI."""
    n = len(data)
    theta_hat = statistic(data)

    # Bootstrap distribution
    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])

    # Bias correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # Acceleration factor a — jackknife estimate
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    a_num = np.sum((jack_mean - jack) ** 3)
    a_den = 6 * np.sum((jack_mean - jack) ** 2) ** 1.5
    a = a_num / a_den if a_den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lo = np.percentile(boot_stats, 100 * p_lo)
    hi = np.percentile(boot_stats, 100 * p_hi)
    return lo, hi, boot_stats


def main():
    # ── Sample from a non-normal (Poisson) distribution ──
    lam_true = 3.5
    n = 80
    data = stats.poisson.rvs(lam_true, size=n)
    sample_mean = np.mean(data)

    print("Bootstrap Confidence Intervals for the Mean")
    print("=" * 55)
    print(f"True lambda = {lam_true},  n = {n},  sample mean = {sample_mean:.3f}\n")

    stat_fn = np.mean
    n_boot = 10_000

    lo_p, hi_p, boots = bootstrap_percentile_ci(data, stat_fn, n_boot)
    lo_b, hi_b, _     = bootstrap_basic_ci(data, stat_fn, n_boot)
    lo_bca, hi_bca, _ = bootstrap_bca_ci(data, stat_fn, n_boot)

    print(f"  Percentile  CI : [{lo_p:.3f}, {hi_p:.3f}]")
    print(f"  Basic       CI : [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"  BCa         CI : [{lo_bca:.3f}, {hi_bca:.3f}]")

    # ── Histogram of bootstrap means ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(boots, bins=50, edgecolor='k', alpha=0.7, density=True)
    ax.axvline(sample_mean, color='red', linestyle='--',
               label=f'Sample mean = {sample_mean:.2f}')
    ax.axvline(lo_bca, color='green', linestyle=':',
               label=f'BCa CI [{lo_bca:.2f}, {hi_bca:.2f}]')
    ax.axvline(hi_bca, color='green', linestyle=':')
    ax.set_xlabel('Bootstrap mean')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap Distribution of the Mean  (B = {n_boot:,})')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
