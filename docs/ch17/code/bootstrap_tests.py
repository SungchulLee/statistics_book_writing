#!/usr/bin/env python3
# ======================================================================
# 25_bootstrap_02_hypothesis_tests.py
# ======================================================================
# Bootstrap hypothesis tests:
#   1. Bootstrap test for a single mean.
#   2. Bootstrap test for comparing two means.
#   3. Bootstrap standard error for the median.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 2 — Data and Sampling Distributions).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def bootstrap_mean_test(data, mu_0=0, n_boot=10_000, alpha=0.05):
    """
    Bootstrap test for H0: mean = mu_0.
    Returns p-value (two-sided).
    """
    n = len(data)
    # Centre data under H0
    centered = data - data.mean() + mu_0
    boot_means = np.array([
        np.mean(centered[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    obs_mean = data.mean()
    p_value = np.mean(np.abs(boot_means - mu_0) >= np.abs(obs_mean - mu_0))
    return obs_mean, p_value, boot_means


def bootstrap_two_sample(x, y, n_boot=10_000):
    """
    Bootstrap test for H0: mean(x) = mean(y).
    Returns observed difference, p-value, and bootstrap distribution.
    """
    obs_diff = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n_x = len(x)
    boot_diffs = []
    for _ in range(n_boot):
        perm = pooled[np.random.randint(0, len(pooled), len(pooled))]
        boot_diffs.append(perm[:n_x].mean() - perm[n_x:].mean())
    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, boot_diffs


def bootstrap_se_median(data, n_boot=10_000):
    """Estimate the standard error of the median via bootstrap."""
    n = len(data)
    boot_medians = np.array([
        np.median(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    se = boot_medians.std(ddof=1)
    bias = boot_medians.mean() - np.median(data)
    return se, bias, boot_medians


def main():
    print("Bootstrap Hypothesis Tests")
    print("=" * 55)

    # ── 1. One-sample test ──
    data = np.random.exponential(scale=5, size=50) + 2
    mu_0 = 5.0
    obs, p, boots = bootstrap_mean_test(data, mu_0)
    print(f"\n1. One-sample bootstrap test  (H0: mu = {mu_0})")
    print(f"   Sample mean = {obs:.3f},  p-value = {p:.4f}")

    # ── 2. Two-sample test ──
    x = np.random.normal(52, 10, 40)
    y = np.random.normal(48, 10, 40)
    diff, p2, boots2 = bootstrap_two_sample(x, y)
    print(f"\n2. Two-sample bootstrap test  (H0: mu_x = mu_y)")
    print(f"   Observed diff = {diff:.3f},  p-value = {p2:.4f}")

    # ── 3. Bootstrap SE of the median ──
    income = np.random.lognormal(mean=10.5, sigma=0.8, size=200)
    se_med, bias, boot_med = bootstrap_se_median(income)
    print(f"\n3. Bootstrap SE of the median  (n = {len(income)})")
    print(f"   Median = {np.median(income):,.0f}")
    print(f"   Bootstrap SE = {se_med:,.0f},  Bias = {bias:,.0f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(boots, bins=50, edgecolor='k', alpha=0.7)
    axes[0].axvline(obs, color='red', linestyle='--', label=f'Obs mean={obs:.2f}')
    axes[0].axvline(mu_0, color='green', linestyle=':', label=f'H0: mu={mu_0}')
    axes[0].set_title(f'One-Sample Test (p={p:.3f})')
    axes[0].legend(fontsize=8)

    axes[1].hist(boots2, bins=50, edgecolor='k', alpha=0.7)
    axes[1].axvline(diff, color='red', linestyle='--', label=f'Obs diff={diff:.2f}')
    axes[1].set_title(f'Two-Sample Test (p={p2:.3f})')
    axes[1].legend(fontsize=8)

    axes[2].hist(boot_med, bins=50, edgecolor='k', alpha=0.7)
    axes[2].axvline(np.median(income), color='red', linestyle='--', label='Sample median')
    axes[2].set_title(f'Median SE = {se_med:,.0f}')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
