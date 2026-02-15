#!/usr/bin/env python3
# ======================================================================
# 14_power_01_sample_size_and_power_curves.py
# ======================================================================
# Demonstrate statistical power analysis:
#   1. Compute required sample size for a two-sample t-test.
#   2. Compute required sample size for a proportion test (A/B test).
#   3. Plot power curves as a function of sample size.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 3 — Statistical Experiments and Significance Testing).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def power_ttest(n, delta, sigma=1.0, alpha=0.05):
    """Compute power of a two-sided two-sample t-test."""
    se = sigma * np.sqrt(2 / n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_effect = delta / se
    power = 1 - stats.norm.cdf(z_crit - z_effect) + stats.norm.cdf(-z_crit - z_effect)
    return power


def sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80):
    """Compute minimum n per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(np.ceil(n))


def sample_size_proportion(p1, p2, alpha=0.05, power=0.80):
    """Compute minimum n per group for a two-proportion z-test."""
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    numer = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numer / (p1 - p2) ** 2
    return int(np.ceil(n))


def main():
    print("Power Analysis")
    print("=" * 55)

    # ── 1. Two-sample t-test sample size ──
    delta = 0.5   # Cohen's d = 0.5 (medium effect)
    n_req = sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80)
    print(f"\n1. Two-sample t-test  (delta={delta}, sigma=1, alpha=0.05, power=0.80)")
    print(f"   Required n per group: {n_req}")

    # ── 2. Proportion test sample size (A/B test) ──
    p1, p2 = 0.0121, 0.011
    n_prop = sample_size_proportion(p1, p2, alpha=0.05, power=0.80)
    print(f"\n2. Proportion test  (p1={p1}, p2={p2})")
    print(f"   Required n per group: {n_prop:,}")

    p1b, p2b = 0.0165, 0.011
    n_prop2 = sample_size_proportion(p1b, p2b, alpha=0.05, power=0.80)
    print(f"\n   Larger effect  (p1={p1b}, p2={p2b})")
    print(f"   Required n per group: {n_prop2:,}")

    # ── 3. Power curves ──
    ns = np.arange(10, 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    for d, ls in [(0.2, '--'), (0.5, '-'), (0.8, ':')]:
        powers = [power_ttest(n, d) for n in ns]
        ax.plot(ns, powers, ls, label=f'd = {d}')

    ax.axhline(0.80, color='grey', linestyle='-.', alpha=0.5, label='Power = 0.80')
    ax.set_xlabel('Sample size per group (n)')
    ax.set_ylabel('Power')
    ax.set_title('Power Curves for Two-Sample t-Test (alpha = 0.05)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
