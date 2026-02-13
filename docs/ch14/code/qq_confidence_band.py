#!/usr/bin/env python3
# ======================================================================
# 26_qq_02_confidence_band_simulation.py
# ======================================================================
# QQ-plot with simulated 95% pointwise confidence band under Normality.
# Steps:
#   * Estimate mean and std from data.
#   * Simulate many normal samples of same size, compute sorted values.
#   * For each order i, take [2.5%, 97.5%] of simulated order stats -> envelope.
# Notes:
#   * Band is pointwise (not simultaneous). Good for visual guidance.
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def qq_with_band(x, B=800, seed=42):
    x = np.asarray(x, dtype=float)
    n = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    # Theoretical quantiles for p_i = (i-0.5)/n
    i = np.arange(1, n+1)
    p = (i - 0.5) / n
    q_theor = stats.norm.ppf(p)

    # Observed ordered data
    x_sorted = np.sort(x)

    # Simulations under fitted normal
    rng = np.random.default_rng(seed)
    sims = np.sort(rng.normal(mu, sd, size=(B, n)), axis=1)

    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(q_theor, x_sorted, s=15)
    # Fitted line through sample mean/sd: y = mu + sd * q
    ax.plot(q_theor, mu + sd*q_theor, linestyle="--")
    ax.fill_between(q_theor, lo, hi, alpha=0.15, label="95% pointwise band")
    ax.set_title("QQ-plot vs Normal with simulated 95% band")
    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel("Ordered data")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    rng = np.random.default_rng(123)
    x = rng.lognormal(mean=0.0, sigma=0.6, size=300)  # skewed example
    qq_with_band(x, B=600, seed=7)

if __name__ == "__main__":
    main()
