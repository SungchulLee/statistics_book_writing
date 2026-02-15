#!/usr/bin/env python3
# ======================================================================
# 04_weibull_01_pdf_and_hazard.py
# ======================================================================
# Weibull distribution:
#   1. PDF for various shape (k) and scale (lambda) parameters.
#   2. Hazard function h(t) = (k/lambda)(t/lambda)^(k-1).
#   3. Survival function S(t) = exp(-(t/lambda)^k).
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 2 — Data and Sampling Distributions).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def weibull_pdf(x, k, lam):
    """Weibull PDF: f(x) = (k/lam)(x/lam)^(k-1) exp(-(x/lam)^k)."""
    return (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)


def weibull_survival(x, k, lam):
    """Survival function: S(x) = exp(-(x/lam)^k)."""
    return np.exp(-(x / lam) ** k)


def weibull_hazard(x, k, lam):
    """Hazard function: h(x) = (k/lam)(x/lam)^(k-1)."""
    return (k / lam) * (x / lam) ** (k - 1)


def main():
    print("Weibull Distribution")
    print("=" * 55)

    x = np.linspace(0.01, 3.0, 500)
    lam = 1.0  # scale parameter

    params = [
        (0.5, "k=0.5 (decreasing hazard)"),
        (1.0, "k=1.0 (exponential)"),
        (1.5, "k=1.5 (increasing hazard)"),
        (3.0, "k=3.0 (near-normal shape)"),
    ]

    print(f"\n  Scale (lambda) = {lam}")
    for k, desc in params:
        rv = stats.weibull_min(k, scale=lam)
        print(f"  k = {k:.1f}: mean = {rv.mean():.4f}, "
              f"var = {rv.var():.4f}  — {desc}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for k, desc in params:
        label = f'k = {k}'
        axes[0].plot(x, weibull_pdf(x, k, lam), linewidth=2, label=label)
        axes[1].plot(x, weibull_survival(x, k, lam), linewidth=2, label=label)
        axes[2].plot(x, weibull_hazard(x, k, lam), linewidth=2, label=label)

    axes[0].set_title('Weibull PDF')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')

    axes[1].set_title('Survival Function S(x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('S(x)')

    axes[2].set_title('Hazard Function h(x)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('h(x)')
    axes[2].set_ylim(0, 5)

    for ax in axes:
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
