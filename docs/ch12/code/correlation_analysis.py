#!/usr/bin/env python3
# ======================================================================
# 20_correlation_01_pearson_spearman_kendall.py
# ======================================================================
# Compute Pearson, Spearman, and Kendall correlation coefficients for
# bivariate data, print results, and show a scatter plot with the
# least-squares regression line.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (11_correlationRegression.ipynb).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def main():
    # ── Generate bivariate data with a known linear relationship ──
    n = 120
    x = np.random.uniform(10, 60, n)
    noise = np.random.normal(0, 8, n)
    y = 0.8 * x + 5 + noise          # positive linear trend + noise

    # ── Correlation coefficients ──
    r_pearson, p_pearson   = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    r_kendall, p_kendall   = stats.kendalltau(x, y)

    print("Correlation Coefficients")
    print("=" * 50)
    print(f"  Pearson  r = {r_pearson: .4f}  (p = {p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman: .4f}  (p = {p_spearman:.2e})")
    print(f"  Kendall  τ = {r_kendall: .4f}  (p = {p_kendall:.2e})")
    print()

    # Verify: Spearman ρ equals Pearson r computed on ranks
    r_rank = stats.pearsonr(stats.rankdata(x), stats.rankdata(y))[0]
    print(f"  Pearson r on ranks = {r_rank:.4f}  "
          f"(should match Spearman ρ = {r_spearman:.4f})")

    # ── Scatter plot with regression line ──
    slope, intercept, _, _, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors='k', linewidths=0.3)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
            label=f'OLS: y = {slope:.2f}x + {intercept:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Pearson r = {r_pearson:.3f},  Spearman ρ = {r_spearman:.3f},  '
                 f'Kendall τ = {r_kendall:.3f}')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
