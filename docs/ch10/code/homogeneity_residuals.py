#!/usr/bin/env python3
# ======================================================================
# 18_homog_04_residual_heatmap_posthoc.py
# ======================================================================
# Post-hoc diagnostics:
#   - Standardized residuals R = (O - E) / sqrt(E)
#   - Approximate two-sided z-tests per cell with Bonferroni correction
#     (illustrative; not a full-blown multiple-comparison framework)
#   - Simple heatmap using matplotlib (no seaborn)
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

def main():
    observed = np.array([
        [25, 30, 20, 25],
        [18, 22, 35, 25],
        [30, 25, 15, 30],
    ], dtype=float)

    row_tot = observed.sum(axis=1, keepdims=True)
    col_tot = observed.sum(axis=0, keepdims=True)
    tot = observed.sum()
    expected = (row_tot @ col_tot) / tot

    # Standardized residuals (Pearson)
    resid = (observed - expected) / np.sqrt(expected)

    # Per-cell z-tests (approx), two-sided
    z = resid.ravel()
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    reject, pvals_bonf, _, _ = multipletests(pvals, method="bonferroni")
    pvals_bonf = pvals_bonf.reshape(observed.shape)
    reject = reject.reshape(observed.shape)

    print("Standardized residuals:")
    print(resid, "\n")
    print("Bonferroni-adjusted per-cell p-values (approx):")
    print(pvals_bonf, "\n")

    # Heatmap-like plot with matplotlib
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(resid, aspect='auto')
    ax.set_title("Standardized residuals heatmap")
    ax.set_xlabel("Category")
    ax.set_ylabel("Population")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate significant cells after Bonferroni
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            mark = "*" if reject[i, j] else ""
            ax.text(j, i, f"{resid[i,j]:.2f}{mark}", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
