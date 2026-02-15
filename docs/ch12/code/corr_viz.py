#!/usr/bin/env python3
# ======================================================================
# 20_correlation_02_heatmap_and_pairplot.py
# ======================================================================
# Visualise a correlation matrix as a heatmap and show a scatter-matrix
# (pair plot) for a multivariate dataset using Matplotlib only
# (no Seaborn dependency).
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (11_correlationRegression.ipynb, 4_dataDisplay.ipynb).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(7)


def main():
    # ── Simulate 4 correlated variables ──
    n = 200
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x1 = z1
    x2 = 0.7 * z1 + 0.3 * z2                     # corr ≈ 0.7 with x1
    x3 = -0.5 * z1 + np.random.randn(n) * 0.8     # weak negative with x1
    x4 = np.random.randn(n)                        # independent

    data = np.column_stack([x1, x2, x3, x4])
    labels = ['X1', 'X2', 'X3', 'X4']
    k = data.shape[1]

    # ── Correlation matrix ──
    corr_matrix = np.corrcoef(data, rowvar=False)

    # ── 1. Heatmap ──
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # ── 2. Scatter matrix (pair plot) ──
    fig, axes = plt.subplots(k, k, figsize=(10, 10))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=20, edgecolor='k', alpha=0.7)
            else:
                ax.scatter(data[:, j], data[:, i], s=8, alpha=0.5)
            if j == 0:
                ax.set_ylabel(labels[i])
            if i == k - 1:
                ax.set_xlabel(labels[j])
            if j != 0:
                ax.set_yticklabels([])
            if i != k - 1:
                ax.set_xticklabels([])
    fig.suptitle('Scatter Matrix (Pair Plot)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
