#!/usr/bin/env python3
# ======================================================================
# 17_independence_01_manual_expected_with_plot.py
# ======================================================================
# Manual computation of expected counts for a chi-square test of independence,
# then compute statistic/p-value and plot the chi-square pdf with shaded tail.
# (No argparse; quick to run.)
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compute_expected(observed_counts: np.ndarray) -> np.ndarray:
    row_totals = observed_counts.sum(axis=1, keepdims=True)
    col_totals = observed_counts.sum(axis=0, keepdims=True)
    total = observed_counts.sum()
    return (row_totals @ col_totals) / total

def main():
    observed_counts = np.array([[934, 1070],
                                [113,   92],
                                [ 20,    8]], dtype=float)

    expected_counts = compute_expected(observed_counts)
    df = (observed_counts.shape[0] - 1) * (observed_counts.shape[1] - 1)

    chi2 = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    p_value = stats.chi2(df).sf(chi2)

    print(f"chi_squared_statistic = {chi2:.02f}")
    print(f"p_value = {p_value:.02%}")

    # Plot chi-square distribution and highlight observed statistic
    fig, ax = plt.subplots(figsize=(12, 4))

    x_left = np.linspace(0, chi2, 200)
    y_left = stats.chi2(df).pdf(x_left)
    ax.plot(x_left, y_left, linewidth=3)
    x_fill_left = np.concatenate([[0], x_left, [chi2], [0]])
    y_fill_left = np.concatenate([[0], y_left, [0], [0]])
    ax.fill(x_fill_left, y_fill_left, alpha=0.1)

    x_right = np.linspace(chi2, max(20, chi2 + 5), 200)
    y_right = stats.chi2(df).pdf(x_right)
    ax.plot(x_right, y_right, linewidth=3)
    x_fill_right = np.concatenate([[chi2], x_right, [max(20, chi2 + 5)], [chi2]])
    y_fill_right = np.concatenate([[0], y_right, [0], [0]])
    ax.fill(x_fill_right, y_fill_right, alpha=0.1)

    ax.annotate(f"p_value = {p_value:.02%}", xy=(chi2*0.8, y_left.max()*0.15),
                xytext=(chi2*0.9 + 5, y_left.max()*0.6),
                fontsize=12, arrowprops=dict(width=0.2, headwidth=8))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
