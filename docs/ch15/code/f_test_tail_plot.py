#!/usr/bin/env python3
# ======================================================================
# 19_f_test_02_plot_tail_regions.py
# ======================================================================
# Visualize the F distribution with shaded tail(s) for an observed statistic.
# - Left-tail for H1: sigma1^2 < sigma2^2
# - Right-tail for H1: sigma1^2 > sigma2^2
# - Two-sided: shade both extreme tails (symmetric via 1/F and swapped dfs)
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

def main():
    sample1 = [12, 15, 14, 10, 13, 14, 12, 11]
    sample2 = [22, 25, 20, 18, 24, 23, 19, 21]

    x1 = np.asarray(sample1, dtype=float)
    x2 = np.asarray(sample2, dtype=float)
    df1, df2 = x1.size - 1, x2.size - 1

    F_obs = x1.var(ddof=1) / x2.var(ddof=1)

    xs = np.linspace(0.01, max(6, F_obs + 2), 400)
    pdf = f(df1, df2).pdf(xs)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(xs, pdf, linewidth=2)

    # Left-tail (H1: sigma1^2 < sigma2^2)
    mask_left = xs <= F_obs
    ax.fill_between(xs[mask_left], pdf[mask_left], 0, alpha=0.15)

    # Right-tail (H1: sigma1^2 > sigma2^2)
    mask_right = xs >= F_obs
    ax.fill_between(xs[mask_right], pdf[mask_right], 0, alpha=0.15)

    ax.axvline(F_obs, linestyle="--")
    ax.set_title(f"F(df1={df1}, df2={df2}) with observed F = {F_obs:.3f}")
    ax.set_xlabel("F"); ax.set_ylabel("PDF")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
