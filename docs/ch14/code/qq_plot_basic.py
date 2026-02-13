#!/usr/bin/env python3
# ======================================================================
# 26_qq_01_basic_probplot_and_tests.py
# ======================================================================
# Basic QQ-plot against Normal + common normality tests:
# - Shapiro–Wilk (good power for small n)
# - D’Agostino’s K^2 (skewness/kurtosis omnibus)
# - Anderson–Darling (critical values for normal)
# Uses scipy.stats.probplot to compute theoretical quantiles and fitted line.
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def qqplot_with_fit(x):
    x = np.asarray(x, dtype=float)
    osm, osr = stats.probplot(x, dist="norm", sparams=(), fit=False)
    # osm: theoretical quantiles; osr: ordered responses
    # Fit a line y = a + b*x by least squares
    b, a = np.polyfit(osm, osr, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(osm, osr, s=15)
    xx = np.linspace(osm.min(), osm.max(), 200)
    ax.plot(xx, a + b*xx, linestyle="--")
    ax.set_title("QQ-plot vs Normal with fitted line")
    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel("Ordered data")
    plt.tight_layout()
    plt.show()
    return a, b

def run_tests(x):
    x = np.asarray(x, dtype=float)
    W, p_sh = stats.shapiro(x)               # Shapiro–Wilk
    K2, p_k2 = stats.normaltest(x)           # D’Agostino’s K^2
    ad = stats.anderson(x, dist="norm")      # Anderson–Darling
    print(f"Shapiro–Wilk: W={W:.4f}, p={p_sh:.4g}")
    print(f"D’Agostino K^2: K2={K2:.4f}, p={p_k2:.4g}")
    print("Anderson–Darling: A2={:.4f}".format(ad.statistic))
    for crit, sig in zip(ad.critical_values, ad.significance_level):
        print(f"  Critical {sig:.0f}%: {crit:.4f}  -> reject if A2 > crit")

def main():
    # Example: mildly non-normal data (mix of normal + heavier tails)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, size=150), rng.standard_t(df=3, size=50)])
    a, b = qqplot_with_fit(x)
    run_tests(x)

if __name__ == "__main__":
    main()
