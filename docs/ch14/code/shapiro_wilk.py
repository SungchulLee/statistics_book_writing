#!/usr/bin/env python3
# ======================================================================
# 35_sw_01_basic_shapiro_wilk.py
# ======================================================================
# Shapiro–Wilk test for normality using SciPy.
# Prints W statistic and p-value; also reports sample skewness and excess kurtosis.
# Note: SciPy warns that p-values may not be accurate for n > 5000.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example: mixture to depart from normality (skew + heavy tails)
    x = np.concatenate([rng.normal(0, 1, size=240), rng.lognormal(0, 0.6, size=60)])

    W, p = stats.shapiro(x)
    g1 = stats.skew(x, bias=False)
    g2 = stats.kurtosis(x, fisher=True, bias=False)

    print(f"Sample size n={x.size}")
    print(f"Shapiro–Wilk: W={W:.4f}, p-value={p:.4g}")
    print(f"Skewness g1={g1:.4f}, Excess kurtosis g2={g2:.4f}")
    if p < 0.05:
        print("=> Reject normality at alpha=0.05.")
    else:
        print("=> Fail to reject normality at alpha=0.05.")

if __name__ == "__main__":
    main()
