#!/usr/bin/env python3
# ======================================================================
# 31_jb_01_basic_jarque_bera.py
# ======================================================================
# Jarque–Bera test for normality using SciPy.
# Reports the JB statistic and p-value along with sample skewness/kurtosis.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example: mixture to deviate from normality (skew + heavy tails)
    x = np.concatenate([rng.normal(0, 1, size=240), rng.lognormal(0, 0.6, size=60)])

    jb_stat, p = stats.jarque_bera(x)
    g1 = stats.skew(x, bias=False)
    g2 = stats.kurtosis(x, fisher=True, bias=False)

    print(f"Sample size n={x.size}")
    print(f"Skewness g1 = {g1:.4f}")
    print(f"Excess kurtosis g2 = {g2:.4f}")
    print(f"Jarque–Bera: JB={jb_stat:.4f}, p-value={p:.4g}")
    if p < 0.05:
        print("=> Reject normality at alpha=0.05.")
    else:
        print("=> Fail to reject normality at alpha=0.05.")

if __name__ == "__main__":
    main()
