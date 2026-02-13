#!/usr/bin/env python3
# ======================================================================
# 30_k2_01_basic_normaltest.py
# ======================================================================
# D'Agostino's K^2 test for normality using SciPy (skewness + kurtosis omnibus).
# Prints K^2 statistic and p-value.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example mixture to depart from normality (skew + heavy tails)
    x = np.concatenate([rng.normal(0, 1, size=240), rng.lognormal(0, 0.6, size=60)])

    K2, p = stats.normaltest(x)  # D'Agostino's K^2
    print(f"Sample size n={x.size}")
    print(f"D'Agostino's K^2 statistic = {K2:.4f}")
    print(f"p-value = {p:.4g}")
    if p < 0.05:
        print('=> Reject normality at alpha=0.05.')
    else:
        print('=> Fail to reject normality at alpha=0.05.')

if __name__ == "__main__":
    main()
