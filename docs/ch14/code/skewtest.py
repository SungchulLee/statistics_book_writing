#!/usr/bin/env python3
# ======================================================================
# 28_skew_01_basic_skewtest.py
# ======================================================================
# One-sample skewness test for normality using SciPy's D'Agostino skewness test.
# Prints the sample skewness (Fisher's g1), the skewtest Z statistic, and p-value.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example: mildly right-skewed data
    x = rng.lognormal(mean=0.0, sigma=0.6, size=300)

    # Fisher-Pearson sample skewness (bias-corrected)
    g1 = stats.skew(x, bias=False)

    # D'Agostino's skewness test
    z, p = stats.skewtest(x)

    print(f"Sample size n={x.size}")
    print(f"Sample skewness (Fisher's g1) = {g1:.4f}")
    print(f"D'Agostino skewness test: Z={z:.4f}, p-value={p:.4g}")
    if p < 0.05:
        print('=> Evidence of non-zero skewness (departing from normality).')
    else:
        print('=> No strong evidence of non-zero skewness.')

if __name__ == "__main__":
    main()
