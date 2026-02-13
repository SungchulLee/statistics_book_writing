#!/usr/bin/env python3
# ======================================================================
# 29_kurt_01_basic_kurtosistest.py
# ======================================================================
# One-sample kurtosis test for normality using SciPy's D'Agostino kurtosis test.
# Prints sample excess kurtosis (Fisher), the kurtosistest Z statistic, and p-value.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example: heavier-tailed mixture to show kurtosis effect
    x = np.concatenate([rng.normal(0, 1, size=220), rng.standard_t(df=4, size=80)])

    # Fisher's excess kurtosis (bias-corrected)
    g2 = stats.kurtosis(x, fisher=True, bias=False)

    # D'Agostino's kurtosis test
    z, p = stats.kurtosistest(x)

    print(f"Sample size n={x.size}")
    print(f"Sample excess kurtosis (Fisher) g2 = {g2:.4f}")
    print(f"D'Agostino kurtosis test: Z={z:.4f}, p-value={p:.4g}")
    if p < 0.05:
        print('=> Evidence of non-normal kurtosis (departing from normality).')
    else:
        print('=> No strong evidence of non-normal kurtosis.')

if __name__ == "__main__":
    main()
