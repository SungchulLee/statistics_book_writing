#!/usr/bin/env python3
# ======================================================================
# 32_ks_01_basic_one_sample_kstest_fixed_params.py
# ======================================================================
# One-sample Kolmogorov–Smirnov test using SciPy against a *fully specified*
# Normal(mu, sigma). NOTE: If you estimate mu/sigma from the data and plug them
# in directly, the vanilla KS p-value is invalid (Lilliefors issue). For that,
# use the bootstrap-based script (32_ks_02_lilliefors_like_bootstrap.py).
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=250)

    # Fully specified H0: Normal(0,1)
    D, p = stats.kstest(x, 'norm', args=(0.0, 1.0))

    print(f"n={x.size}")
    print(f"KS one-sample vs N(0,1): D={D:.4f}, p={p:.4g}")
    if p < 0.05:
        print("=> Reject H0: data may not follow N(0,1).")
    else:
        print("=> Fail to reject H0 at alpha=0.05.")

if __name__ == "__main__":
    main()
