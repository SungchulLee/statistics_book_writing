#!/usr/bin/env python3
# ======================================================================
# 33_lillie_01_normal_basic_bootstrap.py
# ======================================================================
# Lilliefors test for Normality via parametric bootstrap:
#  - Fit mu, sd to the data.
#  - Compute KS distance D between ECDF and N(mu, sd).
#  - Bootstrap B times: simulate N(mu, sd), refit mu*, sd*, compute D*.
#  - p ≈ P(D* >= D_obs).
# Notes: This calibrates the KS test when parameters are estimated (classic
# Lilliefors setting). No external packages beyond NumPy/SciPy.
# ======================================================================

import numpy as np
from scipy import stats

def ks_stat_fitted_normal(x):
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    D, _ = stats.kstest(x, 'norm', args=(mu, sd))
    return float(D), float(mu), float(sd)

def lilliefors_normal_bootstrap(x, B=2000, seed=0):
    x = np.asarray(x, dtype=float)
    n = x.size
    D_obs, mu, sd = ks_stat_fitted_normal(x)

    rng = np.random.default_rng(seed)
    D_star = []
    for _ in range(B):
        xb = rng.normal(mu, sd, size=n)
        Db, _, _ = ks_stat_fitted_normal(xb)
        D_star.append(Db)
    D_star = np.array(D_star)
    p_boot = float(np.mean(D_star >= D_obs))
    return D_obs, p_boot, mu, sd

def main():
    rng = np.random.default_rng(1)
    # Example: skewed alternative to showcase rejection
    x = rng.lognormal(0.0, 0.6, size=300)

    D, p, mu, sd = lilliefors_normal_bootstrap(x, B=1500, seed=7)
    print(f"Fitted Normal: mu={mu:.4f}, sd={sd:.4f}")
    print(f"Lilliefors (Normal) -> KS D={D:.4f}, p≈{p:.4f}")
    if p < 0.05:
        print("=> Reject Normality at alpha=0.05.")
    else:
        print("=> Fail to reject Normality at alpha=0.05.")

if __name__ == "__main__":
    main()
