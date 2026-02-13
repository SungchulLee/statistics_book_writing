#!/usr/bin/env python3
# ======================================================================
# 19_f_test_06_power_simulation_sketch.py
# ======================================================================
# Monte Carlo sketch to estimate power of the two-sided F-test for a given
# variance ratio and sample sizes, assuming Normal data.
# ======================================================================

import numpy as np
from scipy.stats import f

rng = np.random.default_rng(0)

def f_test_two_sided(x1, x2, alpha=0.05):
    n1, n2 = x1.size, x2.size
    df1, df2 = n1-1, n2-1
    F_obs = x1.var(ddof=1) / x2.var(ddof=1)
    p_left  = f.cdf(F_obs, df1, df2)
    p_right = f.sf(F_obs, df1, df2)
    p_two = 2 * min(p_left, p_right)
    return p_two < alpha

def estimate_power(n1=12, n2=12, sigma1=1.0, sigma2=1.5, mu1=0.0, mu2=0.0, n_sims=2000, alpha=0.05):
    hits = 0
    for _ in range(n_sims):
        x1 = rng.normal(mu1, sigma1, size=n1)
        x2 = rng.normal(mu2, sigma2, size=n2)
        if f_test_two_sided(x1, x2, alpha=alpha):
            hits += 1
    return hits / n_sims

def main():
    pow_est = estimate_power(n1=10, n2=10, sigma1=1.0, sigma2=2.0, n_sims=1000, alpha=0.05)
    print(f"Estimated two-sided F-test power (sigma1/sigma2=0.5): {pow_est:.3f}")

if __name__ == "__main__":
    main()
