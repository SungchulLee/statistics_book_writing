#!/usr/bin/env python3
# ======================================================================
# 24_bootvar_01_twosample_nonparametric_percentile_CI_and_pvalue.py
# ======================================================================
# Nonparametric bootstrap for the variance ratio theta = s1^2 / s2^2.
# - Percentile CI on log(theta) (more symmetric)
# - Approx. two-sided bootstrap p-value via resampling distribution
#   (treats bootstrap as sampling dist. around theta_hat; not a strict null test)
# ======================================================================

import numpy as np

rng = np.random.default_rng(0)

def variance_ratio(x1, x2):
    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    return s1 / s2

def bootstrap_varratio(x1, x2, B=2000, seed=None):
    rng_local = np.random.default_rng(seed)
    n1, n2 = len(x1), len(x2)
    stat_obs = variance_ratio(x1, x2)
    # Use log for CI symmetry
    log_obs = np.log(stat_obs)

    boots = []
    for _ in range(B):
        b1 = rng_local.choice(x1, size=n1, replace=True)
        b2 = rng_local.choice(x2, size=n2, replace=True)
        boots.append(np.log(variance_ratio(b1, b2)))
    boots = np.array(boots)

    # Percentile CI on log scale, then exponentiate back
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci = (float(np.exp(lo)), float(np.exp(hi)))

    # Approx two-sided bootstrap p-value (symmetric around log_obs)
    p_two = 2 * min(np.mean(boots <= log_obs), np.mean(boots >= log_obs))
    p_two = float(min(p_two, 1.0))

    return float(stat_obs), ci, p_two, boots

def demo():
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    theta_hat, ci, p, boots = bootstrap_varratio(x1, x2, B=1000, seed=42)
    print(f"theta_hat = s1^2/s2^2 = {theta_hat:.4f}")
    print(f"Percentile 95% CI (log-transformed): ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"Approx bootstrap two-sided p-value: {p:.4f}")

if __name__ == "__main__":
    demo()
