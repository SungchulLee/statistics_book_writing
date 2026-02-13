#!/usr/bin/env python3
# ======================================================================
# 25_bayes_01_posterior_sigma2_and_ratio.py
# ======================================================================
# Conjugate Normal–Inverse-Gamma for each group:
#   mu_i | sigma_i^2 ~ N(m0, sigma_i^2/k0)
#   sigma_i^2 ~ InvGamma(a0, b0)
# Outputs posterior draws for sigma_i^2 and the variance ratio rho = sigma1^2/sigma2^2,
# plus summary stats and credible intervals. Assumes groups are independent.
# ======================================================================

import numpy as np
from scipy.stats import invgamma

def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
    x = np.asarray(x, dtype=float)
    n = x.size
    xbar = x.mean()
    S = np.sum((x - xbar)**2)
    k_n = k0 + n
    m_n = (k0*m0 + n*xbar) / k_n
    a_n = a0 + n/2.0
    b_n = b0 + 0.5*(S + (k0*n/k_n)*(xbar - m0)**2)
    return m_n, k_n, a_n, b_n

def draw_posterior_sigma2(x, n_draws=10000, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    m_n, k_n, a_n, b_n = posterior_params(x, m0=m0, k0=k0, a0=a0, b0=b0)
    # scipy.stats.invgamma uses shape 'a' and 'scale' parameterization
    sig2 = invgamma(a=a_n, scale=b_n).rvs(size=n_draws, random_state=rng)
    return sig2

def summarize(arr, q=(2.5, 50, 97.5)):
    return {f"q{p}": float(np.percentile(arr, p)) for p in q} | {"mean": float(np.mean(arr))}

def main():
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    rng = np.random.default_rng(0)

    s1 = draw_posterior_sigma2(x1, n_draws=20000, rng=rng)
    s2 = draw_posterior_sigma2(x2, n_draws=20000, rng=rng)
    ratio = s1 / s2
    sd_ratio = np.sqrt(ratio)

    print("Posterior sigma1^2:", summarize(s1))
    print("Posterior sigma2^2:", summarize(s2))
    print("Posterior ratio sigma1^2/sigma2^2:", summarize(ratio))
    print("P(sigma1^2 > sigma2^2) ≈", float(np.mean(ratio > 1.0)))
    print("Posterior ratio of std devs (sigma1/sigma2):", summarize(sd_ratio))

if __name__ == "__main__":
    main()
