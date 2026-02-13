#!/usr/bin/env python3
# ===============================================
# 13_testing_03_variance_one_sample_simple.py
# ===============================================
from scipy.stats import chi2

def test_variance_one_sample(n, s2, sigma0, alt="two-sided", alpha=0.05):
    """
    H0: sigma^2 = sigma0^2 (Normal). Provide s2 = sample variance (ddof=1).
    Returns (chi2_stat, pvalue, reject_bool).
    """
    df = n - 1
    chi2_stat = df * s2 / (sigma0**2)
    if alt == "two-sided":
        p = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
    elif alt == "less":
        p = chi2.cdf(chi2_stat, df)
    else:
        p = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p, (p < alpha)

if __name__ == "__main__":
    stat, p, reject = test_variance_one_sample(n=12, s2=2.1**2, sigma0=2.0, alt="greater")
    print("chi2:", stat, "p:", p, "reject:", reject)
