#!/usr/bin/env python3
# ===============================================
# 13_testing_02_proportion_one_sample_simple.py
# ===============================================
from scipy.stats import norm, binomtest
import math

def test_prop_one_sample(k, n, p0=0.5, method="wald", alt="two-sided", alpha=0.05):
    """
    method='wald' (normal approx) or 'exact' (binomial).
    Returns (stat_or_None, pvalue, reject_bool, label).
    """
    phat = k / n
    if method == "exact":
        p = binomtest(k, n, p0, alternative=alt).pvalue
        return None, p, (p < alpha), "exact binomial"
    se0 = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se0
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), "wald z-test"

if __name__ == "__main__":
    stat, p, reject, label = test_prop_one_sample(k=12, n=50, p0=0.2, method="wald", alt="two-sided")
    print(label, "stat:", stat, "p:", p, "reject:", reject)
