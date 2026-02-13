#!/usr/bin/env python3
# ===============================================
# 13_testing_01_mean_one_sample_simple.py
# ===============================================
import math
from scipy.stats import t as tdist, norm

def test_mean_one_sample(xbar, n, mu0=0.0, sd=None, known_sigma=None, alt="two-sided", alpha=0.05):
    """
    If known_sigma is given -> z-test, else t-test using sample sd.
    Returns (stat, pvalue, reject_bool, label).
    """
    if known_sigma is not None:
        se = known_sigma / math.sqrt(n)
        z = (xbar - mu0) / se
        if alt == "two-sided":
            p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
        elif alt == "less":
            p = norm.cdf(z)
        else:
            p = 1 - norm.cdf(z)
        return z, p, (p < alpha), "z-test"
    if sd is None:
        raise ValueError("Provide sd for t-test or known_sigma for z-test.")
    se = sd / math.sqrt(n)
    df = n - 1
    t = (xbar - mu0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha), f"t-test (df={df})"

if __name__ == "__main__":
    stat, p, reject, label = test_mean_one_sample(xbar=3.2, n=25, mu0=3.0, sd=1.1, alt="greater")
    print(label, "stat:", stat, "p:", p, "reject:", reject)
