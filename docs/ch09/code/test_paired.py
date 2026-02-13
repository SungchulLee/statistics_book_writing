#!/usr/bin/env python3
# ===============================================
# 13_testing_05_paired_mean_simple.py
# ===============================================
import math
from scipy.stats import t as tdist

def test_paired_mean(n, dbar, sd_d, mu_d0=0.0, alt="two-sided", alpha=0.05):
    """
    Paired t: differences D = X - Y; supply n, mean(D), sd(D).
    Returns (t, p, reject).
    """
    df = n - 1
    se = sd_d / math.sqrt(n)
    t = (dbar - mu_d0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha)

if __name__ == "__main__":
    t, p, reject = test_paired_mean(n=12, dbar=0.4, sd_d=1.1, mu_d0=0.0, alt="less")
    print("t:", t, "p:", p, "reject:", reject)
