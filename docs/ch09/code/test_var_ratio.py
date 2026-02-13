#!/usr/bin/env python3
# ===============================================
# 13_testing_04_ratio_two_variances_simple.py
# ===============================================
from scipy.stats import f

def test_ratio_two_variances(n1, s1, n2, s2, theta0=1.0, alt="two-sided", alpha=0.05):
    """
    H0: sigma1^2 / sigma2^2 = theta0 (Normal).
    Provide s1,s2 = sample std (ddof=1). Returns (F, p, reject).
    """
    df1, df2 = n1 - 1, n2 - 1
    F = (s1**2 / s2**2) / theta0
    if alt == "two-sided":
        p = 2 * min(f.cdf(F, df1, df2), 1 - f.cdf(F, df1, df2))
    elif alt == "less":
        p = f.cdf(F, df1, df2)
    else:
        p = 1 - f.cdf(F, df1, df2)
    return F, p, (p < alpha)

if __name__ == "__main__":
    F, p, reject = test_ratio_two_variances(n1=15, s1=1.3, n2=12, s2=0.9, theta0=1.0, alt="greater")
    print("F:", F, "p:", p, "reject:", reject)
