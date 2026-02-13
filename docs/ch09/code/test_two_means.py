#!/usr/bin/env python3
# ===============================================
# 13_testing_06_diff_two_means_simple.py
# ===============================================
import math
from scipy.stats import t as tdist

def test_diff_two_means(n1, m1, s1, n2, m2, s2, method="welch", delta0=0.0, alt="two-sided", alpha=0.05):
    """
    H0: mu1 - mu2 = delta0. method='welch' (default) or 'pooled'.
    Returns (t, df, p, reject).
    """
    diff_hat = m1 - m2
    if method == "welch":
        se = math.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        df = num / den
    else:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))

    t = (diff_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, df, p, (p < alpha)

if __name__ == "__main__":
    t, df, p, reject = test_diff_two_means(n1=12, m1=0.0, s1=1.0, n2=10, m2=0.5, s2=1.5, method="welch", alt="greater")
    print("t:", t, "df:", df, "p:", p, "reject:", reject)
