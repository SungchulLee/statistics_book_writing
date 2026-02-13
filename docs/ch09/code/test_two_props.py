#!/usr/bin/env python3
# ===============================================
# 13_testing_07_diff_two_proportions_simple.py
# ===============================================
import math
from scipy.stats import norm

def test_diff_two_props(k1, n1, k2, n2, delta0=0.0, method="pooled", alt="two-sided", alpha=0.05):
    """
    H0: p1 - p2 = delta0.
    When delta0=0 and method='pooled', uses pooled SE; otherwise uses Wald SE.
    Returns (z, p, reject, label).
    """
    p1, p2 = k1 / n1, k2 / n2
    d_hat = p1 - p2
    if delta0 == 0.0 and method == "pooled":
        p_pool = (k1 + k2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        label = "pooled z-test"
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        label = "wald z-test"

    z = (d_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), label

if __name__ == "__main__":
    z, p, reject, label = test_diff_two_props(k1=30, n1=80, k2=18, n2=60, delta0=0.0, method="pooled")
    print(label, "z:", z, "p:", p, "reject:", reject)
