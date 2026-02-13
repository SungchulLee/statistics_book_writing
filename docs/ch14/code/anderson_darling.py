#!/usr/bin/env python3
# ======================================================================
# 34_ad_01_basic_anderson_normal.py
# ======================================================================
# One-sample Anderson–Darling test for Normality using SciPy.
# SciPy reports the A^2 statistic, critical values, and significance levels.
# There is no exact p-value; interpret by comparing A^2 to the critical values.
# ======================================================================

import numpy as np
from scipy import stats

def main():
    rng = np.random.default_rng(0)
    # Example: mixture causing deviation from Normal
    x = np.concatenate([rng.normal(0, 1, size=230), rng.lognormal(0, 0.6, size=70)])

    res = stats.anderson(x, dist="norm")
    print(f"Sample size n={x.size}")
    print(f"Anderson–Darling A^2 statistic = {res.statistic:.4f}")
    print("Critical values vs significance levels:")
    for cv, sl in zip(res.critical_values, res.significance_level):
        print(f"  {sl:.1f}% -> {cv:.4f} (reject if A^2 > {cv:.4f})")
    # Quick interpretation at 5%
    reject_5 = res.statistic > res.critical_values[list(res.significance_level).index(5.0)]
    print("Decision at 5%:", "Reject normality" if reject_5 else "Fail to reject normality")

if __name__ == "__main__":
    main()
