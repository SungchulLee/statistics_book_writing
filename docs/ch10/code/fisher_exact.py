#!/usr/bin/env python3
# ======================================================================
# 17_fisher_exact_test.py
# ======================================================================
# Fisher's exact test for a 2×2 contingency table.
# Used when sample sizes are too small for the chi-square approximation.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (9_testsCategorical.ipynb).
# ======================================================================

import numpy as np
from scipy import stats

def main():
    # ── Example 2×2 contingency table ──
    #               Success   Failure
    # Treatment        1         5
    # Control          8         2
    observed = np.array([[1, 5],
                         [8, 2]])

    print("Observed contingency table:")
    print(observed)
    print()

    # Fisher's exact test computes the exact probability of obtaining
    # a distribution as extreme as (or more extreme than) the observed,
    # given the marginal totals.  Unlike the chi-square test it does
    # not rely on a large-sample approximation.
    odds_ratio, p_value = stats.fisher_exact(observed)

    print(f"Odds ratio : {odds_ratio:.4f}")
    print(f"p-value    : {p_value:.4f}")
    print()

    if p_value < 0.05:
        print("→ Reject H₀: the row and column variables are "
              "significantly associated (α = 0.05).")
    else:
        print("→ Fail to reject H₀: no significant association "
              "detected (α = 0.05).")


if __name__ == "__main__":
    main()
