#!/usr/bin/env python3
# ======================================================================
# 17_mcnemar_test.py
# ======================================================================
# McNemar's test for paired nominal (binary) data.
# Commonly used in before/after studies on the same subjects.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (9_testsCategorical.ipynb).
# ======================================================================

import numpy as np
from scipy import stats

def mcnemar_test(table):
    """
    Perform McNemar's test on a 2×2 table of paired counts.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Contingency table where off-diagonal cells (b, c) represent
        discordant pairs:
            [[a, b],
             [c, d]]

    Returns
    -------
    statistic : float   McNemar chi-square statistic (with continuity correction)
    p_value   : float   Two-sided p-value from chi-square(1)
    """
    table = np.asarray(table)
    b = table[0, 1]
    c = table[1, 0]
    # Continuity-corrected McNemar statistic
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value


def main():
    # ── Example: disease status before / after treatment ──
    #                  After+   After-
    # Before+           101      121
    # Before-            59       33
    table = np.array([[101, 121],
                      [ 59,  33]])

    print("Paired contingency table:")
    print(table)
    print()

    chi2, p = mcnemar_test(table)

    print(f"McNemar χ² = {chi2:.4f}")
    print(f"p-value    = {p:.4e}")
    print()

    if p < 0.05:
        print("→ Reject H₀: there was a significant change in the "
              "outcome after treatment (α = 0.05).")
    else:
        print("→ Fail to reject H₀: no significant change detected "
              "(α = 0.05).")


if __name__ == "__main__":
    main()
