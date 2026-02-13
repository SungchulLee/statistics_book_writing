#!/usr/bin/env python3
# ======================================================================
# 17_independence_04_template_function.py
# ======================================================================
# Template utility: pass any 2D observed table, get chi2 test results
# and (optionally) the expected table.
# ======================================================================

import numpy as np
from scipy import stats

def chi2_independence(observed: np.ndarray, correction: bool=False):
    """Run chi-square test of independence.
    Parameters
    ----------
    observed : np.ndarray
        2D contingency table of observed counts.
    correction : bool
        Yates' continuity correction (only applied to 2x2). Default False.
    Returns
    -------
    chi2, p, df, expected : tuple
        Test statistic, p-value, degrees of freedom, and expected counts.
    """
    return stats.chi2_contingency(observed, correction=correction)

def demo():
    observed = np.array([[30, 20, 10],
                         [12, 25, 18]], dtype=float)
    chi2, p, df, exp = chi2_independence(observed, correction=False)
    print(f"chi2 = {chi2:.3f}, p = {p:.4f}, df = {df}")
    print("expected:\n", exp)

if __name__ == "__main__":
    demo()
