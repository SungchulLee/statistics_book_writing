#!/usr/bin/env python3
# ======================================================================
# 18_homog_01_scipy_basic.py
# ======================================================================
# Chi-square Test of Homogeneity using scipy.stats.chi2_contingency.
# Interpreted as: Are the category proportions the same across populations?
# (Statistically identical to the chi-square test of independence.)
# ======================================================================

import numpy as np
from scipy import stats

def main():
    # Example: 3 populations (rows), 4 categories (cols)
    observed = np.array([
        [25, 30, 20, 25],   # Pop 1
        [18, 22, 35, 25],   # Pop 2
        [30, 25, 15, 30],   # Pop 3
    ], dtype=float)

    chi2, p, df, expected = stats.chi2_contingency(observed, correction=False)
    print("=== Chi-square Test of Homogeneity (scipy) ===")
    print(f"chi2 = {chi2:.4f}, df = {df}, p = {p:.6f}
")
    print("Expected counts under H0 (same proportions):")
    print(expected)

if __name__ == "__main__":
    main()
