#!/usr/bin/env python3
# ======================================================================
# 19_f_test_05_normality_and_robust_alternatives.py
# ======================================================================
# Notes + quick checks:
# - F-test assumes Normality; it's very sensitive to non-normality.
# - Robust alternatives for equality of variances across groups:
#     * Levene's test (center='mean' or 'median' (Brown–Forsythe))
#     * Fligner–Killeen (nonparametric, very robust)
# ======================================================================

import numpy as np
from scipy.stats import levene, fligner, shapiro

def main():
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    # Shapiro-Wilk normality checks (small-sample illustrative)
    W1, p1 = shapiro(x1)
    W2, p2 = shapiro(x2)
    print(f"Shapiro-Wilk x1: W={W1:.4f}, p={p1:.4f}")
    print(f"Shapiro-Wilk x2: W={W2:.4f}, p={p2:.4f}")

    # Levene (mean-centered)
    Wm, pm = levene(x1, x2, center='mean')
    print(f"Levene (mean-centered): W={Wm:.4f}, p={pm:.6f}")

    # Brown–Forsythe (median-centered)
    Wmed, pmed = levene(x1, x2, center='median')
    print(f"Brown–Forsythe (median-centered): W={Wmed:.4f}, p={pmed:.6f}")

    # Fligner–Killeen
    X2, pF = fligner(x1, x2)
    print(f"Fligner–Killeen: X2={X2:.4f}, p={pF:.6f}")

if __name__ == "__main__":
    main()
