#!/usr/bin/env python3
# ======================================================================
# 22_bf_01_basic_scipy.py
# ======================================================================
# Brown–Forsythe test (Levene with median-centering) using SciPy.
# Robust to non-normality.
# ======================================================================

import numpy as np
from scipy.stats import levene

def main():
    g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], dtype=float)

    W, p = levene(g1, g2, g3, center='median')  # Brown–Forsythe
    print(f"Brown–Forsythe (median-centered Levene): W={W:.6f}, p={p:.6f}")

if __name__ == "__main__":
    main()
