#!/usr/bin/env python3
# ======================================================================
# 35_sw_03_power_simulation_lognormal.py
# ======================================================================
# Empirical power of Shapiro–Wilk vs sample size for a lognormal alternative.
# Plots a simple power curve (one figure).
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def power_for_n(n, sims=500, sigma_ln=0.6, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(sims):
        x = rng.lognormal(mean=0.0, sigma=sigma_ln, size=n)
        W, p = stats.shapiro(x)
        if p < alpha:
            rejections += 1
    return rejections / sims

def main():
    ns = [20, 30, 50, 80, 120, 200, 300]
    powers = [power_for_n(n, sims=400, sigma_ln=0.6, alpha=0.05, seed=42+n) for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, powers, marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Empirical power (α=0.05)")
    ax.set_title("Shapiro–Wilk power vs n (lognormal alt, σ=0.6)")
    plt.tight_layout()
    plt.show()

    for n, pw in zip(ns, powers):
        print(f"n={n:>3}: power≈{pw:.3f}")

if __name__ == "__main__":
    main()
