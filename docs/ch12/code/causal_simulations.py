#!/usr/bin/env python3
# ======================================================================
# 20_causal_01_confounding_and_spurious_correlation.py
# ======================================================================
# Simulate three classic pitfalls that make correlation ≠ causation:
# 1. Confounding variable (omitted variable bias).
# 2. Spurious correlation driven by a common cause.
# 3. Simpson's paradox — aggregate vs subgroup trends reverse.
#
# Source:  Concept adapted from *Introduction to Statistics with Python*
#          and the course's Chapter 12 material.
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(21)


def demo_confounding():
    """Show how a confounding variable Z can create a misleading X-Y correlation."""
    n = 300
    Z = np.random.randn(n)              # confounder
    X = 0.6 * Z + np.random.randn(n) * 0.5
    Y = 0.8 * Z + np.random.randn(n) * 0.5   # Y depends on Z, not X

    r_xy, _ = stats.pearsonr(X, Y)
    # Partial correlation controlling for Z
    r_xz, _ = stats.pearsonr(X, Z)
    r_yz, _ = stats.pearsonr(Y, Z)
    r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    print("1. Confounding variable")
    print(f"   Pearson r(X, Y)          = {r_xy:.3f}")
    print(f"   Partial r(X, Y | Z)      = {r_partial:.3f}")
    print(f"   -> After controlling for Z the association nearly vanishes.\n")


def demo_simpsons_paradox():
    """Demonstrate Simpson's paradox: aggregate trend reverses within subgroups."""
    rng = np.random.default_rng(42)

    # Two subgroups with different baselines
    n_a, n_b = 100, 100
    x_a = rng.uniform(10, 30, n_a)
    y_a = -0.4 * x_a + 30 + rng.normal(0, 2, n_a)

    x_b = rng.uniform(25, 50, n_b)
    y_b = -0.4 * x_b + 45 + rng.normal(0, 2, n_b)

    x_all = np.concatenate([x_a, x_b])
    y_all = np.concatenate([y_a, y_b])

    r_all, _  = stats.pearsonr(x_all, y_all)
    r_a, _    = stats.pearsonr(x_a, y_a)
    r_b, _    = stats.pearsonr(x_b, y_b)

    print("2. Simpson's paradox")
    print(f"   Aggregate  r = {r_all:+.3f}  (positive!)")
    print(f"   Subgroup A r = {r_a:+.3f}  (negative)")
    print(f"   Subgroup B r = {r_b:+.3f}  (negative)")
    print(f"   -> Within every subgroup the relationship is negative,\n"
          f"     but aggregating flips the sign.\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_a, y_a, label='Group A', alpha=0.6)
    ax.scatter(x_b, y_b, label='Group B', alpha=0.6, marker='s')
    # Aggregate regression line
    slope, intercept = np.polyfit(x_all, y_all, 1)
    xs = np.linspace(x_all.min(), x_all.max(), 100)
    ax.plot(xs, slope * xs + intercept, 'k--', linewidth=2,
            label=f'Aggregate OLS (r={r_all:+.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Simpson's Paradox - Aggregate vs Subgroup Trends")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("Causal Inference Simulations")
    print("=" * 60 + "\n")
    demo_confounding()
    demo_simpsons_paradox()


if __name__ == "__main__":
    main()
