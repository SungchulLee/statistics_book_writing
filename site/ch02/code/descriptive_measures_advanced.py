"""
Advanced Descriptive Measures — Geometric Mean, Chebyshev, Pop vs Sample
=========================================================================
Adapted from Basic-Statistics-With-Python Chapter 1 notebook.

Covers three topics not in the basic descriptive-statistics code:
1. Geometric mean for portfolio returns
2. Chebyshev's theorem (minimum data within k std devs)
3. Population variance vs sample variance (unbiasedness simulation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── 1. Geometric vs Arithmetic Mean for Returns ────────────
def geometric_vs_arithmetic_mean():
    """
    The arithmetic mean of returns overestimates compound growth.
    The geometric mean gives the true compound growth rate.
    """
    returns = np.array([0.36, 0.23, -0.48, -0.30, 0.15, 0.31])

    arith_mean = np.mean(returns)
    # geometric mean of (1 + r) - 1
    geo_mean = stats.mstats.gmean(1 + returns) - 1

    print("  Period returns:", returns)
    print(f"  Arithmetic mean: {arith_mean:.4f}  ({arith_mean*100:.2f}%)")
    print(f"  Geometric  mean: {geo_mean:.4f}  ({geo_mean*100:.2f}%)")
    print(f"  Compound value of $1: ${np.prod(1 + returns):.4f}")
    print(f"  Using geo mean:       ${(1 + geo_mean)**len(returns):.4f}")
    return arith_mean, geo_mean


# ── 2. Chebyshev's Theorem ─────────────────────────────────
def chebyshev_demo():
    """
    For ANY distribution, at least 1 - 1/k^2 of data lie within
    k standard deviations of the mean.
    """
    def chebyshev(k):
        return 1 - 1 / k**2

    # example: heights with mean=174, std=4 → within 166–182 (k=2)
    mu, sigma = 174, 4
    k = (182 - mu) / sigma
    pct = chebyshev(k)
    print(f"\n  Height example: mu={mu}, sigma={sigma}")
    print(f"    Range [{mu - k*sigma:.0f}, {mu + k*sigma:.0f}] → "
          f"k={k:.1f} → at least {pct*100:.0f}% of data")

    z_vals = np.arange(1.1, 10, 0.1)
    cheb_vals = [chebyshev(z) for z in z_vals]
    return z_vals, cheb_vals


# ── 3. Population vs Sample Variance (Unbiasedness) ────────
def pop_vs_sample_variance(pop_size=1000, sample_size=100, n_samples=10000):
    """
    Show that s^2 (ddof=1) is unbiased for sigma^2 while
    the naive estimator (ddof=0) is biased.
    """
    population = np.random.normal(170, 10, pop_size)
    pop_var = np.var(population)

    biased_vars = []
    unbiased_vars = []
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        biased_vars.append(np.var(sample, ddof=0))
        unbiased_vars.append(np.var(sample, ddof=1))

    biased_vars = np.array(biased_vars)
    unbiased_vars = np.array(unbiased_vars)

    print(f"\n  Population variance: {pop_var:.4f}")
    print(f"  Mean of biased estimator (ddof=0):   {biased_vars.mean():.4f}")
    print(f"  Mean of unbiased estimator (ddof=1): {unbiased_vars.mean():.4f}")
    return pop_var, biased_vars, unbiased_vars


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Advanced Descriptive Measures")
    print("=" * 60)

    # 1. geometric mean
    print("\n--- 1. Geometric vs Arithmetic Mean (Portfolio Returns) ---")
    arith, geo = geometric_vs_arithmetic_mean()

    # 2. Chebyshev
    print("\n--- 2. Chebyshev's Theorem ---")
    z_vals, cheb_vals = chebyshev_demo()

    # 3. pop vs sample var
    print("\n--- 3. Population vs Sample Variance ---")
    pop_var, bv, uv = pop_vs_sample_variance()

    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # panel 1: returns comparison
    ax = axes[0]
    returns = np.array([0.36, 0.23, -0.48, -0.30, 0.15, 0.31])
    cumulative = np.cumprod(1 + returns)
    ax.plot(range(1, len(cumulative)+1), cumulative, "o-", label="Actual")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(cumulative[-1], color="red", linestyle="--",
               label=f"Final = {cumulative[-1]:.3f}")
    ax.set_xlabel("Period")
    ax.set_ylabel("Cumulative value of $1")
    ax.set_title("Compound Returns")
    ax.legend(fontsize=8)

    # panel 2: Chebyshev
    ax = axes[1]
    ax.plot(z_vals, cheb_vals, lw=2, color="seagreen")
    ax.set_xlabel("k (std deviations)")
    ax.set_ylabel("Minimum fraction")
    ax.set_title("Chebyshev's Theorem: 1 - 1/k^2")
    ax.axhline(0.75, color="grey", linestyle=":", alpha=0.5)
    ax.annotate("k=2: >= 75%", (2, 0.75), fontsize=9,
                xytext=(4, 0.6), arrowprops=dict(arrowstyle="->"))

    # panel 3: variance estimators
    ax = axes[2]
    ax.hist(bv, bins=40, alpha=0.5, label="Biased (ddof=0)", density=True)
    ax.hist(uv, bins=40, alpha=0.5, label="Unbiased (ddof=1)", density=True)
    ax.axvline(pop_var, color="red", linestyle="--", lw=2,
               label=f"True sigma^2 = {pop_var:.1f}")
    ax.set_xlabel("Variance estimate")
    ax.set_title("Population vs Sample Variance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("descriptive_measures_advanced.png", dpi=150)
    plt.show()
    print("\nFigure saved: descriptive_measures_advanced.png")


if __name__ == "__main__":
    main()
