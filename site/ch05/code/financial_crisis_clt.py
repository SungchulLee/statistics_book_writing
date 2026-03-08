"""
Financial Crisis — CLT Failure with Dependent Random Variables
===============================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

The Central Limit Theorem requires independence.  When borrowers'
defaults are correlated (e.g. through a shared economic factor),
the Gaussian approximation based on the CLT can drastically
underestimate tail risk.

Model:
  - n = 100 borrowers, each defaults with probability theta
  - Independent case: theta = 2/3 is fixed
  - Dependent case: theta ~ Beta(2,1) is random (shared risk factor)

The dependent model produces a much heavier-tailed distribution
for the number of defaults, showing why the 2008 financial crisis
caught Gaussian-based risk models off guard.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def simulate_independent_defaults(n=100, p=2/3, n_sims=500_000):
    """Each borrower defaults independently with fixed probability p."""
    return np.random.binomial(n, p, size=n_sims)


def simulate_dependent_defaults(n=100, a=2, b=1, n_sims=500_000):
    """
    Shared risk: theta ~ Beta(a, b), then each borrower defaults
    independently with probability theta (conditional independence).
    Marginally, the defaults are dependent.
    """
    thetas = np.random.beta(a, b, size=n_sims)
    return np.array([np.random.binomial(n, th) for th in thetas])


def main():
    print("=" * 60)
    print("Financial Crisis — CLT Failure with Dependent Defaults")
    print("=" * 60)

    n = 100
    p = 2 / 3
    n_sims = 500_000

    print(f"\n  n = {n} borrowers, default probability = {p:.4f}")
    print(f"  Independent model: theta = {p:.4f} (fixed)")
    print(f"  Dependent model:   theta ~ Beta(2, 1), E[theta] = {p:.4f}")

    # Simulations
    d_indep = simulate_independent_defaults(n, p, n_sims)
    d_dep = simulate_dependent_defaults(n, 2, 1, n_sims)

    # Probability of catastrophic loss (> 90 defaults)
    threshold = 90
    p_indep_extreme = np.mean(d_indep > threshold)
    p_dep_extreme = np.mean(d_dep > threshold)
    p_gaussian = 1 - stats.norm.cdf(threshold, n * p,
                                     np.sqrt(n * p * (1 - p)))

    print(f"\n  P(defaults > {threshold}):")
    print(f"    Independent model:    {p_indep_extreme:.6f}")
    print(f"    Gaussian approx (CLT): {p_gaussian:.6f}")
    print(f"    Dependent model:      {p_dep_extreme:.6f}")
    print(f"    Ratio dependent/Gaussian: "
          f"{p_dep_extreme / max(p_gaussian, 1e-12):.1f}x")

    # Theoretical PMF for dependent case:
    # P(D = d) = 2(d+1) / ((n+1)(n+2)) for Beta(2,1)
    d_vals = np.arange(0, n + 1)
    pmf_dep_theory = 2 * (d_vals + 1) / ((n + 1) * (n + 2))

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: all three on one plot
    ax = axes[0]
    bins = np.arange(-0.5, n + 1.5, 1)
    ax.hist(d_dep, bins=bins, density=True, alpha=0.5, color="tomato",
            edgecolor="white", label="Dependent defaults (MC)")
    ax.plot(d_vals, stats.binom.pmf(d_vals, n, p), "o", color="black",
            ms=3, label="Independent defaults")
    x_norm = np.linspace(0, n, 300)
    ax.plot(x_norm, stats.norm.pdf(x_norm, n * p, np.sqrt(n * p * (1 - p))),
            color="gray", lw=3, linestyle="--",
            label="Gaussian approx (CLT)")
    ax.set_xlabel("Number of defaults d")
    ax.set_ylabel("Probability")
    ax.set_title("Default Distributions")
    ax.legend(fontsize=9)

    # Panel 2: focus on the tail
    ax = axes[1]
    tail_range = range(80, n + 1)
    mc_tail = np.array([np.mean(d_dep == d) for d in tail_range])
    binom_tail = stats.binom.pmf(list(tail_range), n, p)
    gauss_tail = stats.norm.pdf(list(tail_range), n * p,
                                np.sqrt(n * p * (1 - p)))
    ax.bar(list(tail_range), mc_tail, color="tomato", alpha=0.7,
           label="Dependent (MC)")
    ax.plot(list(tail_range), binom_tail, "ko-", ms=4,
            label="Independent (Binomial)")
    ax.plot(list(tail_range), gauss_tail, "g--", lw=2,
            label="Gaussian approx")
    ax.set_xlabel("Number of defaults d")
    ax.set_ylabel("Probability")
    ax.set_title("Tail Risk (d >= 80)")
    ax.legend(fontsize=9)

    # Panel 3: Beta prior and how theta varies
    ax = axes[2]
    theta_grid = np.linspace(0, 1, 300)
    ax.plot(theta_grid, stats.beta.pdf(theta_grid, 2, 1), "b-", lw=2.5,
            label="theta ~ Beta(2,1)")
    ax.axvline(p, color="red", linestyle="--", lw=2,
               label=f"E[theta] = {p:.3f}")
    ax.fill_between(theta_grid[theta_grid > 0.9],
                    stats.beta.pdf(theta_grid[theta_grid > 0.9], 2, 1),
                    alpha=0.3, color="tomato",
                    label=f"P(theta > 0.9) = {1 - stats.beta.cdf(0.9, 2, 1):.3f}")
    ax.set_xlabel("theta (shared default probability)")
    ax.set_ylabel("Density")
    ax.set_title("Prior on Shared Risk Factor")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("financial_crisis_clt.png", dpi=150)
    plt.show()
    print("\nFigure saved: financial_crisis_clt.png")


if __name__ == "__main__":
    main()
