"""
Discrete Distributions Suite — Binomial, Poisson, Geometric, Hypergeometric
=============================================================================
Adapted from Basic-Statistics-With-Python Chapter 2 notebook.

Demonstrates four key discrete distributions with financial examples:
  1. Binomial  — bad-credit encounters per month
  2. Poisson   — rare black-swan events over 5 years
  3. Geometric — failures before first success
  4. Hypergeometric — sampling without replacement
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── 1. Binomial Distribution ───────────────────────────────
def binomial_demo():
    """
    A banker meets 50 loan applicants per month.
    30% have bad credit history.
    """
    n, p = 50, 0.3
    k_vals = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k_vals, n, p)

    # probability of exact 14 bad-credit applicants
    p14 = stats.binom.pmf(14, n, p)
    # CDF: P(X <= 12)
    p_le12 = stats.binom.cdf(12, n, p)

    print(f"  Binomial(n={n}, p={p})")
    print(f"    P(X = 14) = {p14:.4f}")
    print(f"    P(X <= 12) = {p_le12:.4f}")
    print(f"    E[X] = {n*p:.1f}, Var(X) = {n*p*(1-p):.2f}")
    return k_vals, pmf


# ── 2. Poisson Distribution ────────────────────────────────
def poisson_demo():
    """
    A trader makes 20 trades/month over 5 years.
    Probability of a complete wipe-out per trade is 1/1000.
    lambda = n*p = 1200 * 0.001 = 1.2
    """
    lam = 20 * 12 * 5 * (1 / 1000)  # = 1.2
    k_vals = np.arange(0, 10)
    pmf = stats.poisson.pmf(k_vals, lam)

    p_exactly2 = stats.poisson.pmf(2, lam)
    p_more2 = 1 - stats.poisson.cdf(2, lam)

    print(f"\n  Poisson(lambda={lam})")
    print(f"    P(X = 2 wipe-outs in 5 yr) = {p_exactly2:.4f}")
    print(f"    P(X > 2) = {p_more2:.4f}")
    return k_vals, pmf, lam


# ── 3. Geometric Distribution ──────────────────────────────
def geometric_demo():
    """
    Probability of success = 0.3.
    How many failures before first success?
    """
    p = 0.3
    k_vals = np.arange(0, 20)
    # P(k failures before 1st success) = (1-p)^k * p
    pmf = stats.geom.pmf(k_vals + 1, p)  # scipy counts trials, not failures

    p5 = (1 - p) ** 5 * p
    cdf5 = stats.geom.cdf(6, p)  # P(X <= 5 failures) = P(trials <= 6)

    print(f"\n  Geometric(p={p})")
    print(f"    P(exactly 5 failures before 1st success) = {p5:.4f}")
    print(f"    P(5 or fewer failures) = {cdf5:.4f}")
    print(f"    E[failures] = {(1-p)/p:.2f}")
    return k_vals, pmf


# ── 4. Hypergeometric Distribution ─────────────────────────
def hypergeometric_demo():
    """
    Population N=100, K=20 defectives. Draw n=5 without replacement.
    What is P(exactly 2 defectives)?
    """
    N, K, n = 100, 20, 5
    k_vals = np.arange(0, n + 1)
    pmf = stats.hypergeom.pmf(k_vals, N, K, n)

    p2 = stats.hypergeom.pmf(2, N, K, n)
    # manual: C(K,k)*C(N-K,n-k) / C(N,n)
    p2_manual = (special.comb(K, 2) * special.comb(N - K, n - 2)
                 / special.comb(N, n))

    print(f"\n  Hypergeometric(N={N}, K={K}, n={n})")
    print(f"    P(X = 2) = {p2:.4f}  (manual = {p2_manual:.4f})")
    print(f"    E[X] = {n*K/N:.2f}")
    return k_vals, pmf


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Discrete Distributions Suite")
    print("=" * 60)

    bk, bpmf = binomial_demo()
    pk, ppmf, lam = poisson_demo()
    gk, gpmf = geometric_demo()
    hk, hpmf = hypergeometric_demo()

    # visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(bk, bpmf, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(50 * 0.3, color="red", linestyle="--", label="E[X] = 15")
    ax.set_title("Binomial(n=50, p=0.3)")
    ax.set_xlabel("k (bad-credit applicants)")
    ax.set_ylabel("P(X = k)")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(pk, ppmf, color="seagreen", edgecolor="white", alpha=0.8)
    ax.set_title(f"Poisson(lambda={lam})")
    ax.set_xlabel("k (wipe-out events in 5 yr)")
    ax.set_ylabel("P(X = k)")

    ax = axes[1, 0]
    ax.bar(gk, gpmf, color="coral", edgecolor="white", alpha=0.8)
    ax.set_title("Geometric(p=0.3)")
    ax.set_xlabel("k (failures before 1st success)")
    ax.set_ylabel("P(X = k)")

    ax = axes[1, 1]
    ax.bar(hk, hpmf, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.set_title("Hypergeometric(N=100, K=20, n=5)")
    ax.set_xlabel("k (defectives drawn)")
    ax.set_ylabel("P(X = k)")

    plt.tight_layout()
    plt.savefig("discrete_distributions_suite.png", dpi=150)
    plt.show()
    print("\nFigure saved: discrete_distributions_suite.png")


if __name__ == "__main__":
    main()
