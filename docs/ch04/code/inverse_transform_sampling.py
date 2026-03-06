"""
Inverse Transform Sampling
============================
Adapted from ps4ds (Probability and Statistics for Data Science).

Given a CDF F, we can generate samples from the corresponding
distribution by computing X = F^{-1}(U) where U ~ Uniform(0, 1).

Demonstrates:
1. Exponential distribution: F^{-1}(u) = -ln(1 - u) / lambda
2. Comparison of transformed samples with true exponential PDF
3. Extension to other distributions (Cauchy, Rayleigh)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def inverse_transform_exponential(n=10_000, lam=1.0):
    """Generate exponential samples via inverse CDF."""
    u = np.random.uniform(0, 1, n)
    x = -np.log(1 - u) / lam
    return u, x


def inverse_transform_cauchy(n=10_000):
    """Generate standard Cauchy samples via inverse CDF."""
    u = np.random.uniform(0, 1, n)
    x = np.tan(np.pi * (u - 0.5))
    return u, x


def main():
    print("=" * 60)
    print("Inverse Transform Sampling")
    print("=" * 60)

    n = 10_000
    n_bins = 50
    lam = 1.0

    u_exp, x_exp = inverse_transform_exponential(n, lam)
    u_cauchy, x_cauchy = inverse_transform_cauchy(n)

    print(f"\n  Exponential(lambda={lam}):")
    print(f"    Theoretical mean = {1/lam:.2f}, "
          f"sample mean = {x_exp.mean():.4f}")
    print(f"    Theoretical var  = {1/lam**2:.2f}, "
          f"sample var  = {x_exp.var():.4f}")

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: exponential
    ax = axes[0, 0]
    ax.hist(u_exp, bins=n_bins, density=True, color="lightgray",
            edgecolor="black")
    ax.set_title("Step 1: U ~ Uniform(0,1)")
    ax.set_xlabel("u")
    ax.set_ylabel("Density")

    ax = axes[0, 1]
    ax.hist(x_exp, bins=n_bins, density=True, color="steelblue",
            edgecolor="white", alpha=0.7, label="Transformed samples")
    t = np.linspace(0, np.max(x_exp), 300)
    ax.plot(t, stats.expon.pdf(t, scale=1/lam), "r-", lw=2.5,
            label=f"Exp(lambda={lam}) PDF")
    ax.set_title("Step 2: X = -ln(1-U) / lambda")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)

    ax = axes[0, 2]
    # show the inverse CDF mapping
    u_grid = np.linspace(0.001, 0.999, 300)
    ax.plot(u_grid, -np.log(1 - u_grid) / lam, "b-", lw=2)
    ax.set_title("Inverse CDF: F^{-1}(u)")
    ax.set_xlabel("u")
    ax.set_ylabel("x = F^{-1}(u)")
    ax.grid(True, alpha=0.3)

    # Row 2: Cauchy
    ax = axes[1, 0]
    ax.hist(u_cauchy, bins=n_bins, density=True, color="lightgray",
            edgecolor="black")
    ax.set_title("U ~ Uniform(0,1)")
    ax.set_xlabel("u")
    ax.set_ylabel("Density")

    ax = axes[1, 1]
    x_clipped = np.clip(x_cauchy, -20, 20)
    ax.hist(x_clipped, bins=80, density=True, color="coral",
            edgecolor="white", alpha=0.7, label="Transformed samples")
    t2 = np.linspace(-20, 20, 500)
    ax.plot(t2, stats.cauchy.pdf(t2), "k-", lw=2.5,
            label="Cauchy PDF")
    ax.set_title("X = tan(pi*(U - 0.5))")
    ax.set_xlabel("x")
    ax.set_xlim(-20, 20)
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.plot(u_grid, np.tan(np.pi * (u_grid - 0.5)), "b-", lw=2)
    ax.set_ylim(-20, 20)
    ax.set_title("Cauchy Inverse CDF")
    ax.set_xlabel("u")
    ax.set_ylabel("x = tan(pi*(u-0.5))")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Inverse Transform Sampling: Uniform -> Target Distribution",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("inverse_transform_sampling.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("\nFigure saved: inverse_transform_sampling.png")


if __name__ == "__main__":
    main()
