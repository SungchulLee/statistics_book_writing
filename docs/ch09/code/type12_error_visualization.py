"""
Type I and Type II Error Visualization
========================================
Adapted from Basic-Statistics-With-Python plot_material.py.

Plots two overlapping normal distributions (null and alternative)
and shades the regions corresponding to:
  - Type I error (alpha): rejecting H0 when H0 is true
  - Type II error (beta): failing to reject H0 when H1 is true
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def plot_type12_error(null_loc=0, alt_loc=3, alpha=0.05):
    """
    Visualise Type I and Type II error areas.

    Parameters
    ----------
    null_loc : float
        Mean of the null distribution.
    alt_loc : float
        Mean of the alternative distribution.
    alpha : float
        Significance level (two-sided).
    """
    x = np.linspace(null_loc - 4, alt_loc + 4, 400)
    y_null = stats.norm.pdf(x, loc=null_loc)
    y_alt = stats.norm.pdf(x, loc=alt_loc)

    # critical value (right tail for one-sided illustration)
    z_crit = stats.norm.ppf(1 - alpha, loc=null_loc)

    fig, ax = plt.subplots(figsize=(12, 5))

    # distributions
    ax.plot(x, y_null, "b-", lw=2, label="Null distribution")
    ax.plot(x, y_alt, "r-", lw=2, label="Alternative distribution")

    # Type I error: area under null beyond critical value
    mask_t1 = x >= z_crit
    ax.fill_between(x[mask_t1], y_null[mask_t1], alpha=0.4, color="blue",
                     label=f"Type I error (alpha)")

    # Type II error: area under alternative below critical value
    mask_t2 = x <= z_crit
    ax.fill_between(x[mask_t2], y_alt[mask_t2], alpha=0.3, color="red",
                     label=f"Type II error (beta)")

    # annotations
    ax.annotate("Null", (null_loc, max(y_null) * 0.55), fontsize=14,
                ha="center", color="blue")
    ax.annotate("Alternative", (alt_loc, max(y_alt) * 0.55), fontsize=14,
                ha="center", color="red")
    ax.axvline(z_crit, color="black", linestyle="--", alpha=0.6,
               label=f"Critical value = {z_crit:.2f}")

    # power annotation
    power = 1 - stats.norm.cdf(z_crit, loc=alt_loc)
    beta = stats.norm.cdf(z_crit, loc=alt_loc)
    ax.set_title("Type I and Type II Errors", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(y_null), max(y_alt)) * 1.15)
    ax.set_xlabel("Test statistic")
    ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig("type12_error_visualization.png", dpi=150)
    plt.show()

    return power, beta


def main():
    print("=" * 60)
    print("Type I and Type II Error Visualization")
    print("=" * 60)

    power, beta = plot_type12_error(null_loc=0, alt_loc=3, alpha=0.05)
    print(f"\n  Null mean = 0,  Alternative mean = 3")
    print(f"  alpha = 0.05")
    print(f"  Type II error (beta) = {beta:.4f}")
    print(f"  Power (1 - beta) = {power:.4f}")

    print("\n--- Effect of separation ---")
    for sep in [1, 2, 3, 4, 5]:
        z_c = stats.norm.ppf(0.95)
        pwr = 1 - stats.norm.cdf(z_c, loc=sep)
        print(f"  Separation = {sep}:  Power = {pwr:.4f}")

    print("\nFigure saved: type12_error_visualization.png")


if __name__ == "__main__":
    main()
