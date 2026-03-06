"""
Rejection Region Demonstration — One-Tail and Two-Tail Tests
==============================================================
Adapted from Basic-Statistics-With-Python plot_material.py.

Shows shaded rejection regions on the t-distribution for:
  1. Two-tailed test  (H1: mu != mu_0)
  2. Left-tailed test (H1: mu < mu_0)
  3. Right-tailed test (H1: mu > mu_0)

Uses height data to illustrate rejection in original units and
in t-statistic units.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

# ── Synthetic height data (inspired by 500-Person dataset) ──
MALE_HEIGHTS = stats.norm.rvs(loc=170, scale=8, size=250)


def two_tail_demo(alpha=0.05):
    """Plot two-tailed rejection region in original and t-units."""
    data = MALE_HEIGHTS
    n = len(data)
    df = n - 1
    xbar = data.mean()
    s = data.std(ddof=1)
    se = s / np.sqrt(n)
    mu0 = 172  # null hypothesis value

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    t_stat = (xbar - mu0) / se

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # --- Panel 1: original units ---
    ax = axes[0]
    x = np.linspace(mu0 - 5, mu0 + 5, 300)
    y = stats.t.pdf((x - mu0) / se, df) / se
    ax.plot(x, y, "tomato", lw=2)

    # rejection regions
    rej_lo = mu0 - t_crit * se
    rej_hi = mu0 + t_crit * se
    x_lo = np.linspace(mu0 - 5, rej_lo, 60)
    x_hi = np.linspace(rej_hi, mu0 + 5, 60)
    ax.fill_between(x_lo, stats.t.pdf((x_lo - mu0) / se, df) / se,
                    color="tomato", alpha=0.5)
    ax.fill_between(x_hi, stats.t.pdf((x_hi - mu0) / se, df) / se,
                    color="tomato", alpha=0.5)
    ax.axvline(xbar, color="blue", linestyle="--", label=f"x̄ = {xbar:.2f}")
    ax.set_title(f"Two-Tailed Rejection Region (original units, mu_0 = {mu0})")
    ax.set_xlabel("Height (cm)")
    ax.legend()
    ax.set_ylim(0, max(y) * 1.15)

    # --- Panel 2: t-statistic units ---
    ax = axes[1]
    x_t = np.linspace(-5, 5, 300)
    y_t = stats.t.pdf(x_t, df)
    ax.plot(x_t, y_t, "tomato", lw=2)

    x_lo_t = np.linspace(-5, -t_crit, 60)
    x_hi_t = np.linspace(t_crit, 5, 60)
    ax.fill_between(x_lo_t, stats.t.pdf(x_lo_t, df),
                    color="tomato", alpha=0.5)
    ax.fill_between(x_hi_t, stats.t.pdf(x_hi_t, df),
                    color="tomato", alpha=0.5)
    ax.axvline(t_stat, color="blue", linestyle="--",
               label=f"t-stat = {t_stat:.2f}")
    ax.set_title("Two-Tailed Rejection Region (t-statistic)")
    ax.set_xlabel("t")
    ax.legend()
    ax.set_ylim(0, max(y_t) * 1.15)

    plt.tight_layout()
    plt.savefig("rejection_region_two_tail.png", dpi=150)
    plt.show()

    return t_stat, t_crit


def one_tail_demo(alpha=0.05):
    """Plot left-tailed and right-tailed rejection regions."""
    df = 100
    x = np.linspace(-5, 5, 300)
    y = stats.t.pdf(x, df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # left tail
    ax = axes[0]
    ax.plot(x, y, "tomato", lw=2)
    t_lo = stats.t.ppf(alpha, df)
    x_rej = np.linspace(-5, t_lo, 60)
    ax.fill_between(x_rej, stats.t.pdf(x_rej, df), color="tomato", alpha=0.5)
    ax.annotate(r"$H_0: \mu = \mu_0$" + "\n" + r"$H_1: \mu < \mu_0$",
                (-4.5, 0.35), fontsize=13)
    ax.set_title("Left-Tailed Rejection Region")
    ax.set_ylim(0, 0.45)

    # right tail
    ax = axes[1]
    ax.plot(x, y, "tomato", lw=2)
    t_hi = stats.t.ppf(1 - alpha, df)
    x_rej = np.linspace(t_hi, 5, 60)
    ax.fill_between(x_rej, stats.t.pdf(x_rej, df), color="tomato", alpha=0.5)
    ax.annotate(r"$H_0: \mu = \mu_0$" + "\n" + r"$H_1: \mu > \mu_0$",
                (3.5, 0.35), fontsize=13)
    ax.set_title("Right-Tailed Rejection Region")
    ax.set_ylim(0, 0.45)

    plt.tight_layout()
    plt.savefig("rejection_region_one_tail.png", dpi=150)
    plt.show()


def main():
    print("=" * 60)
    print("Rejection Region Demonstrations")
    print("=" * 60)

    t_stat, t_crit = two_tail_demo()
    p_val = 2 * stats.t.cdf(-abs(t_stat), len(MALE_HEIGHTS) - 1)
    print(f"\n  Two-tailed test: H0: mu = 172")
    print(f"  x̄ = {MALE_HEIGHTS.mean():.2f},  s = {MALE_HEIGHTS.std(ddof=1):.2f}")
    print(f"  t-stat = {t_stat:.4f},  t-crit = +/-{t_crit:.4f}")
    print(f"  p-value = {p_val:.4f}")
    if abs(t_stat) > t_crit:
        print("  => Reject H0")
    else:
        print("  => Fail to reject H0")

    one_tail_demo()
    print("\nFigures saved: rejection_region_two_tail.png, rejection_region_one_tail.png")


if __name__ == "__main__":
    main()
