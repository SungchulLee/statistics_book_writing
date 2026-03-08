"""
Covariance and Correlation from First Principles
==================================================
Adapted from intro2stats "Basic Metrics" notebook.

Builds covariance and Pearson correlation step by step,
verifies against pandas / numpy, and shows why
"correlation != causation" with a visual.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic state-level weed prices (HighQ weekly means) ──
# CA and NY prices move together (common macro trend) but
# with state-specific noise.  This produces a strong positive
# correlation even though neither state *causes* the other's price.

WEEKS = 48
trend = np.linspace(0, -12, WEEKS)          # shared downward trend

CA_PRICES = 248.0 + trend + np.random.normal(0, 0.5, WEEKS)
NY_PRICES = 350.0 + trend * 0.8 + np.random.normal(0, 0.6, WEEKS)


# ── Step-by-step covariance ─────────────────────────────────
def covariance_step_by_step(x, y):
    """
    Cov(X, Y) = Σ (x_i - x̄)(y_i - ȳ) / (n - 1)

    Returns the covariance and intermediate deviations.
    """
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev) / (n - 1)
    return cov, x_dev, y_dev


# ── Step-by-step Pearson r ──────────────────────────────────
def pearson_r_step_by_step(x, y):
    """
    r = Cov(X, Y) / (s_X * s_Y)
    """
    cov, _, _ = covariance_step_by_step(x, y)
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    return cov / (sx * sy)


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Covariance & Correlation — Step by Step")
    print("=" * 60)

    x, y = CA_PRICES, NY_PRICES

    # step-by-step
    cov, x_dev, y_dev = covariance_step_by_step(x, y)
    r = pearson_r_step_by_step(x, y)

    print(f"\n  CA mean       = {x.mean():.4f}")
    print(f"  NY mean       = {y.mean():.4f}")
    print(f"  CA std (s)    = {x.std(ddof=1):.4f}")
    print(f"  NY std (s)    = {y.std(ddof=1):.4f}")
    print(f"  Covariance    = {cov:.4f}")
    print(f"  Pearson r     = {r:.4f}")

    # verify
    df = pd.DataFrame({"CA": x, "NY": y})
    print(f"\n  pandas cov    = {df['CA'].cov(df['NY']):.4f}")
    print(f"  pandas corr   = {df['CA'].corr(df['NY']):.4f}")
    print(f"  numpy corrcoef= {np.corrcoef(x, y)[0, 1]:.4f}")

    # caution
    print("\n" + "-" * 60)
    print("CAUTION: Correlation != Causation")
    print("-" * 60)
    print("  CA and NY prices are correlated because both follow a")
    print("  common macro trend (confounding variable), not because")
    print("  one state's price causes the other.")

    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # scatter
    axes[0].scatter(x, y, alpha=0.6, edgecolors="grey")
    z = np.polyfit(x, y, 1)
    axes[0].plot(np.sort(x), np.polyval(z, np.sort(x)),
                 color="red", linewidth=2)
    axes[0].set_xlabel("CA HighQ ($)")
    axes[0].set_ylabel("NY HighQ ($)")
    axes[0].set_title(f"Scatter  (r = {r:.3f})")

    # deviation products
    products = x_dev * y_dev
    colours = ["steelblue" if p > 0 else "salmon" for p in products]
    axes[1].bar(range(WEEKS), products, color=colours, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("(x−x̄)(y−ȳ)")
    axes[1].set_title("Deviation Products")

    # time series overlay
    weeks = np.arange(WEEKS)
    axes[2].plot(weeks, x, label="CA", marker="o", markersize=3)
    axes[2].plot(weeks, y, label="NY", marker="s", markersize=3)
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("Price ($)")
    axes[2].set_title("Price Series — Common Trend")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("covariance_from_scratch.png", dpi=150)
    plt.show()
    print("\nFigure saved: covariance_from_scratch.png")


if __name__ == "__main__":
    main()
