"""
Descriptive Statistics from Scratch — Weed Price Data
======================================================
Adapted from intro2stats "Basic Metrics" notebook.

Computes mean, median, mode, variance, and standard deviation
step by step (from first principles) on California weed-price
data, then verifies with pandas built-in methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic weed-price data (California HighQ, monthly means) ─────
# Representative values inspired by the original Weed_Price.csv dataset
CA_PRICES = np.array([
    248.75, 248.59, 248.63, 248.37, 248.02, 247.68, 247.36,
    246.85, 246.44, 246.06, 245.81, 245.48, 245.18, 244.87,
    244.55, 244.23, 243.89, 243.60, 243.34, 243.08, 242.85,
    242.64, 242.36, 242.15, 241.88, 241.64, 241.40, 241.14,
    240.91, 240.65, 240.42, 240.20, 239.96, 239.74, 239.52,
    239.28, 239.07, 238.81, 238.55, 238.34, 238.12, 237.90,
    237.66, 237.43, 237.19, 236.98, 236.76, 236.56,
])

NY_PRICES = np.array([
    350.50, 350.31, 350.02, 349.82, 349.55, 349.30, 349.04,
    348.78, 348.54, 348.27, 348.01, 347.78, 347.51, 347.26,
    346.98, 346.72, 346.48, 346.19, 345.93, 345.68, 345.44,
    345.17, 344.93, 344.67, 344.42, 344.18, 343.91, 343.68,
    343.43, 343.17, 342.93, 342.68, 342.44, 342.18, 341.93,
    341.67, 341.43, 341.16, 340.90, 340.66, 340.41, 340.16,
    339.92, 339.67, 339.41, 339.18, 338.93, 338.70,
])


# ── 1. Mean from scratch ───────────────────────────────────
def mean_from_scratch(data):
    """Sum all values and divide by the count."""
    return np.sum(data) / len(data)


# ── 2. Median from scratch ─────────────────────────────────
def median_from_scratch(data):
    """Sort and pick the middle value (or average of two middle values)."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2


# ── 3. Mode from scratch ───────────────────────────────────
def mode_from_scratch(data, decimals=1):
    """Round to *decimals* places, then find the most frequent value."""
    rounded = np.round(data, decimals)
    values, counts = np.unique(rounded, return_counts=True)
    return values[np.argmax(counts)]


# ── 4. Variance from scratch (Bessel-corrected) ────────────
def variance_from_scratch(data):
    """Sum of squared deviations divided by (n - 1)."""
    m = mean_from_scratch(data)
    return np.sum((data - m) ** 2) / (len(data) - 1)


# ── 5. Standard deviation from scratch ─────────────────────
def std_from_scratch(data):
    return np.sqrt(variance_from_scratch(data))


# ── 6. Covariance from scratch ─────────────────────────────
def covariance_from_scratch(x, y):
    """Cov(X, Y) = Σ (x_i - x̄)(y_i - ȳ) / (n - 1)."""
    n = len(x)
    mx, my = mean_from_scratch(x), mean_from_scratch(y)
    return np.sum((x - mx) * (y - my)) / (n - 1)


# ── 7. Correlation from scratch ────────────────────────────
def correlation_from_scratch(x, y):
    """Pearson r = Cov(X,Y) / (s_X * s_Y)."""
    return covariance_from_scratch(x, y) / (std_from_scratch(x) * std_from_scratch(y))


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Descriptive Statistics — California Weed Prices (HighQ)")
    print("=" * 60)

    data = CA_PRICES
    n = len(data)

    # from-scratch results
    m   = mean_from_scratch(data)
    med = median_from_scratch(data)
    mod = mode_from_scratch(data)
    var = variance_from_scratch(data)
    sd  = std_from_scratch(data)

    print(f"\n  n            = {n}")
    print(f"  Mean         = {m:.4f}")
    print(f"  Median       = {med:.4f}")
    print(f"  Mode (~0.1)  = {mod}")
    print(f"  Variance     = {var:.4f}")
    print(f"  Std Dev      = {sd:.4f}")

    # verify with pandas
    s = pd.Series(data)
    print(f"\n  pandas mean  = {s.mean():.4f}")
    print(f"  pandas median= {s.median():.4f}")
    print(f"  pandas var   = {s.var():.4f}")
    print(f"  pandas std   = {s.std():.4f}")

    # covariance and correlation (CA vs NY)
    print("\n" + "=" * 60)
    print("Covariance & Correlation — CA vs NY (HighQ)")
    print("=" * 60)
    cov = covariance_from_scratch(CA_PRICES, NY_PRICES)
    corr = correlation_from_scratch(CA_PRICES, NY_PRICES)
    print(f"  Covariance   = {cov:.4f}")
    print(f"  Correlation  = {corr:.4f}")

    df = pd.DataFrame({"CA": CA_PRICES, "NY": NY_PRICES})
    print(f"  pandas cov   = {df['CA'].cov(df['NY']):.4f}")
    print(f"  pandas corr  = {df['CA'].corr(df['NY']):.4f}")

    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(data, bins=15, edgecolor="white", alpha=0.7)
    axes[0].axvline(m, color="red", linestyle="--", label=f"Mean {m:.1f}")
    axes[0].axvline(med, color="blue", linestyle=":", label=f"Median {med:.1f}")
    axes[0].set_title("CA HighQ Price Distribution")
    axes[0].set_xlabel("Price ($)")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(data, vert=True)
    axes[1].set_title("Box Plot — CA HighQ")
    axes[1].set_ylabel("Price ($)")

    axes[2].scatter(CA_PRICES, NY_PRICES, alpha=0.6)
    axes[2].set_xlabel("CA HighQ ($)")
    axes[2].set_ylabel("NY HighQ ($)")
    axes[2].set_title(f"CA vs NY  (r = {corr:.3f})")

    plt.tight_layout()
    plt.savefig("descriptive_stats_weed_prices.png", dpi=150)
    plt.show()
    print("\nFigure saved: descriptive_stats_weed_prices.png")


if __name__ == "__main__":
    main()
