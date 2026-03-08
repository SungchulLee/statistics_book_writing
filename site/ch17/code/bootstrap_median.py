#!/usr/bin/env python3
"""
Bootstrap Resampling: Estimating Standard Error of the Median
=============================================================

Demonstrates the bootstrap method for computing standard errors
when no closed-form formula is available.

The median is particularly useful for:
- Data with outliers
- Skewed distributions
- Robust estimation

Unlike the mean, the median has no simple formula for SE, making bootstrap ideal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def simulate_income_data(n=5000, seed=1):
    """
    Create synthetic income data with realistic right-skewed distribution.

    Parameters:
    -----------
    n : int
        Number of observations
    seed : int
        Random seed

    Returns:
    --------
    np.ndarray : Income values
    """
    np.random.seed(seed)
    income = np.random.exponential(scale=50000, size=n) + 20000
    return income


def bootstrap_median(data, n_bootstrap=1000):
    """
    Compute bootstrap distribution of the median.

    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Original sample
    n_bootstrap : int
        Number of bootstrap resamples

    Returns:
    --------
    np.ndarray : Bootstrap medians
    """
    bootstrap_medians = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = resample(data)
        # Compute median of bootstrap sample
        bootstrap_medians.append(np.median(bootstrap_sample))

    return np.array(bootstrap_medians)


def bootstrap_mean_for_comparison(data, n_bootstrap=1000):
    """
    Compute bootstrap distribution of the mean for comparison.

    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Original sample
    n_bootstrap : int
        Number of bootstrap resamples

    Returns:
    --------
    np.ndarray : Bootstrap means
    """
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data)
        bootstrap_means.append(np.mean(bootstrap_sample))

    return np.array(bootstrap_means)


def compute_bootstrap_statistics(data, bootstrap_dist, original_stat, stat_name):
    """
    Compute and print bootstrap statistics.

    Parameters:
    -----------
    data : np.ndarray
        Original sample
    bootstrap_dist : np.ndarray
        Bootstrap distribution
    original_stat : float
        Original statistic value
    stat_name : str
        Name of statistic
    """
    bootstrap_mean = bootstrap_dist.mean()
    bootstrap_std = bootstrap_dist.std()
    bias = bootstrap_mean - original_stat

    print(f"\n{stat_name.upper()}")
    print("-" * 60)
    print(f"Original sample {stat_name}: ${original_stat:>12,.0f}")
    print(f"Mean of bootstrap distribution: ${bootstrap_mean:>8,.0f}")
    print(f"Standard error ({stat_name}):    ${bootstrap_std:>12,.0f}")
    print(f"Bias of {stat_name}:              ${bias:>12,.0f}")

    return bootstrap_mean, bootstrap_std, bias


def visualize_bootstrap_distributions(data, bootstrap_medians, bootstrap_means,
                                      original_median, original_mean):
    """
    Create visualizations of bootstrap distributions.

    Parameters:
    -----------
    data : np.ndarray
        Original sample
    bootstrap_medians : np.ndarray
        Bootstrap median distribution
    bootstrap_means : np.ndarray
        Bootstrap mean distribution
    original_median : float
        Original sample median
    original_mean : float
        Original sample mean
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ===== Top Left: Data Histogram =====
    ax = axes[0, 0]
    ax.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(original_median, color='red', linestyle='--', linewidth=2.5,
               label=f'Median: ${original_median:,.0f}')
    ax.axvline(original_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'Mean: ${original_mean:,.0f}')
    ax.set_title('Original Sample Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Income ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # ===== Top Right: Bootstrap Median Distribution =====
    ax = axes[0, 1]
    ax.hist(bootstrap_medians, bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(original_median, color='darkred', linestyle='--', linewidth=2.5,
               label=f'Original median')
    ax.axvline(bootstrap_medians.mean(), color='blue', linestyle='--', linewidth=2.5,
               label=f'Bootstrap mean')
    ax.set_title('Bootstrap Distribution of Median', fontsize=12, fontweight='bold')
    ax.set_xlabel('Median Income ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # ===== Bottom Left: Bootstrap Mean Distribution =====
    ax = axes[1, 0]
    ax.hist(bootstrap_means, bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(original_mean, color='darkgreen', linestyle='--', linewidth=2.5,
               label=f'Original mean')
    ax.axvline(bootstrap_means.mean(), color='blue', linestyle='--', linewidth=2.5,
               label=f'Bootstrap mean')
    ax.set_title('Bootstrap Distribution of Mean', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Income ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # ===== Bottom Right: Side-by-side Comparison =====
    ax = axes[1, 1]
    # Normalize for comparison
    med_normalized = (bootstrap_medians - original_median) / bootstrap_medians.std()
    mean_normalized = (bootstrap_means - original_mean) / bootstrap_means.std()

    ax.hist(med_normalized, bins=30, alpha=0.6, color='coral', edgecolor='black',
            label='Median (normalized)', density=True)
    ax.hist(mean_normalized, bins=30, alpha=0.6, color='lightgreen', edgecolor='black',
            label='Mean (normalized)', density=True)

    # Overlay normal distribution
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x), 'k-', linewidth=2, label='Standard normal')

    ax.set_title('Bootstrap Distributions (Normalized)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Standardized Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


def robustness_comparison(data):
    """
    Demonstrate robustness of median vs mean to outliers.

    Parameters:
    -----------
    data : np.ndarray
        Original sample
    """
    print("\n" + "=" * 80)
    print("ROBUSTNESS COMPARISON: MEDIAN VS MEAN")
    print("=" * 80)

    # Original statistics
    original_mean = np.mean(data)
    original_median = np.median(data)

    print(f"\nOriginal Data (n={len(data)}):")
    print(f"  Mean:   ${original_mean:,.0f}")
    print(f"  Median: ${original_median:,.0f}")

    # Add extreme outlier
    data_with_outlier = np.append(data, 1000000)  # Add $1M outlier

    mean_with_outlier = np.mean(data_with_outlier)
    median_with_outlier = np.median(data_with_outlier)

    change_mean = mean_with_outlier - original_mean
    change_median = median_with_outlier - original_median

    print(f"\nAfter Adding $1M Outlier:")
    print(f"  Mean:   ${mean_with_outlier:,.0f} (change: ${change_mean:,.0f}, {change_mean/original_mean*100:.2f}%)")
    print(f"  Median: ${median_with_outlier:,.0f} (change: ${change_median:,.0f}, {change_median/original_median*100:.2f}%)")

    print("\nKey Insight:")
    print(f"  The mean changed by {change_mean/original_mean*100:.2f}% (AFFECTED by outlier)")
    print(f"  The median changed by {change_median/original_median*100:.2f}% (ROBUST to outlier)")


def confidence_intervals(bootstrap_dist, original_stat, confidence_levels=[90, 95, 99]):
    """
    Compute and print confidence intervals.

    Parameters:
    -----------
    bootstrap_dist : np.ndarray
        Bootstrap distribution
    original_stat : float
        Original statistic
    confidence_levels : list
        Confidence levels (e.g., [90, 95, 99])
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    print("\nUsing percentile method (simplest bootstrap CI):\n")

    for cl in confidence_levels:
        alpha = (100 - cl) / 2
        lower = np.percentile(bootstrap_dist, alpha)
        upper = np.percentile(bootstrap_dist, 100 - alpha)
        width = upper - lower

        print(f"{cl}% CI: [${lower:>10,.0f}, ${upper:>10,.0f}]  Width: ${width:>10,.0f}")

    print(f"\nNote: Higher confidence → wider interval (precision vs confidence trade-off)")


from scipy import stats


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("BOOTSTRAP RESAMPLING: MEDIAN ESTIMATION")
    print("=" * 80)

    # Step 1: Load data
    print("\nStep 1: Loading income data...")
    data = simulate_income_data()
    original_median = np.median(data)
    original_mean = np.mean(data)

    print(f"  Sample size: {len(data):,}")
    print(f"  Original median: ${original_median:,.0f}")
    print(f"  Original mean: ${original_mean:,.0f}")

    # Step 2: Bootstrap
    print("\nStep 2: Running bootstrap (n_bootstrap=1000)...")
    bootstrap_medians = bootstrap_median(data, n_bootstrap=1000)
    bootstrap_means = bootstrap_mean_for_comparison(data, n_bootstrap=1000)

    # Step 3: Compute bootstrap statistics
    print("\nStep 3: Computing bootstrap statistics...")
    compute_bootstrap_statistics(data, bootstrap_medians, original_median, 'median')
    compute_bootstrap_statistics(data, bootstrap_means, original_mean, 'mean')

    # Step 4: Visualize
    print("\nStep 4: Generating visualizations...")
    visualize_bootstrap_distributions(data, bootstrap_medians, bootstrap_means,
                                      original_median, original_mean)

    # Step 5: Robustness comparison
    robustness_comparison(data)

    # Step 6: Confidence intervals
    confidence_intervals(bootstrap_medians, original_median)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. BOOTSTRAP allows SE estimation for ANY statistic without formulas
2. MEDIAN is more ROBUST than mean for skewed/outlier-prone data
3. BOOTSTRAP distribution reveals sampling variability
4. PERCENTILE METHOD provides simple confidence intervals
5. ROBUSTNESS matters in finance: use median for skewed returns
""")


if __name__ == "__main__":
    main()
