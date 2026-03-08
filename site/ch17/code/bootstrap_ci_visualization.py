#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals: Visualization
==============================================

Demonstrates the bootstrap method for constructing confidence intervals
at different confidence levels (90% vs 95%).

This script:
1. Draws a sample from a population
2. Uses bootstrap to estimate the sampling distribution
3. Constructs CIs at multiple confidence levels
4. Visualizes the trade-off between confidence and precision

Key insight: Higher confidence level → wider interval (less precise)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def simulate_income_data(n=5000, seed=3):
    """
    Create synthetic income data with right-skewed distribution.

    Parameters:
    -----------
    n : int
        Population size
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray : Income values
    """
    np.random.seed(seed)
    income = np.random.exponential(scale=50000, size=n) + 20000
    return income


def draw_sample(population, n_sample=20, seed=3):
    """
    Draw a single sample from the population without replacement.

    Parameters:
    -----------
    population : np.ndarray
        Population of values
    n_sample : int
        Sample size
    seed : int
        Random seed

    Returns:
    --------
    np.ndarray : Sample values
    """
    np.random.seed(seed)
    return resample(population, n_samples=n_sample, replace=False)


def bootstrap_sampling_distribution(sample, n_bootstrap=500):
    """
    Generate bootstrap sampling distribution of the mean.

    Parameters:
    -----------
    sample : np.ndarray
        Original sample
    n_bootstrap : int
        Number of bootstrap resamples

    Returns:
    --------
    np.ndarray : Bootstrap statistics
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(sample)  # with replacement
        bootstrap_means.append(bootstrap_sample.mean())
    return np.array(bootstrap_means)


def compute_confidence_intervals(bootstrap_dist):
    """
    Compute confidence intervals at multiple levels.

    Parameters:
    -----------
    bootstrap_dist : np.ndarray
        Bootstrap sampling distribution

    Returns:
    --------
    dict : CIs at different confidence levels
    """
    ci_90 = np.percentile(bootstrap_dist, [5, 95])
    ci_95 = np.percentile(bootstrap_dist, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_dist, [0.5, 99.5])

    return {
        '90%': ci_90,
        '95%': ci_95,
        '99%': ci_99,
    }


def print_results(original_sample, bootstrap_dist, cis):
    """Print summary of results."""
    print("=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL RESULTS")
    print("=" * 80)
    print()

    print(f"Original Sample (n={len(original_sample)}):")
    print(f"  Mean: ${original_sample.mean():,.0f}")
    print(f"  Std:  ${original_sample.std():,.0f}")
    print()

    print("Bootstrap Distribution (n_bootstrap=500):")
    print(f"  Mean: ${bootstrap_dist.mean():,.0f}")
    print(f"  Std:  ${bootstrap_dist.std():,.0f}")
    print()

    print("Confidence Intervals:")
    for level, (lower, upper) in cis.items():
        width = upper - lower
        print(f"  {level} CI: [${lower:>10,.0f}, ${upper:>10,.0f}]  Width: ${width:,.0f}")
    print()


def visualize_confidence_levels(bootstrap_dist, original_mean, cis):
    """
    Create side-by-side visualizations of 90% and 95% CIs.

    Parameters:
    -----------
    bootstrap_dist : np.ndarray
        Bootstrap sampling distribution
    original_mean : float
        Mean of original sample
    cis : dict
        Dictionary of confidence intervals
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Panel 1: 90% CI =====
    ax1.hist(bootstrap_dist, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    ci_90_lower, ci_90_upper = cis['90%']
    ci_90_mid = (ci_90_lower + ci_90_upper) / 2

    # Plot CI bounds
    ax1.axvline(ci_90_lower, color='darkred', linestyle='--', linewidth=2.5)
    ax1.axvline(ci_90_upper, color='darkred', linestyle='--', linewidth=2.5)

    # Shade CI region
    ax1.axvspan(ci_90_lower, ci_90_upper, alpha=0.2, color='green')

    # Annotate
    ax1.text(ci_90_mid, ax1.get_ylim()[1] * 0.85,
             f'90% CI\n[${ci_90_lower:,.0f}, ${ci_90_upper:,.0f}]',
             ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', linewidth=1.5))

    # Mark original mean
    ax1.axvline(original_mean, color='black', linestyle='-', linewidth=2,
                label=f'Sample mean: ${original_mean:,.0f}')

    ax1.set_xlabel('Bootstrap Sample Mean ($)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('90% Bootstrap Confidence Interval', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Panel 2: 95% CI =====
    ax2.hist(bootstrap_dist, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    ci_95_lower, ci_95_upper = cis['95%']
    ci_95_mid = (ci_95_lower + ci_95_upper) / 2

    # Plot CI bounds
    ax2.axvline(ci_95_lower, color='darkblue', linestyle='--', linewidth=2.5)
    ax2.axvline(ci_95_upper, color='darkblue', linestyle='--', linewidth=2.5)

    # Shade CI region
    ax2.axvspan(ci_95_lower, ci_95_upper, alpha=0.2, color='orange')

    # Annotate
    ax2.text(ci_95_mid, ax2.get_ylim()[1] * 0.85,
             f'95% CI\n[${ci_95_lower:,.0f}, ${ci_95_upper:,.0f}]',
             ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkblue', linewidth=1.5))

    # Mark original mean
    ax2.axvline(original_mean, color='black', linestyle='-', linewidth=2,
                label=f'Sample mean: ${original_mean:,.0f}')

    ax2.set_xlabel('Bootstrap Sample Mean ($)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('95% Bootstrap Confidence Interval', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def simulate_coverage(population, true_mean, n_samples=20, n_bootstrap=500, n_simulations=100):
    """
    Simulate coverage of bootstrap CIs (long-run behavior).

    Parameters:
    -----------
    population : np.ndarray
        Population of values
    true_mean : float
        True population mean
    n_samples : int
        Sample size for each iteration
    n_bootstrap : int
        Number of bootstrap resamples
    n_simulations : int
        Number of simulations

    Returns:
    --------
    float : Coverage percentage
    """
    coverage_90 = []
    coverage_95 = []

    for _ in range(n_simulations):
        # Draw sample
        sample = np.random.choice(population, size=n_samples, replace=False)

        # Bootstrap
        boot_means = np.array([
            np.mean(np.random.choice(sample, size=len(sample), replace=True))
            for _ in range(n_bootstrap)
        ])

        # Compute CIs
        ci_90 = np.percentile(boot_means, [5, 95])
        ci_95 = np.percentile(boot_means, [2.5, 97.5])

        # Check coverage
        coverage_90.append(ci_90[0] <= true_mean <= ci_90[1])
        coverage_95.append(ci_95[0] <= true_mean <= ci_95[1])

    pct_90 = 100 * np.mean(coverage_90)
    pct_95 = 100 * np.mean(coverage_95)

    return pct_90, pct_95


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL VISUALIZATION")
    print("=" * 80 + "\n")

    # Step 1: Create population and draw sample
    population = simulate_income_data()
    true_mean = population.mean()

    sample = draw_sample(population, n_sample=20)
    sample_mean = sample.mean()

    print(f"Population (N={len(population):,}):")
    print(f"  True mean: ${true_mean:,.0f}")
    print()

    print(f"Sample (n={len(sample)}):")
    print(f"  Sample mean: ${sample_mean:,.0f}")
    print()

    # Step 2: Bootstrap
    bootstrap_dist = bootstrap_sampling_distribution(sample, n_bootstrap=500)

    # Step 3: Compute CIs
    cis = compute_confidence_intervals(bootstrap_dist)

    # Step 4: Print results
    print_results(sample, bootstrap_dist, cis)

    # Step 5: Visualize
    print("Generating visualization...")
    visualize_confidence_levels(bootstrap_dist, sample_mean, cis)

    # Step 6 (optional): Coverage simulation
    print("=" * 80)
    print("COVERAGE SIMULATION (Long-run Behavior)")
    print("=" * 80)
    print("\nSimulating 100 repetitions of sampling and bootstrap procedure...")

    pct_90, pct_95 = simulate_coverage(population, true_mean, n_samples=20,
                                       n_bootstrap=500, n_simulations=100)

    print(f"\n90% CI coverage: {pct_90:.1f}% (expected ≈ 90%)")
    print(f"95% CI coverage: {pct_95:.1f}% (expected ≈ 95%)")
    print("\nNote: Actual coverage should be close to nominal level over many repetitions.")
    print("A single interval either contains the true mean or doesn't—no probability!")
    print()


if __name__ == "__main__":
    main()
