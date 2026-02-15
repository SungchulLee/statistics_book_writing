#!/usr/bin/env python3
"""
Sampling Distribution Visualization
====================================

Demonstrates how the sampling distribution concentrates with increasing sample size.
Uses simulated income data to show the Central Limit Theorem in action.

The script creates three panels:
1. Population sample (individual incomes)
2. Sampling distribution with n=5 (mean of 5)
3. Sampling distribution with n=20 (mean of 20)

This visualization clearly shows that as sample size increases, the distribution
of sample means becomes more concentrated and more normally distributed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_income_data(n=10000, seed=1):
    """
    Create synthetic income data with left-skewed distribution
    (similar to real loan income data).

    Parameters:
    -----------
    n : int
        Number of observations
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.Series : Income values
    """
    np.random.seed(seed)
    # Right-skewed exponential + shift creates realistic income distribution
    income = np.random.exponential(scale=50000, size=n) + 20000
    return pd.Series(income)


def create_sampling_distributions(loans_income, n_samples=1000):
    """
    Create three datasets for visualization:
    1. Sample of individual values
    2. Distribution of means (n=5)
    3. Distribution of means (n=20)

    Parameters:
    -----------
    loans_income : pd.Series
        Population of income values
    n_samples : int
        Number of samples to draw for each sampling distribution

    Returns:
    --------
    pd.DataFrame : Combined data with type column
    """
    # 1. Sample 1000 individual incomes
    sample_data = pd.DataFrame({
        'income': loans_income.sample(1000),
        'type': 'Population Sample\n(n=1000)',
    })

    # 2. Distribution of means when n=5
    sample_mean_05 = pd.DataFrame({
        'income': [loans_income.sample(5).mean() for _ in range(n_samples)],
        'type': 'Sampling Distribution\n(Mean of 5)',
    })

    # 3. Distribution of means when n=20
    sample_mean_20 = pd.DataFrame({
        'income': [loans_income.sample(20).mean() for _ in range(n_samples)],
        'type': 'Sampling Distribution\n(Mean of 20)',
    })

    return pd.concat([sample_data, sample_mean_05, sample_mean_20], ignore_index=True)


def compute_summary_statistics(results):
    """Print summary statistics for each distribution type."""
    print("=" * 80)
    print("Summary Statistics by Distribution Type")
    print("=" * 80)

    summary = results.groupby('type')['income'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ])

    # Format for display
    summary['Mean'] = summary['Mean'].apply(lambda x: f"${x:,.0f}")
    summary['Std Dev'] = summary['Std Dev'].apply(lambda x: f"${x:,.0f}")
    summary['Min'] = summary['Min'].apply(lambda x: f"${x:,.0f}")
    summary['Max'] = summary['Max'].apply(lambda x: f"${x:,.0f}")

    print(summary)
    print()


def visualize_distributions(results):
    """
    Create side-by-side histograms of the three distributions.

    Parameters:
    -----------
    results : pd.DataFrame
        Combined data from create_sampling_distributions()
    """
    g = sns.FacetGrid(
        results,
        col='type',
        col_wrap=1,
        height=2.5,
        aspect=2.5
    )

    g.map(
        plt.hist,
        'income',
        bins=40,
        range=[0, 200000],
        color='steelblue',
        edgecolor='black',
        alpha=0.8
    )

    g.set_axis_labels('Income ($)', 'Frequency')
    g.set_titles('{col_name}')

    # Clean up aesthetics
    for ax in g.axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.tight_layout()
    plt.show()


def verify_standard_error(loans_income):
    """
    Verify the theoretical relationship SE = σ / sqrt(n).

    Parameters:
    -----------
    loans_income : pd.Series
        Population of income values
    """
    print("=" * 80)
    print("Standard Error Verification")
    print("=" * 80)

    pop_std = loans_income.std()
    se_5 = pop_std / np.sqrt(5)
    se_20 = pop_std / np.sqrt(20)

    print(f"Population standard deviation: ${pop_std:,.0f}")
    print(f"Theoretical SE (n=5):          ${se_5:,.0f}")
    print(f"Theoretical SE (n=20):         ${se_20:,.0f}")
    print(f"Ratio SE(5)/SE(20):            {se_5/se_20:.2f}")
    print()
    print("Note: To reduce SE by half, increase sample size by a factor of 4")
    print(f"  4 × 5 = 20, so SE(20) ≈ SE(5) / 2")
    print()


def empirical_standard_error(loans_income):
    """
    Compute empirical standard errors from bootstrap sampling distributions.

    Parameters:
    -----------
    loans_income : pd.Series
        Population of income values
    """
    print("=" * 80)
    print("Empirical Standard Errors (from sampling distributions)")
    print("=" * 80)

    sample_means_5 = np.array([
        np.mean(np.random.choice(loans_income, 5))
        for _ in range(1000)
    ])

    sample_means_20 = np.array([
        np.mean(np.random.choice(loans_income, 20))
        for _ in range(1000)
    ])

    se_5_empirical = sample_means_5.std()
    se_20_empirical = sample_means_20.std()

    print(f"Empirical SE (n=5):  ${se_5_empirical:,.0f}")
    print(f"Empirical SE (n=20): ${se_20_empirical:,.0f}")
    print()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("SAMPLING DISTRIBUTION VISUALIZATION")
    print("=" * 80 + "\n")

    # Step 1: Create income data
    loans_income = simulate_income_data()
    print(f"Created synthetic income data with {len(loans_income):,} observations")
    print(f"  Mean:    ${loans_income.mean():,.0f}")
    print(f"  Std Dev: ${loans_income.std():,.0f}")
    print()

    # Step 2: Create sampling distributions
    results = create_sampling_distributions(loans_income)

    # Step 3: Display summary statistics
    compute_summary_statistics(results)

    # Step 4: Verify standard error formula
    verify_standard_error(loans_income)

    # Step 5: Compute empirical standard errors
    empirical_standard_error(loans_income)

    # Step 6: Visualize
    print("Generating visualization...")
    visualize_distributions(results)


if __name__ == "__main__":
    main()
