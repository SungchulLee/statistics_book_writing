#!/usr/bin/env python3
# ======================================================================
# group_comparison_examples.py
# ======================================================================
# Visualize distributions across groups using boxplots and violin plots.
# Demonstrates comparing categorical and numeric data with real-world
# examples (airline delays).
#
# Techniques shown:
# - Grouped boxplots: Quick summary of medians, quartiles, and outliers
# - Violin plots: Full distributional shape comparison
# - Practical interpretation for decision-making
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def example_airline_delays():
    """
    Example: Airline carrier delays across different airlines.

    Real-world scenario: Flight delays are critical for passenger logistics.
    This analysis compares delay distributions across carriers to identify
    systemic issues and plan mitigation strategies.
    """

    # ── Load or simulate airline data ──
    # In practice: airline_stats = pd.read_csv('airline_stats.csv')
    # For demonstration, create realistic synthetic data

    np.random.seed(42)
    airlines = ['American', 'Delta', 'Southwest', 'United']
    n_obs = 100

    data_list = []
    for airline in airlines:
        # Different airlines have different delay characteristics
        if airline == 'American':
            delays = np.random.gamma(shape=2, scale=3, size=n_obs)  # More frequent delays
        elif airline == 'Delta':
            delays = np.random.gamma(shape=1.5, scale=2.5, size=n_obs)  # Moderate
        elif airline == 'Southwest':
            delays = np.random.gamma(shape=1.2, scale=2, size=n_obs)  # Fewer delays
        else:  # United
            delays = np.random.gamma(shape=1.8, scale=3.2, size=n_obs)  # High variability

        for delay in delays:
            data_list.append({'airline': airline, 'pct_carrier_delay': delay})

    airline_stats = pd.DataFrame(data_list)

    # ── Plot 1: Grouped Boxplots ──
    fig, ax = plt.subplots(figsize=(8, 5))
    airline_stats.boxplot(by='airline', column='pct_carrier_delay', ax=ax)
    ax.set_xlabel('Airline')
    ax.set_ylabel('Daily % of Delayed Flights')
    ax.set_title('Airline Delay Comparison: Boxplots')
    plt.suptitle('')  # Remove the automatic title
    plt.tight_layout()
    plt.show()

    # Interpretation guide for the boxplot
    print("=" * 60)
    print("BOXPLOT INTERPRETATION")
    print("=" * 60)
    print("\nKey Elements:")
    print("  - Box spans Q1 to Q3 (middle 50% of delays)")
    print("  - Line inside box is the median (50th percentile)")
    print("  - Whiskers extend to ~1.5×IQR from box edges")
    print("  - Points beyond whiskers are outliers")
    print()
    print("What this tells us:")
    for airline in airlines:
        subset = airline_stats[airline_stats['airline'] == airline]['pct_carrier_delay']
        print(f"\n{airline}:")
        print(f"  Median:  {subset.median():.2f}%")
        print(f"  Q1-Q3:   {subset.quantile(0.25):.2f}% - {subset.quantile(0.75):.2f}%")
        print(f"  Max:     {subset.max():.2f}%")

    # ── Plot 2: Violin Plots (with quartile info) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=airline_stats, x='airline', y='pct_carrier_delay',
                   ax=ax, inner='quartile', color='lightblue')
    ax.set_xlabel('Airline')
    ax.set_ylabel('Daily % of Delayed Flights')
    ax.set_title('Airline Delay Comparison: Violin Plots\n(Shape shows full distribution)')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("VIOLIN PLOT INTERPRETATION")
    print("=" * 60)
    print("\nWhat the shape tells us:")
    print("  - Wide sections: Many flights with those delay percentages")
    print("  - Narrow sections: Few flights with those delays")
    print("  - Bimodal (two humps): Two typical delay scenarios")
    print("  - Skewed shape: Asymmetric delay distribution")
    print("\nCompare the width (bulges) across airlines to see which")
    print("have different delay characteristics.")

    # ── Plot 3: Side-by-side comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    airline_stats.boxplot(by='airline', column='pct_carrier_delay', ax=ax1)
    ax1.set_xlabel('Airline')
    ax1.set_ylabel('Daily % of Delayed Flights')
    ax1.set_title('Boxplot View')
    ax1.get_figure().suptitle('')

    # Violin plot
    sns.violinplot(data=airline_stats, x='airline', y='pct_carrier_delay',
                   ax=ax2, inner='box')
    ax2.set_xlabel('Airline')
    ax2.set_ylabel('Daily % of Delayed Flights')
    ax2.set_title('Violin Plot View')

    plt.tight_layout()
    plt.show()

    return airline_stats


def example_housing_values():
    """
    Example: Housing values across zip codes (King County, WA).

    Real-world scenario: Real estate investors want to understand how
    neighborhood (zip code) affects home values.
    """

    # Simulate housing data with different price distributions by zip code
    np.random.seed(123)
    zip_codes = [98188, 98105, 98108, 98126]
    n_homes = 150

    data_list = []
    for zip_code in zip_codes:
        # Different zip codes have different price characteristics
        base_price = 300_000 if zip_code in [98105, 98108] else 450_000
        price_std = 100_000
        prices = np.random.normal(base_price, price_std, n_homes)
        prices = np.clip(prices, 50_000, 2_000_000)  # Realistic bounds

        for price in prices:
            data_list.append({'ZipCode': str(zip_code), 'TaxAssessedValue': price})

    housing = pd.DataFrame(data_list)

    # Create boxplots
    fig, ax = plt.subplots(figsize=(8, 5))
    housing.boxplot(by='ZipCode', column='TaxAssessedValue', ax=ax)
    ax.set_xlabel('Zip Code')
    ax.set_ylabel('Tax Assessed Value ($)')
    ax.set_title('Housing Values Across Neighborhoods (King County, WA)')
    plt.suptitle('')
    plt.tight_layout()
    plt.show()

    # Statistical summary
    print("\n" + "=" * 60)
    print("HOUSING VALUE COMPARISON BY ZIP CODE")
    print("=" * 60)
    for zip_code in zip_codes:
        subset = housing[housing['ZipCode'] == str(zip_code)]['TaxAssessedValue']
        print(f"\nZip Code {zip_code}:")
        print(f"  Median:     ${subset.median():>12,.0f}")
        print(f"  Mean:       ${subset.mean():>12,.0f}")
        print(f"  Std Dev:    ${subset.std():>12,.0f}")
        print(f"  Q1–Q3:      ${subset.quantile(0.25):>12,.0f} – ${subset.quantile(0.75):>12,.0f}")

    return housing


def example_income_by_loan_grade():
    """
    Example: Income distribution by loan credit grade.

    Real-world scenario: Lenders assess credit risk by examining income
    distributions across loan grades (A=best, G=worst).
    """

    # Simulate realistic income distribution by credit grade
    np.random.seed(456)
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    n_per_grade = 100

    data_list = []
    for grade in grades:
        # Lower grades tend to have lower, more variable incomes
        grade_idx = ord(grade) - ord('A')
        base_income = 80_000 - grade_idx * 8_000
        income_std = 15_000 + grade_idx * 5_000

        incomes = np.random.normal(base_income, income_std, n_per_grade)
        incomes = np.clip(incomes, 10_000, 200_000)  # Realistic bounds

        for income in incomes:
            data_list.append({'grade': grade, 'income': income})

    loans = pd.DataFrame(data_list)

    # Create violin plot with split visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=loans, x='grade', y='income', ax=ax, color='lightgreen')
    ax.set_xlabel('Loan Grade (A=best, G=worst)')
    ax.set_ylabel('Annual Income ($)')
    ax.set_title('Income Distribution by Credit Grade')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    plt.tight_layout()
    plt.show()

    # Insights
    print("\n" + "=" * 60)
    print("INCOME INSIGHTS BY CREDIT GRADE")
    print("=" * 60)
    print("\nObservations:")
    print("  - Grade A: Higher, more concentrated income")
    print("  - Grade G: Lower, more dispersed income")
    print("  - Violin shape: Shows where incomes cluster")
    print("\nFor lenders:")
    print("  - Grade A: Stable, predictable income → lower risk")
    print("  - Grade G: Variable income → higher default risk")

    return loans


def main():
    """Run all group comparison examples."""

    print("\n" + "=" * 60)
    print("GROUP COMPARISON: BOXPLOTS & VIOLIN PLOTS")
    print("=" * 60)

    # Example 1: Airline delays
    print("\n\n1. AIRLINE CARRIER DELAYS")
    print("-" * 60)
    airline_stats = example_airline_delays()

    # Example 2: Housing values
    print("\n\n2. HOUSING VALUES BY ZIP CODE")
    print("-" * 60)
    housing = example_housing_values()

    # Example 3: Income by loan grade
    print("\n\n3. INCOME BY LOAN CREDIT GRADE")
    print("-" * 60)
    loans = example_income_by_loan_grade()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Both boxplots and violin plots enable group comparison:

BOXPLOTS:
  Pros: Compact, shows outliers, easy to read
  Cons: Hide distributional details, multimodality

VIOLIN PLOTS:
  Pros: Show full distribution shape, reveal multimodality
  Cons: Require more space, less familiar to some audiences

BEST PRACTICE: Use both for complete understanding!
""")


if __name__ == "__main__":
    main()
