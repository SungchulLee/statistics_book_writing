#!/usr/bin/env python3
# ======================================================================
# A/B Testing with Permutation Tests
# ======================================================================
# Complete examples of permutation tests for A/B testing scenarios:
#   1. Web page stickiness (session duration)
#   2. Conversion rate testing
#   3. Headline click-through rates
#   4. Visualization and statistical comparison
#
# Source: Adapted from "Practical Statistics for Data Scientists"
#         (Chapter 3 — Statistical Experiments and Significance Testing)
# ======================================================================

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(42)
random.seed(42)


# =============================================================================
# 1. WEB PAGE STICKINESS TEST (Session Duration in Seconds)
# =============================================================================

print("=" * 70)
print("1. WEB PAGE STICKINESS TEST: A/B Testing Session Duration")
print("=" * 70)

# Sample session times for two pages
session_data = {
    'Time': [185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
             173, 181, 182, 170, 169, 177, 168, 183, 169, 164],
    'Page': ['Page A'] * 10 + ['Page B'] * 10
}
session_times = pd.DataFrame(session_data)

# Calculate observed difference
mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()
obs_diff = mean_b - mean_a

print(f"\nObserved means:")
print(f"  Page A: {mean_a:.2f} seconds")
print(f"  Page B: {mean_b:.2f} seconds")
print(f"  Difference: {obs_diff:.2f} seconds")


def perm_test_two_sample_means(data_col, group_col, nA, nB, n_perms=1000):
    """
    Permutation test for difference of means between two groups.

    Parameters:
    -----------
    data_col : pd.Series
        Column with numerical values
    group_col : pd.Series
        Column with group labels
    nA : int
        Size of group A
    nB : int
        Size of group B
    n_perms : int
        Number of permutations

    Returns:
    --------
    p_value : float
        Two-sided p-value
    perm_diffs : list
        Permuted test statistics
    """
    obs_diff = data_col[group_col == 'Page A'].mean() - data_col[group_col == 'Page B'].mean()
    pooled = data_col.values.copy()

    perm_diffs = []
    for _ in range(n_perms):
        # Shuffle and partition
        np.random.shuffle(pooled)
        diff = pooled[:nA].mean() - pooled[nA:].mean()
        perm_diffs.append(diff)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return p_value, perm_diffs, obs_diff


nA = (session_times.Page == 'Page A').sum()
nB = (session_times.Page == 'Page B').sum()

p_perm, perm_diffs, _ = perm_test_two_sample_means(
    session_times.Time, session_times.Page, nA, nB, n_perms=1000
)

print(f"\nPermutation test (1,000 permutations):")
print(f"  p-value: {p_perm:.4f}")
print(f"  Conclusion: {'Reject H₀' if p_perm < 0.05 else 'Fail to reject H₀'}")

# Compare with t-test
t_stat, p_ttest = stats.ttest_ind(
    session_times[session_times.Page == 'Page A'].Time,
    session_times[session_times.Page == 'Page B'].Time,
    equal_var=False
)
print(f"\nWelch's t-test (for comparison):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_ttest:.4f}")


# =============================================================================
# 2. CONVERSION RATE A/B TEST (Binary Outcome)
# =============================================================================

print("\n" + "=" * 70)
print("2. CONVERSION RATE A/B TEST: Control vs. Treatment")
print("=" * 70)

# Data: Conversions out of total users
n_control = 23739
conv_control = 200
n_treatment = 22588
conv_treatment = 182

# Observed conversion rates
rate_control = conv_control / n_control
rate_treatment = conv_treatment / n_treatment
obs_diff_rate = rate_treatment - rate_control

print(f"\nObserved conversion rates:")
print(f"  Control: {rate_control:.4f} ({conv_control}/{n_control})")
print(f"  Treatment: {rate_treatment:.4f} ({conv_treatment}/{n_treatment})")
print(f"  Difference: {obs_diff_rate:.4f} ({100*obs_diff_rate:.2f}%)")


def perm_test_proportion(n_control, conv_control, n_treatment, conv_treatment,
                          n_perms=1000):
    """
    Permutation test for difference in conversion rates.

    Parameters:
    -----------
    n_control : int
        Total control users
    conv_control : int
        Control conversions
    n_treatment : int
        Total treatment users
    conv_treatment : int
        Treatment conversions
    n_perms : int
        Number of permutations

    Returns:
    --------
    p_value : float
        Two-sided p-value
    perm_diffs : list
        Permuted differences
    """
    # Create binary response vector
    binary = np.zeros(n_control + n_treatment, dtype=int)
    binary[:conv_control + conv_treatment] = 1

    obs_diff = conv_treatment / n_treatment - conv_control / n_control

    perm_diffs = []
    for _ in range(n_perms):
        np.random.shuffle(binary)
        perm_rate_control = binary[:n_control].mean()
        perm_rate_treatment = binary[n_control:].mean()
        perm_diffs.append(perm_rate_treatment - perm_rate_control)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return p_value, perm_diffs, obs_diff


p_ab, perm_diffs_ab, _ = perm_test_proportion(
    n_control, conv_control, n_treatment, conv_treatment, n_perms=1000
)

print(f"\nPermutation test (1,000 permutations):")
print(f"  p-value: {p_ab:.4f}")
print(f"  Conclusion: {'Reject H₀' if p_ab < 0.05 else 'Fail to reject H₀'}")

# Compare with chi-square test
from scipy.stats import chi2_contingency
contingency = np.array([
    [conv_control, n_control - conv_control],
    [conv_treatment, n_treatment - conv_treatment]
])
chi2, p_chi2, df, expected = chi2_contingency(contingency)
print(f"\nChi-square test (for comparison):")
print(f"  χ² statistic: {chi2:.4f}")
print(f"  p-value: {p_chi2:.4f}")


# =============================================================================
# 3. HEADLINE CLICK RATES: Three Headlines A/B Test
# =============================================================================

print("\n" + "=" * 70)
print("3. HEADLINE CLICK RATES: Testing Three Headlines")
print("=" * 70)

# Data: Clicks for three headlines
clicks_data = pd.DataFrame({
    'Headline': ['Headline A', 'Headline B', 'Headline C'],
    'Clicks': [14, 8, 12],
    'No_Clicks': [986, 992, 988]
})

print("\nObserved data:")
print(clicks_data.to_string(index=False))

# Contingency table
contingency_headlines = np.array([
    [14, 8, 12],
    [986, 992, 988]
])

# Chi-square test
chi2_h, p_chi2_h, df_h, expected_h = chi2_contingency(contingency_headlines)
print(f"\nChi-square test of independence:")
print(f"  χ² statistic: {chi2_h:.4f}")
print(f"  p-value: {p_chi2_h:.4f}")
print(f"  Degrees of freedom: {df_h}")
print(f"  Conclusion: {'Headlines differ significantly' if p_chi2_h < 0.05 else 'No significant difference'}")


# =============================================================================
# 4. VISUALIZATION: Permutation Distributions
# =============================================================================

print("\n" + "=" * 70)
print("4. VISUALIZING PERMUTATION DISTRIBUTIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Page Stickiness
ax = axes[0, 0]
ax.hist(perm_diffs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(obs_diff, color='red', linewidth=2, label=f'Observed = {obs_diff:.2f}')
ax.axvline(-obs_diff, color='red', linewidth=2, linestyle='--')
ax.set_xlabel('Difference in Means (seconds)')
ax.set_ylabel('Frequency')
ax.set_title(f'Page Stickiness Permutation Distribution\np-value = {p_perm:.4f}')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 2: Conversion Rate
ax = axes[0, 1]
ax.hist(100 * np.array(perm_diffs_ab), bins=30, alpha=0.7, color='forestgreen',
        edgecolor='black')
ax.axvline(100 * obs_diff_rate, color='red', linewidth=2,
           label=f'Observed = {100*obs_diff_rate:.2f}%')
ax.set_xlabel('Difference in Conversion Rate (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'A/B Conversion Test Permutation Distribution\np-value = {p_ab:.4f}')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 3: Comparison - Session Duration (Permutation vs. t-test)
ax = axes[1, 0]
ax.hist(perm_diffs, bins=30, alpha=0.6, color='steelblue', label='Permutation', edgecolor='black')
# Overlay theoretical t-distribution
from scipy.stats import t as t_dist
df_t = nA + nB - 2
x_range = np.linspace(min(perm_diffs), max(perm_diffs), 100)
# Scale t-distribution to match histogram
scale = len(perm_diffs) * (perm_diffs[1] - perm_diffs[0])
ax.plot(x_range, scale * t_dist.pdf(x_range / (session_times.Time.std() * np.sqrt(1/nA + 1/nB)), df_t),
        'r-', linewidth=2, label='t-distribution')
ax.axvline(obs_diff, color='black', linewidth=2, label=f'Observed = {obs_diff:.2f}')
ax.set_xlabel('Difference in Means')
ax.set_ylabel('Frequency')
ax.set_title('Permutation vs. Parametric Distributions\n(Page Stickiness)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 4: Sample sizes and power considerations
ax = axes[1, 1]
sample_sizes = np.array([100, 200, 500, 1000, 2000, 5000])
effect_sizes = [0.2, 0.5, 0.8]
colors = ['blue', 'green', 'red']

for d, color in zip(effect_sizes, colors):
    powers = []
    for n in sample_sizes:
        z_alpha = stats.norm.ppf(0.975)  # 0.05 two-sided
        z_beta = stats.norm.ppf(0.80)    # 0.80 power
        # Approximate power for two-sample test
        ncp = d * np.sqrt(n / 2)  # non-centrality parameter
        power = 1 - stats.nct.cdf(stats.t.ppf(0.975, 2*n-2), 2*n-2, ncp)
        powers.append(power)
    ax.plot(sample_sizes, powers, 'o-', linewidth=2, markersize=6,
            label=f'd = {d:.1f}', color=color)

ax.axhline(0.80, color='gray', linestyle='--', alpha=0.5, label='80% Power')
ax.set_xlabel('Sample Size (per group)')
ax.set_ylabel('Statistical Power')
ax.set_title('Power Analysis: Effect Size vs. Sample Size')
ax.set_xscale('log')
ax.set_ylim([0, 1.05])
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('ab_testing_permutation_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'ab_testing_permutation_analysis.png'")
plt.show()


# =============================================================================
# 5. SUMMARY TABLE: Test Comparison
# =============================================================================

print("\n" + "=" * 70)
print("5. SUMMARY: Permutation vs. Parametric Tests")
print("=" * 70)

summary_data = {
    'Test': [
        'Page Stickiness',
        'A/B Conversion',
        'Headline Clicks'
    ],
    'Permutation p-value': [
        f'{p_perm:.4f}',
        f'{p_ab:.4f}',
        'N/A (chi-square used)'
    ],
    'Parametric p-value': [
        f'{p_ttest:.4f}',
        f'{p_chi2:.4f}',
        f'{p_chi2_h:.4f}'
    ],
    'Conclusion': [
        'Agree' if abs(p_perm - p_ttest) < 0.05 else 'Differ',
        'Agree' if abs(p_ab - p_chi2) < 0.05 else 'Differ',
        'N/A'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. Permutation tests provide p-values similar to parametric tests when
   assumptions (normality, equal variances) are met.

2. Permutation tests are more robust to violations of distributional assumptions,
   especially with smaller sample sizes.

3. A/B testing context: Permutation tests naturally align with the concept of
   randomization in controlled experiments.

4. Computational efficiency: Modern computers make 1,000-5,000 permutations
   feasible for sample sizes up to ~10,000.

5. Interpretation: A permutation test p-value directly reflects the proportion
   of random shuffles producing statistics as extreme as observed.
""")
