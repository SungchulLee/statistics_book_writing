"""
Two-Sample t-Test with Weed Price Data
========================================
Adapted from intro2stats "Hypothesis Testing" notebook.

Tests whether high-quality weed prices in California differ
significantly between Jan 2014 and Jan 2015.

Workflow:
1. Check normality (Shapiro-Wilk) before applying t-test.
2. Compute confidence interval for the mean.
3. Run independent-samples t-test.
4. Run chi-square goodness-of-fit on quality proportions.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic data (inspired by Weed_Price.csv CA HighQ) ────
# Jan 2014: slightly higher prices
CA_JAN2014 = np.array([
    248.75, 248.59, 248.63, 248.37, 248.02, 247.68, 247.36,
    246.85, 246.44, 246.06, 245.81, 245.48, 245.18, 244.87,
    244.55, 244.23, 243.89, 243.60, 243.34, 243.08, 242.85,
    242.64, 242.36, 242.15, 241.88, 241.64, 241.40, 241.14,
    240.91, 240.65, 240.42,
])

# Jan 2015: lower prices
CA_JAN2015 = np.array([
    245.02, 244.88, 244.76, 244.65, 244.53, 244.42, 244.30,
    244.18, 244.08, 243.97, 243.85, 243.74, 243.63, 243.52,
    243.40, 243.28, 243.17, 243.06, 242.95, 242.83, 242.72,
    242.61, 242.49, 242.38, 242.27, 242.15, 242.04, 241.93,
    241.81, 241.70, 241.59,
])


def check_normality(data, label):
    """Run Shapiro-Wilk test and report."""
    stat, p = stats.shapiro(data)
    result = "PASS" if p > 0.05 else "FAIL"
    print(f"  Shapiro-Wilk ({label}): W={stat:.4f}, p={p:.4f}  [{result}]")
    return p > 0.05


def confidence_interval(data, confidence=0.95):
    """CI for the mean assuming normality."""
    n = len(data)
    m = data.mean()
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m - h, m + h


def two_sample_ttest(x, y, equal_var=True):
    """Independent two-sample t-test."""
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
    return t_stat, p_value


def chi_square_gof(observed, expected):
    """Chi-square goodness-of-fit test."""
    stat, p = stats.chisquare(observed, f_exp=expected)
    return stat, p


def main():
    print("=" * 60)
    print("Hypothesis Testing — CA Weed Prices (Jan 2014 vs Jan 2015)")
    print("=" * 60)

    x, y = CA_JAN2014, CA_JAN2015

    # descriptives
    print(f"\n  Jan 2014: mean = {x.mean():.2f},  std = {x.std(ddof=1):.2f},  n = {len(x)}")
    print(f"  Jan 2015: mean = {y.mean():.2f},  std = {y.std(ddof=1):.2f},  n = {len(y)}")
    print(f"  Effect size (diff of means): {x.mean() - y.mean():.2f}")

    # 1. normality check
    print(f"\n--- Step 1: Normality Check ---")
    ok1 = check_normality(x, "Jan 2014")
    ok2 = check_normality(y, "Jan 2015")
    if ok1 and ok2:
        print("  Both samples appear normally distributed; t-test is valid.")

    # 2. confidence interval
    print(f"\n--- Step 2: 95% Confidence Interval for Mean ---")
    ci14 = confidence_interval(x)
    ci15 = confidence_interval(y)
    print(f"  Jan 2014: [{ci14[0]:.2f}, {ci14[1]:.2f}]")
    print(f"  Jan 2015: [{ci15[0]:.2f}, {ci15[1]:.2f}]")

    # 3. two-sample t-test
    print(f"\n--- Step 3: Independent Two-Sample t-Test ---")
    t_stat, p_val = two_sample_ttest(x, y, equal_var=True)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_val:.6f}")
    if p_val < 0.05:
        print("  => Reject H0: prices differ significantly between years.")
    else:
        print("  => Fail to reject H0.")

    # 4. chi-square GOF (quality proportions)
    print(f"\n--- Step 4: Chi-Square Goodness-of-Fit (Quality Proportions) ---")
    # synthetic proportions: HighQ, MedQ, LowQ purchasers
    expected_2014 = np.array([453020, 688699, 271937])
    observed_2015 = np.array([461900, 695432, 267120])
    chi2, p_chi = chi_square_gof(observed_2015, expected_2014)
    print(f"  Expected (2014 proportions): {expected_2014}")
    print(f"  Observed (2015):             {observed_2015}")
    print(f"  Chi-square stat: {chi2:.2f},  p-value: {p_chi:.6f}")
    if p_chi < 0.05:
        print("  => Reject H0: 2015 proportions differ from 2014.")
    else:
        print("  => Fail to reject H0.")

    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # histograms
    ax = axes[0]
    ax.hist(x, bins=12, alpha=0.6, label="Jan 2014", edgecolor="white")
    ax.hist(y, bins=12, alpha=0.6, label="Jan 2015", edgecolor="white")
    ax.axvline(x.mean(), color="C0", linestyle="--")
    ax.axvline(y.mean(), color="C1", linestyle="--")
    ax.set_xlabel("Price ($)")
    ax.set_title("Price Distributions")
    ax.legend()

    # QQ plot for Jan 2014
    ax = axes[1]
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Jan 2014")

    # QQ plot for Jan 2015
    ax = axes[2]
    stats.probplot(y, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Jan 2015")

    plt.tight_layout()
    plt.savefig("ttest_weed_prices.png", dpi=150)
    plt.show()
    print("\nFigure saved: ttest_weed_prices.png")


if __name__ == "__main__":
    main()
