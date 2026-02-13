"""
Chi-Squared Test for Variance

Tests whether the variance of a population has a pre-determined value
using the chi-squared distribution.

Test statistic: T = (n-1)*S² / σ₀²  ~  χ²(n-1)

Usage:
    python chi2_test_for_variance.py --seed 0
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def load_data(size, seed):
    """Generate paired samples with varying variance ratios."""
    data_dict = {}
    x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

    for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
        y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
        data_dict[f"{scale:.2f}"] = (x, y)

    return data_dict


def chi2_test_for_variance(data, sigma2_0=1.0):
    """
    One-sample chi-squared test for variance.

    H₀: σ² = σ₀²
    H₁: σ² ≠ σ₀²

    Parameters
    ----------
    data : array-like
        Sample data.
    sigma2_0 : float
        Hypothesized population variance.

    Returns
    -------
    statistic : float
        Chi-squared test statistic.
    p_value : float
        Two-sided p-value.
    """
    n = len(data)
    s2 = np.var(data, ddof=1)
    statistic = (n - 1) * s2 / sigma2_0
    p_value = 2 * min(
        stats.chi2(df=n - 1).cdf(statistic),
        stats.chi2(df=n - 1).sf(statistic)
    )
    return statistic, p_value


def main():
    parser = argparse.ArgumentParser(description='Chi-Squared Test for Variance')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--size', type=int, default=100, help='sample size')
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dict = load_data(args.size, args.seed)

    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))
    for ax, (label, (x, y)) in zip(axes, data_dict.items()):
        _, pval = chi2_test_for_variance(y, sigma2_0=1.0)
        ax.set_title(f"σ={label}, p={pval:.3f}", fontsize=10)
        ax.hist(x, bins=20, density=True, alpha=0.3, label='x (σ=1)')
        ax.hist(y, bins=20, density=True, alpha=0.3, label=f'y (σ={label})')
        ax.legend(fontsize=7)
    plt.suptitle("Chi-Squared Test for Variance (H₀: σ² = 1)", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
