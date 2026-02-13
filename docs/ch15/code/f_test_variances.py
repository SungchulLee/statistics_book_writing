"""
F-Test of Equality of Variances

Tests the null hypothesis that two normal populations have the same variance
using the ratio of sample variances.

Test statistic: F = S₁² / S₂²  ~  F(n₁-1, n₂-1)  under H₀: σ₁² = σ₂²

Note: This test is very sensitive to the assumption of normality.
      Levene's test is generally preferred in practice.

Usage:
    python f_test_equality_of_variances.py --seed 1
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


def f_test(data_0, data_1):
    """
    Two-sample F-test for equality of variances.

    H₀: σ₁² = σ₂²
    H₁: σ₁² ≠ σ₂²

    Parameters
    ----------
    data_0 : array-like
        First sample.
    data_1 : array-like
        Second sample.

    Returns
    -------
    statistic : float
        F test statistic (ratio of sample variances).
    p_value : float
        Two-sided p-value.
    """
    statistic = data_0.var(ddof=1) / data_1.var(ddof=1)
    df1 = data_0.shape[0] - 1
    df2 = data_1.shape[0] - 1
    p_value = 2 * min(
        stats.f(df1, df2).cdf(statistic),
        stats.f(df1, df2).sf(statistic)
    )
    return statistic, p_value


def main():
    parser = argparse.ArgumentParser(description='F-Test of Equality of Variances')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--size', type=int, default=100, help='sample size')
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dict = load_data(args.size, args.seed)

    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))
    for ax, (label, (x, y)) in zip(axes, data_dict.items()):
        stat, pval = f_test(x, y)
        ax.set_title(f"σ_y={label}, F={stat:.2f}\np={pval:.3f}", fontsize=9)
        ax.hist(x, bins=20, density=True, alpha=0.3, label='x (σ=1)')
        ax.hist(y, bins=20, density=True, alpha=0.3, label=f'y (σ={label})')
        ax.legend(fontsize=7)
    plt.suptitle("F-Test of Equality of Variances", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
