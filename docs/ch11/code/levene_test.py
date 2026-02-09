"""
Levene's Test for Equality of Variances

Tests homoscedasticity (equal variances) across two or more groups.
Levene's test is more robust to departures from normality than Bartlett's test,
making it the preferred choice in practice.

Usage:
    python levene_test.py --seed 1
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


def main():
    parser = argparse.ArgumentParser(description="Levene's Test")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--size', type=int, default=100, help='sample size')
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dict = load_data(args.size, args.seed)

    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))
    for ax, (label, (x, y)) in zip(axes, data_dict.items()):
        stat, pval = stats.levene(x, y)
        ax.set_title(f"σ_y={label}, F={stat:.2f}\np={pval:.3f}", fontsize=9)
        ax.hist(x, bins=20, density=True, alpha=0.3, label='x (σ=1)')
        ax.hist(y, bins=20, density=True, alpha=0.3, label=f'y (σ={label})')
        ax.legend(fontsize=7)
    plt.suptitle("Levene's Test for Equality of Variances", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
