"""
Chi-Square Distribution Visualization

Demonstrates the chi-square distribution through:
1. PDF and CDF plots
2. Sampling from χ²(df) directly vs constructing from Normal(0,1)²

The chi-square distribution is defined as:
    Σᵢ Zᵢ² ~ χ²(d)   where Zᵢ IID N(0,1)

Usage:
    python chi_square_distribution.py --df 5 --seed 1
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_pdf_sampling(df, seed):
    """Plot chi-square PDF against histogram of random samples."""
    data = stats.chi2(df=df).rvs(10_000, random_state=seed)

    _, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.7, label='Samples')
    y = stats.chi2(df=df).pdf(bins)
    ax.plot(bins, y, '--r', lw=3, label='PDF')
    ax.set_title(f"χ²({df}) Sampling Distribution")
    ax.legend()
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_position('zero')
    plt.tight_layout()
    plt.show()


def plot_pdf_cdf(df):
    """Plot chi-square PDF and CDF together."""
    x = np.linspace(0, 30, 100)
    pdf = stats.chi2(df=df).pdf(x)
    cdf = stats.chi2(df=df).cdf(x)

    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, pdf, label="PDF")
    ax.plot(x, cdf, label="CDF")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend()
    ax.set_title(f"PDF and CDF of χ²({df})")
    plt.tight_layout()
    plt.show()


def plot_construction_from_normal(df, seed):
    """
    Compare direct chi-square sampling vs construction from sum of squared normals.
    Demonstrates that Σ Zᵢ² ~ χ²(df) where Zᵢ ~ N(0,1).
    """
    data_direct = stats.chi2(df=df).rvs(10_000, random_state=seed)
    data_from_norm = np.sum(
        stats.norm().rvs(size=(df, 10_000), random_state=seed) ** 2,
        axis=0
    )

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    _, bins, _ = ax0.hist(data_direct, bins=100, density=True, alpha=0.7)
    y = stats.chi2(df=df).pdf(bins)
    ax0.plot(bins, y, '--r', lw=3)
    ax0.set_title(f"Direct χ²({df}) Sampling")

    ax1.hist(data_from_norm, bins=bins, density=True, alpha=0.7)
    ax1.plot(bins, y, '--r', lw=3)
    ax1.set_title(f"Σ Z² Construction (df={df})")

    for ax in (ax0, ax1):
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_position('zero')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Chi-Square Distribution')
    parser.add_argument('--df', type=int, default=5, help='degrees of freedom')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    plot_pdf_sampling(args.df, args.seed)
    plot_pdf_cdf(args.df)
    plot_construction_from_normal(args.df, args.seed)


if __name__ == "__main__":
    main()
