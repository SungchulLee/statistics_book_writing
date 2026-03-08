"""
Central Limit Theorem — Multi-Distribution Visualization
==========================================================
Adapted from Basic-Statistics-With-Python plot_material.py.

Demonstrates CLT by drawing repeated samples of increasing size
from three non-normal parent distributions (Uniform, Beta, Gamma)
and plotting the sampling distribution of the sample mean.

As n grows, the sampling distribution converges to a bell shape
regardless of the parent distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

N_REPS = 2000  # number of repeated samples


def sample_means(dist_rvs, sample_sizes, n_reps=N_REPS):
    """For each sample size, draw n_reps samples and return array of means."""
    results = {}
    for n in sample_sizes:
        means = np.array([dist_rvs(n).mean() for _ in range(n_reps)])
        results[n] = means
    return results


def main():
    print("=" * 60)
    print("Central Limit Theorem Visualization")
    print("=" * 60)

    sample_sizes = [2, 10, 100]
    distributions = {
        "Uniform(2, 8)": {
            "rvs": lambda n: np.random.uniform(2, 8, n),
            "color": "tomato",
            "pop_x": np.linspace(2, 8, 200),
            "pop_pdf": lambda x: np.ones_like(x) / 6,
        },
        "Beta(6, 2)": {
            "rvs": lambda n: stats.beta.rvs(6, 2, size=n),
            "color": "seagreen",
            "pop_x": np.linspace(0, 1, 200),
            "pop_pdf": lambda x: stats.beta.pdf(x, 6, 2),
        },
        "Gamma(6, 1)": {
            "rvs": lambda n: stats.gamma.rvs(6, size=n),
            "color": "steelblue",
            "pop_x": np.linspace(0, 25, 200),
            "pop_pdf": lambda x: stats.gamma.pdf(x, 6),
        },
    }

    n_dists = len(distributions)
    n_rows = 1 + len(sample_sizes)
    fig, axes = plt.subplots(n_rows, n_dists, figsize=(6 * n_dists, 4 * n_rows))

    for col, (name, d) in enumerate(distributions.items()):
        c = d["color"]

        # row 0: population distribution
        ax = axes[0, col]
        ax.plot(d["pop_x"], d["pop_pdf"](d["pop_x"]), lw=3, color=c)
        ax.fill_between(d["pop_x"], d["pop_pdf"](d["pop_x"]), alpha=0.3, color=c)
        ax.set_title(name, fontsize=14, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Population PDF", fontsize=11)

        # rows 1-3: sampling distributions
        means_dict = sample_means(d["rvs"], sample_sizes)
        for row, n in enumerate(sample_sizes, start=1):
            ax = axes[row, col]
            ax.hist(means_dict[n], bins=30, color=c, alpha=0.5,
                    edgecolor="white", density=True)
            ax.set_title(f"n = {n}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Sampling Dist (n={n})", fontsize=10)

        # print summary
        print(f"\n  {name}:")
        for n in sample_sizes:
            m = means_dict[n]
            print(f"    n={n:>4}  mean(x̄)={m.mean():.4f}  "
                  f"std(x̄)={m.std():.4f}")

    plt.suptitle("Central Limit Theorem: Sampling Distribution of x̄",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("clt_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: clt_visualization.png")


if __name__ == "__main__":
    main()
