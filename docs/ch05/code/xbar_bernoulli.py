#!/usr/bin/env python3
# ================================================
# xbar_bernoulli.py  (Sampling Distribution of p̂)
# ================================================
# Goal:
#   Simulate the sampling distribution of the sample proportion p̂
#   from Bernoulli populations with different success probabilities.
#   Overlay the theoretical Normal approximation N(p, p(1-p)/n).
#
# Key idea:
#   By the Central Limit Theorem, for large n,
#       p̂ ≈ N(p, p(1-p)/n)
#   This script visualises how well the normal approximation
#   fits the simulated sampling distribution of p̂ for several p values.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ── Parameters ──────────────────────────────────────────────
np.random.seed(1)

n_population = 10_000   # size of the finite population
n_sample = 100           # size of each sample drawn
n_sim = 1_000            # number of simulated sample proportions

# Different population proportions to compare
p_values = [0.4, 0.5, 0.6, 0.7]


def main():
    # Generate Bernoulli populations and simulate p̂ for each
    fig, axes = plt.subplots(1, len(p_values), figsize=(14, 3.5))

    for ax, p in zip(axes, p_values):
        # Create a finite Bernoulli population
        population = stats.binom(n=1, p=p).rvs(n_population, random_state=1)

        # Simulate sampling distribution of p̂
        p_hat_sims = np.array([
            np.random.choice(population, size=n_sample, replace=False).mean()
            for _ in range(n_sim)
        ])

        # Histogram of simulated p̂ values
        _, bins, _ = ax.hist(p_hat_sims, density=True, bins=15,
                             alpha=0.5, edgecolor='white',
                             label=r"simulated $\hat{p}$")

        # Theoretical Normal approximation: N(p, p(1-p)/n)
        se = np.sqrt(p * (1 - p) / n_sample)
        x_grid = np.linspace(bins[0], bins[-1], 200)
        pdf = stats.norm(loc=p, scale=se).pdf(x_grid)
        ax.plot(x_grid, pdf, '--r', lw=2, alpha=0.7, label="Normal approx.")

        ax.set_title(f"p = {p}")
        ax.set_xlabel(r"$\hat{p}$")

    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)

    fig.suptitle(
        r"Sampling Distribution of $\hat{p}$  (n = %d, %d simulations)"
        % (n_sample, n_sim),
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
