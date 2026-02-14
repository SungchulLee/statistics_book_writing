#!/usr/bin/env python3
# ================================================
# xbar_normal.py
# ================================================
# Goal:
#   Show the population distribution, a single sample, and the
#   sampling distribution of X̄ when the population is Normal.
#
# Key idea:
#   If X ~ N(μ, σ²), then X̄ ~ N(μ, σ²/n) exactly for every n.
#   The sampling distribution is Normal regardless of sample size.

import matplotlib.pyplot as plt
import numpy as np

# ── Parameters ──────────────────────────────────────────────
np.random.seed(1)

sample_size = 5          # size of a single random sample
n_samples = 10_000       # number of samples for the sampling distribution
n_population = 10_000    # size of the simulated population


def plot_distributions():
    """
    Generates a 3-panel plot:
      1. Population distribution (Normal)
      2. A single random sample (scatter)
      3. Sampling distribution of X̄
    """
    # Generate a large population from a standard Normal distribution
    population = np.random.normal(loc=0, scale=1, size=n_population)

    # Draw a single random sample from the population
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # Simulate the sampling distribution of X̄
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # Create a 3-row subplot
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot the population distribution
    ax0.hist(population, bins=100, edgecolor='white')
    ax0.set_title('Population Distribution  N(0, 1)', fontsize=18)

    # Plot a single sample
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution  (n = {sample_size})', fontsize=18)

    # Plot the sampling distribution of X̄
    ax2.hist(sample_means, bins=100, edgecolor='white')
    ax2.set_title(r'Sampling Distribution of $\bar{X}$', fontsize=18)

    # Clean up axes
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_distributions()
