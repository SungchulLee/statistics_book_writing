#!/usr/bin/env python3
# ===================================================
# s2_normal.py  (Sampling Distribution of S²)
# ===================================================
# Goal:
#   Simulate the sampling distribution of the sample variance S²
#   from several population shapes (Normal, Exponential, Chi-squared,
#   Uniform) and overlay the theoretical chi-square-based density.
#
# Key idea:
#   If X ~ N(μ, σ²), then (n-1)S²/σ² ~ χ²(n-1).
#   For non-normal populations the chi-square approximation is only
#   approximate, but by CLT it improves as n grows.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ── Parameters ──────────────────────────────────────────────
np.random.seed(1)

n_population = 10_000   # size of the simulated population
n_sample = 100           # size of each sample drawn
n_sim = 1_000            # number of simulated S² values

# Build populations from different distributions
populations = {
    "Normal(0,1)":   stats.norm().rvs(n_population, random_state=1),
    "Exp(1)":        stats.expon().rvs(n_population, random_state=2),
    "χ²(2)":         stats.chi2(df=2).rvs(n_population, random_state=3),
    "Uniform(0,1)":  stats.uniform().rvs(n_population, random_state=4),
}


def main():
    fig, axes = plt.subplots(1, len(populations), figsize=(16, 3.5))

    for ax, (name, population) in zip(axes, populations.items()):
        # Simulate sampling distribution of S²
        s2_sims = np.array([
            np.random.choice(population, size=n_sample, replace=False).var(ddof=1)
            for _ in range(n_sim)
        ])

        # Histogram
        _, bins, _ = ax.hist(s2_sims, density=True, bins=30,
                             alpha=0.5, edgecolor='white',
                             label=r"simulated $S^2$")

        # Theoretical chi-square density scaled to S² units
        #   (n-1)S²/σ² ~ χ²(n-1)  ⟹  S² ~ (σ²/(n-1)) × χ²(n-1)
        df = n_sample - 1
        sigma2 = population.var()                 # population variance
        c = df / sigma2                           # scaling constant
        x_grid = np.linspace(bins[0], bins[-1], 300)
        pdf = stats.chi2(df).pdf(x_grid * c) * c
        ax.plot(x_grid, pdf, '--r', lw=2, alpha=0.7,
                label=r"$\chi^2$-based PDF")

        ax.set_title(name)
        ax.set_xlabel(r"$S^2$")

    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)

    fig.suptitle(
        r"Sampling Distribution of $S^2$  (n = %d, %d simulations)"
        % (n_sample, n_sim),
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
