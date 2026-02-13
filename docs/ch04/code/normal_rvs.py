#!/usr/bin/env python3
# ===============================
# 01_scipy_stats_02_normal_rvs.py
# ===============================
# Goal:
#   Demonstrate both:
#     (1) Theoretical PDF of Normal(μ, σ²)
#     (2) Random samples (rvs) drawn from the same distribution.
#
# Notes:
#   - rvs() = random variates (draws random samples)
#   - pdf() = theoretical probability density function
#   - As sample size increases, histogram of samples approximates the PDF.
#
# Reference:
#   stats.norm(loc=μ, scale=σ) → “frozen” Normal(μ, σ²) distribution object
#   - .pdf(x) → evaluates the density at x
#   - .rvs(size=n) → draws n random samples

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Parameters of the Normal distribution
mu = 1        # mean (μ)
sigma = 2     # standard deviation (σ)

# 1. Create an x grid to evaluate the theoretical PDF
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
y = stats.norm(loc=mu, scale=sigma).pdf(x)

# 2. Draw random samples from the same Normal(μ, σ²) distribution
n_samples = 10_000
samples = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)

# 3. Plot both histogram of random samples and theoretical PDF
fig, ax = plt.subplots(figsize=(12, 3))

# Histogram of simulated samples
#   density=True → normalize histogram so total area = 1 (to compare with PDF)
ax.hist(samples, bins=50, density=True, alpha=0.3, color='C0', label='Random Samples (rvs)')

# Theoretical PDF curve
ax.plot(x, y, 'r-', lw=2, label='Theoretical PDF')

# 4. Add plot labels, legend, and grid
ax.set_title(f"Normal(μ={mu}, σ={sigma}) — PDF vs. Random Samples")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, linestyle=':')

# 5. Show the plot
plt.show()