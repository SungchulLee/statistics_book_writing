#!/usr/bin/env python3
# ===============================
# 01_scipy_stats_01_normal_pdf.py
# ===============================
# Goal:
#   Plot the probability density function (PDF) of a Normal(μ, σ²) distribution
#   using SciPy's stats.norm and Matplotlib.
#
# References:
#   - For x ~ Normal(μ, σ²), the PDF is
#       f(x) = (1 / (σ√(2π))) * exp( - (x - μ)² / (2σ²) )
#   - About 99.7% of the mass lies within μ ± 3σ (68–95–99.7 rule).

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Parameters of the Normal distribution
mu = 1        # mean (μ)
sigma = 2     # standard deviation (σ)  [must be > 0 in general]

# Build an evenly spaced grid of x values over [μ - 3σ, μ + 3σ]
# Using ±3σ covers essentially all of the visible density for plotting.
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

# Evaluate the Normal PDF at each x value.
# stats.norm(loc=μ, scale=σ) constructs a Normal(μ, σ²) "frozen" distribution object.
# .pdf(x) returns the density values at the points in x.
y = stats.norm(loc=mu, scale=sigma).pdf(x)

# Create the figure and axes. figsize=(12, 3) gives a wide, short plot.
fig, ax = plt.subplots(figsize=(12, 3))

# Plot the PDF curve.
ax.plot(x, y)

# Render the plot window (or inline figure if using notebooks).
plt.show()