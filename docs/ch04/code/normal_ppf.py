#!/usr/bin/env python3
# =======================================
# normal_ppf.py  (Percent Point Function)
# =======================================
# Goal:
#   Visualize the PPF (quantile function / inverse CDF) of a
#   Normal(μ, σ²) distribution.
#
# The PPF answers:  "What value x satisfies P(X ≤ x) = q?"
#   ppf(q) = CDF⁻¹(q)
#
# Example:
#   For N(0,1), ppf(0.975) ≈ 1.96  →  the famous 97.5th percentile
#   used in two-sided 95 % confidence intervals.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Parameters
mu = 0        # mean (μ)
sigma = 1     # standard deviation (σ)
prob = 0.975  # cumulative probability whose quantile we highlight

# Build the frozen distribution object
dist = stats.norm(loc=mu, scale=sigma)

# x-grid for plotting the PDF
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1_000)
pdf = dist.pdf(x)

# Compute the quantile (PPF value)
z = dist.ppf(prob)

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))

# PDF curve
ax.plot(x, pdf, color='b', lw=2, label='PDF')

# Vertical line at the quantile
ax.plot([z, z], [0, dist.pdf(z)], color='k', lw=3)

# Shaded area = P(X ≤ z) = prob
ax.fill_between(
    x[x <= z], pdf[x <= z], 0,
    interpolate=True, color='r', alpha=0.25,
    label=f"P(X ≤ {z:.2f}) = {prob}"
)

# Annotate
ax.text(z + 0.05, dist.pdf(z) / 2,
        f"ppf({prob}) = {z:.4f}",
        fontsize=11, va='center')

# Axis styling
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_xlabel('x')
ax.legend(loc='upper left', frameon=False)
ax.set_title(f"Normal({mu}, {sigma}) — PPF (Quantile Function)", fontsize=13)

plt.tight_layout()
plt.show()
