#!/usr/bin/env python3
# ======================================
# normal_sf.py  (Survival Function)
# ======================================
# Goal:
#   Plot the survival function (SF) of a Normal(μ, σ²) alongside
#   its CDF to illustrate their complementary relationship.
#
# The survival function is defined as:
#       SF(x) = P(X > x) = 1 − CDF(x)
#
# It is useful whenever we care about exceedance probabilities,
# e.g.  "What is the probability that a loss exceeds a threshold?"

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Parameters
mu = 0        # mean (μ)
sigma = 1     # standard deviation (σ)

# Build the frozen distribution
dist = stats.norm(loc=mu, scale=sigma)

# x-grid
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)

# CDF and SF
cdf = dist.cdf(x)
sf = dist.sf(x)

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(x, cdf, lw=2, label='CDF  P(X ≤ x)')
ax.plot(x, sf,  lw=2, label='SF   P(X > x)')

# Reference lines at x = 0
ax.axvline(0, ls=':', color='gray', alpha=0.6)
ax.axhline(0.5, ls=':', color='gray', alpha=0.6)

# Annotate the complementary relationship
ax.annotate(
    "CDF + SF = 1",
    xy=(1.2, 0.5), fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray')
)

ax.set_xlabel('x')
ax.set_ylabel('Probability')
ax.set_ylim(-0.03, 1.03)
ax.legend(loc='center left', frameon=False)
ax.set_title(f"Normal({mu}, {sigma}) — CDF vs Survival Function", fontsize=13)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
