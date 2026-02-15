#!/usr/bin/env python3
# =====================================
# 01_scipy_stats_12_lognormal_pdf.py
# =====================================
# Goal:
#   Plot the PDF of the Log-Normal(μ, σ) distribution for several σ values
#   using SciPy and Matplotlib.
#
# References:
#   - If X ~ Normal(μ, σ²) then Y = exp(X) ~ LogNormal(μ, σ²).
#   - PDF:  f(y) = 1/(yσ√(2π)) · exp(-(ln y - μ)² / (2σ²)),  y > 0
#   - The log-normal is widely used to model asset prices, incomes, and
#     other strictly positive, right-skewed quantities.
#   - SciPy parameterisation:
#       stats.lognorm(s=σ, scale=exp(μ))
#     where s is the shape (σ of the underlying normal) and
#     scale = exp(μ) is the median of the distribution.
#
# Source:  Adapted from the continuous-distribution overview in
#          *Introduction to Statistics with Python* (6_distContinuous.ipynb).

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0                              # μ of underlying Normal
sigmas = [0.5, 1.0, 1.5, 2.0]      # σ values to compare
x = np.linspace(0.001, 8, 500)

fig, ax = plt.subplots(figsize=(12, 4))

for sigma in sigmas:
    rv = stats.lognorm(s=sigma, scale=np.exp(mu))
    ax.plot(x, rv.pdf(x), label=rf'$\sigma={sigma}$')

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(r'Log-Normal Distribution – PDF ($\mu=0$, varying $\sigma$)')
ax.legend()
ax.set_ylim(bottom=-0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
