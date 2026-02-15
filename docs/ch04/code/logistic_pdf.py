#!/usr/bin/env python3
# ====================================
# 01_scipy_stats_11_logistic_pdf.py
# ====================================
# Goal:
#   Plot the PDF of the Logistic(μ, s) distribution and compare it to a
#   Normal distribution with the same mean and variance.
#
# References:
#   - PDF: f(x) = exp(-(x-μ)/s) / (s · (1 + exp(-(x-μ)/s))²)
#   - Mean = μ,  Variance = s²π²/3.
#   - The logistic distribution has heavier tails than the Normal,
#     making it relevant for modelling fat-tailed financial returns.
#   - SciPy parameterisation: stats.logistic(loc=μ, scale=s).
#
# Source:  Adapted from the continuous-distribution overview in
#          *Introduction to Statistics with Python* (6_distContinuous.ipynb).

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0      # location
s  = 1      # scale

x = np.linspace(-8, 8, 400)

rv_logistic = stats.logistic(loc=mu, scale=s)
# Matching Normal: same mean, same variance → σ² = s²π²/3
sigma = s * np.pi / np.sqrt(3)
rv_normal = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, rv_logistic.pdf(x), label='Logistic(0, 1)')
ax.plot(x, rv_normal.pdf(x), '--', label=rf'Normal(0, {sigma:.2f}²) [same var]')

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Logistic vs Normal Distribution – PDF')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
