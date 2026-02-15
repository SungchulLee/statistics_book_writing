#!/usr/bin/env python3
# ======================================
# 01_scipy_stats_09_exponential_pdf.py
# ======================================
# Goal:
#   Plot the probability density function (PDF) of the Exponential(λ)
#   distribution for several rate parameters using SciPy and Matplotlib.
#
# References:
#   - For x ~ Exponential(λ),  the PDF is
#       f(x) = λ exp(-λx),   x ≥ 0
#   - SciPy parameterises via *scale* = 1/λ:
#       stats.expon(scale=1/λ)
#   - The mean is 1/λ and the variance is 1/λ².
#
# Source:  Adapted from the continuous-distribution overview in
#          *Introduction to Statistics with Python* (6_distContinuous.ipynb).

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Rate parameters to compare
lambdas = [0.5, 1.0, 2.0]
x = np.linspace(0, 6, 300)

fig, ax = plt.subplots(figsize=(12, 4))

for lam in lambdas:
    # SciPy uses scale = 1/λ for the exponential distribution.
    rv = stats.expon(scale=1 / lam)
    ax.plot(x, rv.pdf(x), label=rf'$\lambda={lam}$')

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Exponential Distribution – PDF')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
