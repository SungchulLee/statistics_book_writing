#!/usr/bin/env python3
# ===================================
# 01_scipy_stats_10_uniform_pdf.py
# ===================================
# Goal:
#   Plot the probability density function (PDF) of the continuous
#   Uniform(a, b) distribution using SciPy and Matplotlib.
#
# References:
#   - For x ~ Uniform(a, b), the PDF is
#       f(x) = 1 / (b - a),   a ≤ x ≤ b
#   - SciPy parameterises via loc=a and scale=b-a:
#       stats.uniform(loc=a, scale=b-a)
#   - The mean is (a+b)/2 and the variance is (b-a)²/12.
#
# Source:  Adapted from the continuous-distribution overview in
#          *Introduction to Statistics with Python* (6_distContinuous.ipynb).

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Several Uniform distributions to compare
intervals = [(0, 1), (-2, 2), (1, 5)]
x = np.linspace(-3, 6, 500)

fig, ax = plt.subplots(figsize=(12, 4))

for a, b in intervals:
    rv = stats.uniform(loc=a, scale=b - a)
    ax.plot(x, rv.pdf(x), label=f'Uniform({a}, {b})')

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Uniform Distribution – PDF')
ax.legend()
ax.set_ylim(bottom=-0.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
