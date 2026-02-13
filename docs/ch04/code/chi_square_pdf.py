#!/usr/bin/env python3
# ==================================
# 01_scipy_stats_07_chi_square_pdf.py
# ==================================
# Goal:
#   Plot the probability density function (PDF) of a Chi-square distribution
#   using SciPy, with optional reference lines for mean and mode.
#
# Facts:
#   - If X ~ χ²_k (degrees of freedom k), then:
#       * support: x ≥ 0
#       * mean:    E[X] = k
#       * var:     Var[X] = 2k
#       * mode:    max(k - 2, 0)   (for k ≥ 2; 0 if k < 2)
#   - SciPy parameterization: stats.chi2(df=k, loc=0, scale=1)
#
# Tip:
#   - Use a quantile-based x-range (ppf) rather than a fixed [0, b] so the plot
#     automatically adapts for very small/large k.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# -------------------------
# Parameters (edit freely)
# -------------------------
k = 5.0                 # degrees of freedom (> 0)
show_refs = True        # show mean/mode reference lines
shade_central_mass = False  # set True to shade central mass between two quantiles

# Central mass to shade (e.g., 90%)
central_mass = 0.90
# -------------------------

# Distribution object
chi2 = stats.chi2(df=k)

# x-grid from low to high quantiles (avoid 0 and 1 endpoints to prevent infinities)
p_lo, p_hi = 1e-6, 1 - 1e-6
x_lo = chi2.ppf(p_lo)
x_hi = chi2.ppf(p_hi)
x = np.linspace(x_lo, x_hi, 600)

# PDF values
y = chi2.pdf(x)

# Mean and mode
mean = k
mode = max(k - 2.0, 0.0)

# Optional shading of central mass
if shade_central_mass:
    alpha = (1 - central_mass) / 2.0
    a = chi2.ppf(alpha)
    b = chi2.ppf(1 - alpha)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(x, y, lw=2, label=f"χ² PDF (k={k:g})")

if shade_central_mass:
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], 0, y[mask], alpha=0.25, label=f"{int(central_mass*100)}% central mass")

if show_refs:
    ax.axvline(mean, linestyle='--', alpha=0.8, label=f"mean = {mean:g}")
    ax.axvline(mode, linestyle=':',  alpha=0.8, label=f"mode = {mode:g}")

ax.set_title("Chi-square Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.grid(True, linestyle=":")
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.show()
