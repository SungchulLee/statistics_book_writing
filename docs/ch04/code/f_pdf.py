#!/usr/bin/env python3
# ===============================
# 01_scipy_stats_08_f_pdf.py
# ===============================
# Goal:
#   Plot the probability density function (PDF) of an F distribution using SciPy,
#   with (optional) reference lines for mean and mode, and optional shading of a
#   central probability mass.
#
# Facts (F ~ F_{d1, d2}):
#   - Support: x ≥ 0
#   - Mean:    E[X] = d2 / (d2 - 2)             (exists if d2 > 2)
#   - Var:     Var[X] = 2 d2^2 (d1 + d2 - 2)
#                       -----------------------  (exists if d2 > 4)
#                       d1 (d2 - 2)^2 (d2 - 4)
#   - Mode:    ((d1 - 2)/d1) * (d2 / (d2 + 2))  (valid if d1 > 2; otherwise 0)
#
# SciPy parameterization:
#   stats.f(dfn=d1, dfd=d2, loc=0, scale=1)
#
# Tip:
#   Use quantile-based x-limits (ppf) to adapt the visible range automatically.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# -------------------------
# Parameters (edit freely)
# -------------------------
d1 = 5.0                 # numerator degrees of freedom (> 0)
d2 = 12.0                # denominator degrees of freedom (> 0)
show_refs = True         # draw mean/mode reference lines when defined
shade_central_mass = False
central_mass = 0.90      # e.g., 90% central probability region
use_logx = False         # set True to use log-scale on x for skewed shapes

# -------------------------
# Distribution object
# -------------------------
f_dist = stats.f(dfn=d1, dfd=d2)

# x-grid from quantiles (avoid 0 and 1 endpoints to prevent infinities)
p_lo, p_hi = 1e-6, 1 - 1e-6
x_lo = f_dist.ppf(p_lo)
x_hi = f_dist.ppf(p_hi)
x = np.linspace(x_lo, x_hi, 600)

# PDF values
y = f_dist.pdf(x)

# Mean (if exists) and mode (if d1 > 2)
mean = None
if d2 > 2:
    mean = d2 / (d2 - 2)

mode = None
if d1 > 2:
    mode = ((d1 - 2) / d1) * (d2 / (d2 + 2))
else:
    mode = 0.0  # density peaks at 0 when d1 ≤ 2

# Optional shading of central mass
if shade_central_mass:
    alpha = (1 - central_mass) / 2.0
    a = f_dist.ppf(alpha)
    b = f_dist.ppf(1 - alpha)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(x, y, lw=2, label=f"F PDF (d1={d1:g}, d2={d2:g})")

if shade_central_mass:
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], 0, y[mask], alpha=0.25,
                    label=f"{int(central_mass*100)}% central mass")

if show_refs:
    if mean is not None and np.isfinite(mean):
        ax.axvline(mean, linestyle='--', alpha=0.85, label=f"mean = {mean:.3g}")
    if mode is not None and np.isfinite(mode):
        ax.axvline(mode, linestyle=':',  alpha=0.85, label=f"mode ≈ {mode:.3g}")

ax.set_title("F Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.grid(True, linestyle=":")
ax.legend(frameon=False, loc="best")

if use_logx:
    ax.set_xscale("log")

plt.tight_layout()
plt.show()

# -------------------------
# Notes:
# - For very small d1 or d2, the F PDF can be highly skewed; log-x can help.
# - The variance formula only applies when d2 > 4; it is infinite otherwise.
# -------------------------
