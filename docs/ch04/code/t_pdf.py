#!/usr/bin/env python3
# ===============================
# 01_scipy_stats_06_t_pdf.py
# ===============================
# Goal:
#   Plot the probability density function (PDF) of a Student's t distribution
#   using SciPy, with an optional Normal overlay to highlight heavier tails.
#
# Facts:
#   - If T ~ t_ν (df = ν), then its PDF is heavier-tailed than Normal.
#   - Mean exists for ν > 1 (and equals loc). Variance exists for ν > 2 and is:
#         Var(T) = (ν / (ν - 2)) * scale^2
#   - SciPy parameterization: stats.t(df=ν, loc=μ, scale=σ)
#
# Tips:
#   - Instead of a fixed ±k·σ window, we build the x-range from quantiles
#     via ppf to ensure a sensible domain even for small ν (very heavy tails).

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# -------------------------
# Parameters (edit freely)
# -------------------------
nu    = 5.0     # degrees of freedom ν (> 0)
mu    = 0.0     # location (mean if ν > 1)
sigma = 1.0     # scale (> 0)

# Toggle Normal overlay (same μ, σ) for tail comparison
show_normal_overlay = True

# -------------------------
# Distribution objects
# -------------------------
t_dist = stats.t(df=nu, loc=mu, scale=sigma)
n_dist = stats.norm(loc=mu, scale=sigma)

# -------------------------
# x-grid: quantile-based
#   Avoid endpoints 0 and 1 to prevent infinities in ppf.
#   Use a wide central mass (e.g., from 1e-4 to 1-1e-4).
# -------------------------
p_lo, p_hi = 1e-4, 1 - 1e-4
x_lo = t_dist.ppf(p_lo)
x_hi = t_dist.ppf(p_hi)
x = np.linspace(x_lo, x_hi, 600)

# -------------------------
# Evaluate PDFs
# -------------------------
y_t = t_dist.pdf(x)
y_n = n_dist.pdf(x) if show_normal_overlay else None

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(x, y_t, lw=2, label=f"t PDF (ν={nu:g}, μ={mu:g}, σ={sigma:g})")

if show_normal_overlay:
    ax.plot(x, y_n, lw=1.8, linestyle='--', label="Normal PDF (same μ, σ)")

# Title, labels, grid, legend
ax.set_title("Student's t Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.grid(True, linestyle=":")
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.show()

# -------------------------
# Notes on moments:
#   - If nu <= 1: mean is undefined (SciPy still uses 'loc' as a shift).
#   - If 1 < nu <= 2: mean exists but variance is infinite.
#   - If nu > 2: variance = (nu / (nu - 2)) * sigma^2.
# -------------------------

