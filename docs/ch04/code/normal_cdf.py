#!/usr/bin/env python3
# =====================================
# 01_scipy_stats_03_normal_cdf.py
# =====================================
# Goal:
#   Visualize the CDF of a Normal(μ, σ²) and overlay the PDF on the same figure.
#
# Notes:
#   - Left y-axis: CDF (probability P(X ≤ x), range [0,1])
#   - Right y-axis: PDF (density, integrates to 1)
#   - We use a secondary y-axis (twinx) so both scales remain interpretable.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Parameters
mu = 1        # mean (μ)
sigma = 2     # standard deviation (σ)

# x-grid (±3σ covers most of the mass visually)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)

# CDF and PDF
dist = stats.norm(loc=mu, scale=sigma)
y_cdf = dist.cdf(x)
y_pdf = dist.pdf(x)

# Example reference points (μ−σ, μ, μ+σ)
x_values = [mu - sigma, mu, mu + sigma]
cdf_values = dist.cdf(x_values)

# Figure and axes
fig, ax_cdf = plt.subplots(figsize=(12, 3))

# --- CDF on left y-axis ---
cdf_line, = ax_cdf.plot(x, y_cdf, lw=2, label="CDF (P[X ≤ x])")
ax_cdf.set_xlabel("x")
ax_cdf.set_ylabel("P(X ≤ x)  (CDF)")
ax_cdf.set_ylim(-0.02, 1.02)
ax_cdf.grid(True, linestyle=':')

# Vertical reference lines + labels for CDF values
for xv, yv in zip(x_values, cdf_values):
    ax_cdf.axvline(xv, linestyle='--', color='gray', alpha=0.7)
    ax_cdf.text(xv, min(1.0, yv + 0.05), f"P(X≤{xv:.1f})={yv:.3f}",
                ha='center', va='bottom', fontsize=9)

# --- PDF on right y-axis (twin) ---
ax_pdf = ax_cdf.twinx()
pdf_line, = ax_pdf.plot(x, y_pdf, lw=2, color='tab:red', label="PDF (density)")
ax_pdf.set_ylabel("Density  (PDF)", color='tab:red')
ax_pdf.tick_params(axis='y', labelcolor='tab:red')

# Combined legend (from both axes)
lines = [cdf_line, pdf_line]
labels = [l.get_label() for l in lines]
ax_cdf.legend(lines, labels, loc='lower right', frameon=False)

ax_cdf.set_title(f"Normal(μ={mu}, σ={sigma}) — CDF with PDF Overlay")
fig.tight_layout()
plt.show()
