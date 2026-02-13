#!/usr/bin/env python3
# ======================================================================
# 16_gof_01_manual_equal_expected_3cats.py
# ======================================================================
# Manual chi-square GOF with equal expected counts (k=3) + visualization.
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Observed data and expected mean-based values
observed_counts = np.array([4, 13, 7])
expected_counts = np.ones(3) * observed_counts.mean()
degrees_of_freedom = observed_counts.shape[0] - 1

# Chi-square test statistic and p-value calculation
chi_square_statistic = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
p_value = stats.chi2(degrees_of_freedom).sf(chi_square_statistic)

# Display the statistic and p-value
print(f"Chi-square Statistic = {chi_square_statistic:.4f}")
print(f"p-value = {p_value:.4f}\n")

# Plotting setup
fig, ax = plt.subplots(figsize=(12, 4))

# Chi-square distribution plot up to observed statistic
x_left = np.linspace(0, chi_square_statistic, 100)
y_left = stats.chi2(degrees_of_freedom).pdf(x_left)
ax.plot(x_left, y_left, linewidth=3)

# Fill left area under the curve (non-significant region)
x_fill_left = np.concatenate([[0], x_left, [chi_square_statistic], [0]])
y_fill_left = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill_left, y_fill_left, alpha=0.1)

# Chi-square distribution plot for tail area (significant region)
x_right = np.linspace(chi_square_statistic, 20, 100)
y_right = stats.chi2(degrees_of_freedom).pdf(x_right)
ax.plot(x_right, y_right, linewidth=3)

# Fill right area under the curve (significant region)
x_fill_right = np.concatenate([[chi_square_statistic], x_right, [20], [chi_square_statistic]])
y_fill_right = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, alpha=0.1)

# Annotate p-value with an arrow
annotation_xy = ((12.5 + 15.0) / 2, 0.01)  # Position for the p-value label
annotation_xytext = (16.5, 0.10)            # Position for the text
arrow_properties = dict(width=0.2, headwidth=8)
ax.annotate(f'p-value = {p_value:.02%}', annotation_xy, xytext=annotation_xytext,
            fontsize=15, arrowprops=arrow_properties)

# Customize plot aesthetics
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")

plt.tight_layout()
plt.show()
