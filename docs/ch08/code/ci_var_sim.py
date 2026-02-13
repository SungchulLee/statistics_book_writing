#!/usr/bin/env python3
# ====================================================================
# 11_confidence_interval_03_variance_confidence_interval_simulation.py
# ====================================================================
"""
Variance CI via Chi-square. Inline computation only.

Critical Normality assumption
-----------------------------
• This CI is **exact only for Normal data**. The pivotal result
      (n−1)·S² / σ²  ~  χ²_{n−1}
  holds **iff** the population is Normal. If the population is skewed or heavy‑tailed,
  this relationship breaks and the chi‑square CI can **under‑ or over‑cover**, even for
  large n. (The CLT that helps means does **not** rescue this variance CI.)

When to use
-----------
• Data plausibly come from a **Normal population** (check a histogram/QQ plot; look for
  symmetry and light tails).
• Measurement‑error or process data that are well‑modeled by Normal noise.
• Teaching/demo of exact small‑sample inference under Normality.

When to be cautious / alternatives
----------------------------------
• **Skewed/heavy‑tailed** data or notable **outliers** → chi‑square CI can miscover.
  Consider a **bootstrap CI** for σ or σ² (percentile or BCa), or use a **robust scale**
  estimator (e.g., MAD) with bootstrap for a CI on spread.
• **Transformations** (e.g., log) may Normalize, but then the CI is for the variance
  on the transformed scale (not σ² of the original units).
• For comparing two variances, the F‑interval has the **same Normality requirement**.

Sampling design note (10% condition)
------------------------------------
• The chi-square variance CI assumes **i.i.d. Normal** data. If sampling **without replacement**
  from a finite population of size **N**, the 10% condition (**n ≤ 0.10·N**) justifies treating
  observations as approximately independent. For larger sampling fractions or complex designs,
  model-based or design-based methods are more appropriate (advanced topic).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# -----------------------------
# Configuration
# -----------------------------
rng_seed = None
n_simulations = 100
n_samples = 12
mu = 0.0
sigma = 2.0
alpha = 0.05
report_sigma_not_sigma2 = False  # if True, show CI for σ instead of σ²

def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    true_var = sigma**2
    lowers = np.empty(n_simulations, dtype=float)
    uppers = np.empty(n_simulations, dtype=float)
    centers = np.empty(n_simulations, dtype=float)

    df = n_samples - 1
    # Precompute chi-square quantiles (exact under Normality)
    chi2_lo = chi2(df=df).ppf(alpha/2.0)          # lower-tail quantile
    chi2_hi = chi2(df=df).ppf(1 - alpha/2.0)      # upper-tail quantile

    for i in range(n_simulations):
        # Simulate one Normal sample of size n_samples
        x = np.random.normal(loc=mu, scale=sigma, size=n_samples)

        # Unbiased sample variance
        s2 = x.var(ddof=1)

        # Chi-square CI for σ² (exact only if data are Normal)
        lo = df * s2 / chi2_hi
        hi = df * s2 / chi2_lo
        lowers[i] = lo
        uppers[i] = hi
        centers[i] = s2

    # Coverage against the true variance used in simulation
    covered = (lowers <= true_var) & (true_var <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    # Optional: report on the σ (std dev) scale instead of σ²
    if report_sigma_not_sigma2:
        lowers_plot = np.sqrt(lowers)
        uppers_plot = np.sqrt(uppers)
        centers_plot = np.sqrt(centers)
        true_ref = np.sqrt(true_var)
        x_label = "Standard Deviation (σ)"
        title_prefix = "σ (std dev)"
    else:
        lowers_plot = lowers
        uppers_plot = uppers
        centers_plot = centers
        true_ref = true_var
        x_label = "Variance (σ²)"
        title_prefix = "σ² (variance)"

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers_plot[i], uppers_plot[i]], [i, i], lw=2, color=color)  # CI segment
        ax.plot(centers_plot[i], i, marker="o", ms=3, color=color)            # point estimate

    ax.axvline(true_ref, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Chi-square {title_prefix} CIs  |  n={n_samples}, df={df}, "
        f"CL={int((1-alpha)*100)}%  |  Fail={n_fail}  (Coverage ≈ {coverage_pct:.1f}%)",
        fontsize=12
    )
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()