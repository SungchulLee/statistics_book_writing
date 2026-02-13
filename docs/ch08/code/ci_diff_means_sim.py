#!/usr/bin/env python3
# ==========================================================================
# 11_confidence_interval_06_diff_two_means_confidence_interval_simulation.py
# ==========================================================================
"""
Two-sample mean CI for Δ = μ₁ − μ₂ (independent samples).

Methods
-------
• 'welch'   : Welch's t (unequal variances; Satterthwaite df)  ▶ Default
• 'pooled'  : Pooled-variance t (assumes σ₁² = σ₂²)
• 'z_known' : z-interval with KNOWN σ₁, σ₂  (uses sigma1, sigma2 below)
• 'z_plugin': z-interval with sample s₁, s₂  (large-n Normal critical value)

Rule of thumb
-------------
- Prefer Welch unless you have strong justification for equal variances.
- Use 'z_known' only when population σ's are genuinely known.
- 'z_plugin' is a large-sample approximation: use when n₁,n₂ are big and CLT is credible.

Assumptions & notes
-------------------
- Independent samples; data ~ approximately Normal (or n large for CLT).
- Heavy tails/outliers → consider robust/transform/bootstrap (advanced).
- This simulator helps visualize **coverage** by method under (σ₁, σ₂), (n₁, n₂).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# -----------------------------
# Configuration (edit these)
# -----------------------------
rng_seed = None            # int for reproducibility; None → random each run
n_simulations = 100        # number of repeated experiments

n1, n2 = 12, 10            # sample sizes
mu1, mu2 = 0.0, 0.5        # true means → true Δ = μ₁ − μ₂
sigma1, sigma2 = 1.0, 1.5  # TRUE std devs (used only by 'z_known' for SE)
alpha = 0.05               # CI level = 1 − α
method = "welch"           # 'welch' | 'pooled' | 'z_known' | 'z_plugin'

def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    delta_true = mu1 - mu2

    lowers = np.empty(n_simulations, dtype=float)
    uppers = np.empty(n_simulations, dtype=float)
    centers = np.empty(n_simulations, dtype=float)

    for i in range(n_simulations):
        # 1) Simulate independent samples
        x = np.random.normal(loc=mu1, scale=sigma1, size=n1)
        y = np.random.normal(loc=mu2, scale=sigma2, size=n2)

        # 2) Sample stats
        xbar, ybar = x.mean(), y.mean()
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        diff_hat = xbar - ybar
        centers[i] = diff_hat

        # 3) Choose method: compute SE, df (if t), and critical z/t
        if method == "welch":
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
            df = num / den
            crit = t.ppf(1 - alpha/2.0, df=df)

        elif method == "pooled":
            df = n1 + n2 - 2
            sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
            se = np.sqrt(sp2 * (1.0/n1 + 1.0/n2))
            crit = t.ppf(1 - alpha/2.0, df=df)

        elif method == "z_known":
            # Known-population-variance case
            se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
            crit = norm.ppf(1 - alpha/2.0)

        elif method == "z_plugin":
            # Large-sample Normal approx with sample s1, s2
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            crit = norm.ppf(1 - alpha/2.0)

        else:
            raise ValueError("Unknown method; choose 'welch', 'pooled', 'z_known', or 'z_plugin'.")

        # 4) CI endpoints
        lowers[i] = diff_hat - crit * se
        uppers[i] = diff_hat + crit * se

    # Empirical coverage
    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    # Plot all intervals (misses in red)
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(delta_true, linestyle="--", linewidth=1.5)
    ax.set_title(
        f"{n_simulations} Two-Sample Mean CIs ({method}) | n1={n1}, n2={n2}, "
        f"CL={int((1-alpha)*100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)",
        fontsize=12
    )
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Δ = μ₁ − μ₂")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()