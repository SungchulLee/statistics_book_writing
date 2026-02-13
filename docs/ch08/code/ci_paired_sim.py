#!/usr/bin/env python3
# ============================================================
# 11_confidence_interval_05_paired_mean_confidence_interval_simulation.py
# ============================================================
"""
Paired-sample mean CI for μ_D via:
    • 't'        : Student's t on differences (default)
    • 'z_known'  : z with KNOWN σ_D (uses model σ_x, σ_y, ρ)
    • 'z_plugin' : z with sample s_D (large-n plug-in)

Sampling design note (10% condition & CLT)
-----------------------------------------
• Unit is the pair difference D = X − Y. If pairs are sampled without replacement
  from a finite frame of size N_pairs, independence of D_i is approx when n ≤ 0.10·N_pairs.
• For uncertain shape of D, n ≥ 30 pairs supports Normal-based CIs. With small n,
  inspect differences for outliers/heavy tails.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# -----------------------------
# Config
# -----------------------------
rng_seed = None
n_simulations = 100
n = 12

mu_x, mu_y = 0.5, 0.0
sigma_x, sigma_y = 1.0, 1.2
rho = 0.6
alpha = 0.05
method = "t"   # 't' | 'z_known' | 'z_plugin'

def main():
    # RNG
    rng = np.random.default_rng(rng_seed)

    # True Δ and σ_D (for z_known)
    delta_true = mu_x - mu_y
    var_d_true = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
    sigma_d_true = np.sqrt(max(var_d_true, 0.0))  # numeric guard

    # Precompute transforms/criticals
    cov = rho * sigma_x * sigma_y
    Sigma = np.array([[sigma_x**2, cov], [cov, sigma_y**2]], dtype=float)
    L = np.linalg.cholesky(Sigma)

    df = n - 1
    t_star = t.ppf(1 - alpha / 2.0, df=df)
    z_star = norm.ppf(1 - alpha / 2.0)

    lowers = np.empty(n_simulations, dtype=float)
    uppers = np.empty(n_simulations, dtype=float)
    centers = np.empty(n_simulations, dtype=float)

    for i in range(n_simulations):
        # Correlated pairs via Cholesky
        z = rng.standard_normal(size=(2, n))
        xy = (L @ z).T
        x = xy[:, 0] + mu_x
        y = xy[:, 1] + mu_y

        d = x - y
        dbar = d.mean()
        s_d = d.std(ddof=1)

        if method == "t":
            se = s_d / np.sqrt(n)
            crit = t_star
        elif method == "z_known":
            se = sigma_d_true / np.sqrt(n)
            crit = z_star
        elif method == "z_plugin":
            se = s_d / np.sqrt(n)
            crit = z_star
        else:
            raise ValueError("method must be 't', 'z_known', or 'z_plugin'")

        lowers[i] = dbar - crit * se
        uppers[i] = dbar + crit * se
        centers[i] = dbar

    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Paired {method} CIs for μ_D | n={n}, ρ={rho:.2f}, "
        f"CL={int((1-alpha)*100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)",
        fontsize=12
    )
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("μ_D = μ_X − μ_Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()