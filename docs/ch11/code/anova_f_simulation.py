"""
ANOVA F-Statistic Simulation
==============================
Adapted from Basic-Statistics-With-Python plot_material.py (anova_plot).

Runs 1000 Monte-Carlo replications of one-way ANOVA under nine
different parameter settings, showing how the F-statistic
distribution and p-value histogram change with:
  - group-mean separation
  - within-group variance
  - sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

N_SIM = 1000


def simulate_f(mu, sigma, sizes, n_sim=N_SIM):
    """
    Generate 3 normal samples with given means, stds, and sizes,
    run one-way ANOVA, and repeat n_sim times.
    Returns arrays of F-statistics and p-values.
    """
    F_vals, p_vals = [], []
    for _ in range(n_sim):
        groups = [stats.norm.rvs(m, s, n)
                  for m, s, n in zip(mu, sigma, sizes)]
        F, p = stats.f_oneway(*groups)
        F_vals.append(F)
        p_vals.append(p)
    return np.array(F_vals), np.array(p_vals)


# ── Nine simulation scenarios ──────────────────────────────
SCENARIOS = [
    # (label, mu_list, sigma_list, size_list)
    ("Large separation, equal var",
     [3, 6, 9], [6, 6, 6], [10, 20, 30]),
    ("Tiny separation, equal var",
     [3, 3.1, 2.9], [6, 6, 6], [10, 20, 30]),
    ("Tiny separation, unequal var",
     [3, 3.1, 2.9], [6, 12, 18], [10, 20, 30]),
    ("Large separation, large var",
     [3, 6, 9], [10, 10, 10], [10, 20, 30]),
    ("Moderate separation, large var, small n",
     [3, 5, 6], [10, 10, 10], [10, 10, 10]),
    ("Moderate separation, large var, large n",
     [3, 5, 6], [10, 10, 10], [5000, 5000, 5000]),
    ("No separation, huge var",
     [3, 3, 3], [100, 100, 100], [10, 10, 10]),
    ("No separation, unequal var",
     [3, 3, 3], [1, 1, 2], [10, 20, 30]),
    ("No separation, small var",
     [3, 3, 3], [1, 1, 2], [10, 20, 30]),
]


def main():
    print("=" * 60)
    print("ANOVA F-Statistic Simulation (9 Scenarios)")
    print("=" * 60)

    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(n_scen, 2, figsize=(14, 4 * n_scen))

    for i, (label, mu, sigma, sizes) in enumerate(SCENARIOS):
        F_vals, p_vals = simulate_f(mu, sigma, sizes)

        df1 = 2  # k - 1
        df2 = sum(sizes) - 3  # N - k
        F_crit = stats.f.ppf(0.95, df1, df2)
        reject_pct = np.mean(F_vals > F_crit) * 100

        # F-statistic histogram
        ax = axes[i, 0]
        ax.hist(F_vals, bins=50, edgecolor="white", alpha=0.7)
        ax.axvline(F_crit, color="red", linestyle="--", linewidth=2,
                   label=f"F_crit = {F_crit:.2f}")
        ax.set_title(f"Sim {i+1}: {label}", fontsize=10)
        ax.set_xlabel("F")
        ax.legend(fontsize=7)

        # p-value histogram
        ax = axes[i, 1]
        ax.hist(p_vals, bins=50, edgecolor="white", alpha=0.7, color="seagreen")
        ax.axvline(0.05, color="red", linestyle="--", linewidth=2,
                   label="alpha = 0.05")
        ax.set_title(f"p-values (reject {reject_pct:.1f}%)", fontsize=10)
        ax.set_xlabel("p-value")
        ax.legend(fontsize=7)

        # console output
        print(f"\n  Sim {i+1}: {label}")
        print(f"    mu={mu}, sigma={sigma}, n={sizes}")
        print(f"    F_crit={F_crit:.2f}, reject%={reject_pct:.1f}%")
        print(f"    median F={np.median(F_vals):.2f}, "
              f"median p={np.median(p_vals):.4f}")

    plt.tight_layout()
    plt.savefig("anova_f_simulation.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: anova_f_simulation.png")


if __name__ == "__main__":
    main()
