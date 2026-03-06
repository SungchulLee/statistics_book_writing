"""
Manual ANOVA Computation with Fisher LSD Post-Hoc
====================================================
Adapted from Basic-Statistics-With-Python Chapter 5 notebook.

Demonstrates one-way ANOVA from scratch:
1. Compute SST, SSE, MST, MSE, F-statistic manually
2. Compare with scipy.stats.f_oneway
3. Fisher LSD (Least Significant Difference) for pairwise comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic height data (3 nationalities) ────────────────
# Inspired by the notebook's Dutch/Japanese/Danish height comparison
DUTCH    = stats.norm.rvs(loc=183, scale=6, size=30)
JAPANESE = stats.norm.rvs(loc=172, scale=5, size=30)
DANISH   = stats.norm.rvs(loc=181, scale=7, size=30)

GROUPS = {"Dutch": DUTCH, "Japanese": JAPANESE, "Danish": DANISH}


# ── Manual one-way ANOVA ───────────────────────────────────
def manual_anova(groups):
    """
    Compute SST (between), SSE (within), MST, MSE, F-stat from scratch.
    groups: dict of {name: array}
    """
    all_data = np.concatenate(list(groups.values()))
    grand_mean = all_data.mean()
    N = len(all_data)
    k = len(groups)

    # SST (Sum of Squares for Treatments / between-group)
    SST = sum(len(g) * (g.mean() - grand_mean) ** 2
              for g in groups.values())
    # SSE (Sum of Squares for Error / within-group)
    SSE = sum(np.sum((g - g.mean()) ** 2) for g in groups.values())
    # total
    SS_total = SST + SSE

    MST = SST / (k - 1)
    MSE = SSE / (N - k)
    F = MST / MSE
    p_value = 1 - stats.f.cdf(F, k - 1, N - k)

    return {
        "SST": SST, "SSE": SSE, "SS_total": SS_total,
        "MST": MST, "MSE": MSE,
        "F": F, "p": p_value,
        "df_between": k - 1, "df_within": N - k,
        "grand_mean": grand_mean,
    }


# ── Fisher LSD post-hoc ────────────────────────────────────
def fisher_lsd(groups, MSE, alpha=0.05):
    """
    LSD = t_{alpha/2} * sqrt(MSE * (1/n_i + 1/n_j))
    If |mean_i - mean_j| > LSD, the pair differs significantly.
    """
    names = list(groups.keys())
    N_total = sum(len(g) for g in groups.values())
    k = len(groups)
    df_within = N_total - k

    results = []
    for (n1, g1), (n2, g2) in combinations(groups.items(), 2):
        t_crit = stats.t.ppf(1 - alpha / 2, df_within)
        lsd_val = t_crit * np.sqrt(MSE * (1/len(groups[n1]) + 1/len(groups[n2])))
        diff = abs(groups[n1].mean() - groups[n2].mean())
        sig = diff > lsd_val
        results.append({
            "pair": f"{n1} vs {n2}",
            "diff": diff, "LSD": lsd_val, "significant": sig,
        })
    return results


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Manual One-Way ANOVA with Fisher LSD")
    print("=" * 60)

    # group summaries
    print("\n--- Group Summaries ---")
    for name, data in GROUPS.items():
        print(f"  {name:>10s}: n={len(data)}, mean={data.mean():.2f}, "
              f"var={data.var(ddof=1):.2f}")

    # manual ANOVA
    res = manual_anova(GROUPS)
    print(f"\n--- Manual ANOVA ---")
    print(f"  Grand mean  = {res['grand_mean']:.2f}")
    print(f"  SST (between) = {res['SST']:.2f}")
    print(f"  SSE (within)  = {res['SSE']:.2f}")
    print(f"  SS Total      = {res['SS_total']:.2f}")
    print(f"  MST = {res['MST']:.2f},  MSE = {res['MSE']:.2f}")
    print(f"  F-stat = {res['F']:.4f},  p-value = {res['p']:.6f}")
    print(f"  df_between = {res['df_between']},  df_within = {res['df_within']}")

    F_crit = stats.f.ppf(0.95, res["df_between"], res["df_within"])
    print(f"  F_crit (alpha=0.05) = {F_crit:.4f}")
    if res["F"] > F_crit:
        print("  => Reject H0: at least one group mean differs.")
    else:
        print("  => Fail to reject H0.")

    # scipy verification
    F_scipy, p_scipy = stats.f_oneway(*GROUPS.values())
    print(f"\n  scipy.f_oneway: F={F_scipy:.4f}, p={p_scipy:.6f}")

    # Fisher LSD
    print(f"\n--- Fisher LSD Post-Hoc (alpha=0.05) ---")
    lsd_results = fisher_lsd(GROUPS, res["MSE"])
    for r in lsd_results:
        mark = "*" if r["significant"] else " "
        print(f"  {mark} {r['pair']:>20s}: |diff|={r['diff']:.2f}, "
              f"LSD={r['LSD']:.2f}  "
              f"{'SIGNIFICANT' if r['significant'] else 'not significant'}")

    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # boxplot
    ax = axes[0]
    bp = ax.boxplot([GROUPS[k] for k in GROUPS],
                    labels=list(GROUPS.keys()), patch_artist=True)
    colors = ["steelblue", "seagreen", "salmon"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Height (cm)")
    ax.set_title("Height by Nationality")

    # F-distribution with test stat
    ax = axes[1]
    x = np.linspace(0, max(res["F"] * 1.3, F_crit * 2), 300)
    y = stats.f.pdf(x, res["df_between"], res["df_within"])
    ax.plot(x, y, "b-", lw=2)
    ax.axvline(F_crit, color="red", linestyle="--",
               label=f"F_crit = {F_crit:.2f}")
    ax.axvline(res["F"], color="green", linestyle="-", lw=2,
               label=f"F_obs = {res['F']:.2f}")
    mask = x >= F_crit
    ax.fill_between(x[mask], y[mask], alpha=0.3, color="red")
    ax.set_title("F-Distribution with Rejection Region")
    ax.set_xlabel("F")
    ax.legend(fontsize=8)

    # LSD comparison
    ax = axes[2]
    pairs = [r["pair"] for r in lsd_results]
    diffs = [r["diff"] for r in lsd_results]
    lsd_vals = [r["LSD"] for r in lsd_results]
    y_pos = range(len(pairs))
    bar_colors = ["green" if r["significant"] else "grey" for r in lsd_results]
    ax.barh(y_pos, diffs, color=bar_colors, alpha=0.7, edgecolor="white",
            label="|Mean diff|")
    for i, lv in enumerate(lsd_vals):
        ax.plot(lv, i, "rx", markersize=12, markeredgewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("Difference / LSD threshold")
    ax.set_title("Fisher LSD Pairwise Comparison")
    ax.legend(["LSD threshold", "|Mean diff|"], fontsize=8)

    plt.tight_layout()
    plt.savefig("anova_manual_computation.png", dpi=150)
    plt.show()
    print("\nFigure saved: anova_manual_computation.png")


if __name__ == "__main__":
    main()
