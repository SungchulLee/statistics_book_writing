"""
False Discovery Rate and Resampling-Based Multiple Testing
===========================================================
Adapted from ISL (Introduction to Statistical Learning) Chapter 13 Lab.

When testing many hypotheses simultaneously, controlling false
discoveries is critical.  Demonstrates:
1. FWER growth — P(>= 1 false positive) vs number of tests
2. Bonferroni vs Holm corrections for FWER control
3. Benjamini-Hochberg (BH) procedure for FDR control
4. Resampling (permutation) approach to estimate FDR
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── 1. FWER growth ──────────────────────────────────────────
def fwer_growth():
    """P(at least one Type I error) = 1 - (1 - alpha)^m."""
    m_vals = np.arange(1, 501)
    alphas = [0.05, 0.01, 0.001]
    curves = {}
    for a in alphas:
        curves[a] = 1 - (1 - a) ** m_vals
    return m_vals, curves


# ── 2. Simulated multiple testing ────────────────────────────
def simulate_tests(n_tests=2000, n_true_alt=200, n_obs=50,
                   effect_size=0.5):
    """
    Simulate n_tests one-sample t-tests.
    First n_true_alt have a real effect; the rest are null.
    """
    p_values = np.zeros(n_tests)
    truth = np.zeros(n_tests, dtype=int)  # 0 = null, 1 = alternative
    truth[:n_true_alt] = 1

    for i in range(n_tests):
        mu = effect_size if i < n_true_alt else 0.0
        data = np.random.normal(mu, 1.0, n_obs)
        _, p_values[i] = stats.ttest_1samp(data, 0)

    return p_values, truth


# ── 3. Resampling-based FDR estimation ──────────────────────
def resampling_fdr(X_group1, X_group2, n_permutations=500):
    """
    Estimate FDR using permutation-based null distribution.
    X_group1, X_group2: (n_samples, n_features) arrays
    Returns: thresholds, R (rejections), V_hat (estimated false),
             FDR estimates
    """
    n1 = X_group1.shape[0]
    n2 = X_group2.shape[0]
    n_features = X_group1.shape[1]

    # Observed test statistics
    X_combined = np.vstack([X_group1, X_group2])
    t_obs = np.zeros(n_features)
    for j in range(n_features):
        t_obs[j] = stats.ttest_ind(X_group1[:, j], X_group2[:, j]).statistic

    # Permutation null distribution
    t_perm = np.zeros((n_permutations, n_features))
    for b in range(n_permutations):
        perm_idx = np.random.permutation(n1 + n2)
        X_perm1 = X_combined[perm_idx[:n1]]
        X_perm2 = X_combined[perm_idx[n1:]]
        for j in range(n_features):
            t_perm[b, j] = stats.ttest_ind(
                X_perm1[:, j], X_perm2[:, j]).statistic

    # Estimate FDR at each threshold
    abs_t_sorted = np.sort(np.abs(t_obs))
    Rs = []
    Vs = []
    FDRs = []
    for thresh in abs_t_sorted:
        R = np.sum(np.abs(t_obs) >= thresh)
        V = np.sum(np.abs(t_perm) >= thresh) / n_permutations
        fdr = V / max(R, 1)
        Rs.append(R)
        Vs.append(V)
        FDRs.append(fdr)

    return np.array(Rs), np.array(FDRs)


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Multiple Testing: FWER, BH-FDR, and Resampling FDR")
    print("=" * 60)

    # ── Part 1: FWER growth ──────────────────────────────────
    m_vals, fwer_curves = fwer_growth()

    # ── Part 2: Simulated tests with corrections ─────────────
    n_tests = 2000
    n_true_alt = 200
    p_values, truth = simulate_tests(n_tests, n_true_alt, n_obs=50,
                                     effect_size=0.5)

    # Apply corrections
    _, p_bonf, _, _ = multipletests(p_values, method="bonferroni")
    _, p_holm, _, _ = multipletests(p_values, method="holm")
    _, p_bh, _, _ = multipletests(p_values, method="fdr_bh")

    alpha = 0.05
    rej_none = p_values < alpha
    rej_bonf = p_bonf < alpha
    rej_holm = p_holm < alpha
    rej_bh = p_bh < alpha

    def report(name, rejected, truth):
        tp = np.sum(rejected & (truth == 1))
        fp = np.sum(rejected & (truth == 0))
        fn = np.sum(~rejected & (truth == 1))
        total_rej = np.sum(rejected)
        fdr = fp / max(total_rej, 1)
        power = tp / max(np.sum(truth == 1), 1)
        print(f"  {name:15s}: rejected={total_rej:4d}, "
              f"TP={tp:3d}, FP={fp:3d}, "
              f"FDR={fdr:.3f}, Power={power:.3f}")

    print(f"\n  {n_tests} tests, {n_true_alt} truly non-null, "
          f"effect = 0.5, alpha = {alpha}")
    report("Uncorrected", rej_none, truth)
    report("Bonferroni", rej_bonf, truth)
    report("Holm", rej_holm, truth)
    report("BH (FDR)", rej_bh, truth)

    # ── Part 3: Resampling FDR ───────────────────────────────
    print("\n--- Resampling-Based FDR Estimation ---")
    n_features = 100
    n_diff = 30  # first 30 features have a true difference
    n1, n2 = 25, 25
    X1 = np.random.normal(0, 1, (n1, n_features))
    X2 = np.random.normal(0, 1, (n2, n_features))
    X1[:, :n_diff] += 1.0  # add true effect to first n_diff features

    Rs, FDRs = resampling_fdr(X1, X2, n_permutations=300)

    # Find how many rejections at FDR <= 0.1 and 0.2
    fdr_10 = np.max(Rs[FDRs <= 0.1]) if np.any(FDRs <= 0.1) else 0
    fdr_20 = np.max(Rs[FDRs <= 0.2]) if np.any(FDRs <= 0.2) else 0
    print(f"  {n_features} features, {n_diff} truly different")
    print(f"  Rejections at FDR <= 0.1: {fdr_10}")
    print(f"  Rejections at FDR <= 0.2: {fdr_20}")

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: FWER growth
    ax = axes[0, 0]
    for a, curve in fwer_curves.items():
        ax.plot(m_vals, curve, lw=2, label=f"alpha = {a}")
    ax.axhline(0.05, color="gray", linestyle=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of tests (m)")
    ax.set_ylabel("P(at least 1 false positive)")
    ax.set_title("FWER Growth with Number of Tests")
    ax.legend(fontsize=9)

    # Panel 2: p-value histogram
    ax = axes[0, 1]
    ax.hist(p_values[truth == 0], bins=50, alpha=0.5, density=True,
            color="gray", edgecolor="white", label="Null (true H0)")
    ax.hist(p_values[truth == 1], bins=50, alpha=0.5, density=True,
            color="tomato", edgecolor="white", label="Alternative")
    ax.axvline(0.05, color="black", linestyle="--", lw=1.5)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("P-value Distribution (2000 tests)")
    ax.legend(fontsize=9)

    # Panel 3: comparison bar chart
    ax = axes[1, 0]
    methods = ["None", "Bonferroni", "Holm", "BH"]
    rejects = [rej_none, rej_bonf, rej_holm, rej_bh]
    tps = [np.sum(r & (truth == 1)) for r in rejects]
    fps = [np.sum(r & (truth == 0)) for r in rejects]
    x_pos = np.arange(len(methods))
    width = 0.35
    ax.bar(x_pos - width/2, tps, width, label="True Positives",
           color="seagreen", edgecolor="white")
    ax.bar(x_pos + width/2, fps, width, label="False Positives",
           color="tomato", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count")
    ax.set_title("Correction Methods: TP vs FP")
    ax.legend(fontsize=9)

    # Panel 4: resampling FDR curve
    ax = axes[1, 1]
    ax.plot(Rs, FDRs, "k-", lw=2)
    ax.axhline(0.1, color="red", linestyle="--", lw=1.5,
               label="FDR = 0.1")
    ax.axhline(0.2, color="orange", linestyle="--", lw=1.5,
               label="FDR = 0.2")
    ax.set_xlabel("Number of Rejections")
    ax.set_ylabel("Estimated FDR")
    ax.set_title("Resampling-Based FDR Estimation")
    ax.legend(fontsize=9)

    plt.suptitle("Multiple Testing: FWER Control, BH-FDR, "
                 "and Resampling FDR",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("fdr_resampling_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: fdr_resampling_demo.png")


if __name__ == "__main__":
    main()
