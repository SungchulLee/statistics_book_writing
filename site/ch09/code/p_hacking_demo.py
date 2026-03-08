"""
P-Hacking Demonstration
=========================
Adapted from ps4ds (Probability and Statistics for Data Science).

P-hacking occurs when researchers selectively report results,
test multiple hypotheses without correction, or choose analysis
methods to obtain statistically significant (p < 0.05) results.

Demonstrates:
1. Running many t-tests under H0 (no real effect) and seeing how
   often we get p < 0.05 (should be ~5%)
2. "Researcher degrees of freedom": choosing which subset, outcome,
   or test yields significance
3. Distribution of p-values under H0 (should be Uniform)
4. How p-hacking inflates the apparent significance rate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def simulate_null_pvalues(n_experiments=10_000, n_per_group=30):
    """
    Run n_experiments two-sample t-tests where both groups are
    drawn from the same distribution (H0 is true).
    Returns the p-values.
    """
    pvals = np.zeros(n_experiments)
    for i in range(n_experiments):
        group_a = np.random.normal(0, 1, n_per_group)
        group_b = np.random.normal(0, 1, n_per_group)
        _, pvals[i] = stats.ttest_ind(group_a, group_b)
    return pvals


def simulate_phacking(n_experiments=1000, n_outcomes=20, n_per_group=30):
    """
    For each experiment, test n_outcomes dependent variables.
    Report the minimum p-value (cherry-picking the "best" result).
    Under H0, P(min p < 0.05) = 1 - (1-0.05)^20 ≈ 64%.
    """
    min_pvals = np.zeros(n_experiments)
    for i in range(n_experiments):
        pvals = []
        for _ in range(n_outcomes):
            a = np.random.normal(0, 1, n_per_group)
            b = np.random.normal(0, 1, n_per_group)
            _, p = stats.ttest_ind(a, b)
            pvals.append(p)
        min_pvals[i] = min(pvals)
    return min_pvals


def simulate_optional_stopping(n_experiments=1000, n_max=200,
                               check_interval=10):
    """
    Simulate optional stopping: keep collecting data and stop
    as soon as p < 0.05 (or reach n_max per group).
    """
    stopped_pvals = []
    stopped_ns = []
    for _ in range(n_experiments):
        a = []
        b = []
        found_sig = False
        for n in range(check_interval, n_max + 1, check_interval):
            a.extend(np.random.normal(0, 1, check_interval).tolist())
            b.extend(np.random.normal(0, 1, check_interval).tolist())
            _, p = stats.ttest_ind(a, b)
            if p < 0.05:
                stopped_pvals.append(p)
                stopped_ns.append(n)
                found_sig = True
                break
        if not found_sig:
            stopped_pvals.append(p)
            stopped_ns.append(n_max)
    return np.array(stopped_pvals), np.array(stopped_ns)


def main():
    print("=" * 60)
    print("P-Hacking Demonstration")
    print("=" * 60)

    n_exp = 10_000

    # 1. Honest testing under H0
    print("\n--- 1. Honest Testing Under H0 ---")
    honest_pvals = simulate_null_pvalues(n_exp)
    false_pos_rate = np.mean(honest_pvals < 0.05)
    print(f"  {n_exp} tests, all under H0 (no real effect)")
    print(f"  False positive rate: {false_pos_rate:.4f} "
          f"(expected: 0.05)")

    # 2. Cherry-picking from 20 outcomes
    print("\n--- 2. Cherry-Picking (20 Outcomes per Experiment) ---")
    n_outcomes = 20
    phack_pvals = simulate_phacking(1000, n_outcomes)
    phack_rate = np.mean(phack_pvals < 0.05)
    theoretical = 1 - (1 - 0.05) ** n_outcomes
    print(f"  Report min p-value from {n_outcomes} tests")
    print(f"  'Significant' rate: {phack_rate:.4f} "
          f"(theoretical: {theoretical:.4f})")

    # 3. Optional stopping
    print("\n--- 3. Optional Stopping ---")
    stop_pvals, stop_ns = simulate_optional_stopping(1000)
    stop_rate = np.mean(stop_pvals < 0.05)
    print(f"  Stop collecting data as soon as p < 0.05")
    print(f"  'Significant' rate: {stop_rate:.4f}")
    print(f"  Median sample size when stopping: {np.median(stop_ns):.0f}")

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: p-value distribution under H0
    ax = axes[0, 0]
    ax.hist(honest_pvals, bins=50, density=True, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", lw=2,
               label="Uniform(0,1)")
    ax.axvline(0.05, color="black", linestyle=":", lw=2,
               label="alpha = 0.05")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("P-values Under H0 (Honest Testing)")
    ax.legend(fontsize=9)

    # Panel 2: cherry-picked p-values
    ax = axes[0, 1]
    ax.hist(phack_pvals, bins=50, density=True, color="tomato",
            edgecolor="white", alpha=0.8)
    ax.axvline(0.05, color="black", linestyle=":", lw=2,
               label="alpha = 0.05")
    ax.set_xlabel("Minimum p-value (from 20 tests)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cherry-Picked P-values\n"
                 f"({phack_rate*100:.1f}% 'significant' vs 5% expected)")
    ax.legend(fontsize=9)

    # Panel 3: optional stopping p-values
    ax = axes[1, 0]
    ax.hist(stop_pvals, bins=50, density=True, color="coral",
            edgecolor="white", alpha=0.8)
    ax.axvline(0.05, color="black", linestyle=":", lw=2,
               label="alpha = 0.05")
    ax.set_xlabel("p-value at stopping")
    ax.set_ylabel("Density")
    ax.set_title(f"Optional Stopping\n"
                 f"({stop_rate*100:.1f}% 'significant')")
    ax.legend(fontsize=9)

    # Panel 4: comparison bar chart
    ax = axes[1, 1]
    methods = ["Honest\nTesting", "Cherry-Pick\n(20 outcomes)",
               "Optional\nStopping"]
    rates = [false_pos_rate, phack_rate, stop_rate]
    colors = ["steelblue", "tomato", "coral"]
    bars = ax.bar(methods, rates, color=colors, edgecolor="white",
                  width=0.5)
    ax.axhline(0.05, color="red", linestyle="--", lw=2,
               label="Nominal alpha = 0.05")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("P-Hacking Inflates False Positives")
    ax.set_ylim(0, max(rates) * 1.3)
    ax.legend(fontsize=9)

    plt.suptitle("P-Hacking: How Researcher Degrees of Freedom "
                 "Inflate False Discoveries",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("p_hacking_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: p_hacking_demo.png")


if __name__ == "__main__":
    main()
