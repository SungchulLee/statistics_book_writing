"""
Resampling with Shoe Sales — Permutation Test, Effect Size, Bootstrap CI
=========================================================================
Adapted from intro2stats "Resampling" notebook.

An e-commerce company optimised shoe prices.  Weekly sales for 12 weeks
before and after the optimisation are compared using:

1. Permutation test  — is the difference in means statistically significant?
2. Effect size        — how large is the practical impact?
3. Bootstrap CI       — 90 % confidence interval for the mean difference.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

# ── Data ────────────────────────────────────────────────────
BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
AFTER  = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])

NUM_PERMUTATIONS = 100_000
NUM_BOOTSTRAP    = 100_000


# ── 1. Permutation test ────────────────────────────────────
def permutation_test(before, after, n_perm=NUM_PERMUTATIONS):
    """
    Randomly reassign 'before' / 'after' labels and compute the
    fraction of permutations whose mean difference is at least as
    large as the observed difference.
    """
    observed_diff = after.mean() - before.mean()
    combined = np.concatenate([before, after])
    n_before = len(before)
    count = 0

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(combined)
        perm_diffs[i] = combined[n_before:].mean() - combined[:n_before].mean()
        if perm_diffs[i] >= observed_diff:
            count += 1

    p_value = count / n_perm
    return observed_diff, perm_diffs, p_value


# ── 2. Effect size (% change) ──────────────────────────────
def effect_size(before, after):
    """Return the absolute difference and percentage change."""
    diff = after.mean() - before.mean()
    pct  = diff / before.mean() * 100
    return diff, pct


# ── 3. Bootstrap confidence interval ───────────────────────
def bootstrap_ci(before, after, n_boot=NUM_BOOTSTRAP, ci=90):
    """
    Resample *with replacement* from each group independently,
    compute the mean difference each time, then report percentile CI.
    """
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        b = np.random.choice(before, size=len(before), replace=True)
        a = np.random.choice(after,  size=len(after),  replace=True)
        diffs[i] = a.mean() - b.mean()

    lo = (100 - ci) / 2
    hi = 100 - lo
    return diffs, np.percentile(diffs, [lo, hi])


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Shoe Sales Resampling Analysis")
    print("=" * 60)

    # descriptives
    print(f"\nBefore optimisation  mean: {BEFORE.mean():.2f}")
    print(f"After  optimisation  mean: {AFTER.mean():.2f}")

    # permutation test
    obs_diff, perm_diffs, p_val = permutation_test(BEFORE, AFTER)
    print(f"\n--- Permutation Test ({NUM_PERMUTATIONS:,} permutations) ---")
    print(f"Observed difference: {obs_diff:.2f}")
    print(f"p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("=> Reject H0: the increase is statistically significant.")
    else:
        print("=> Fail to reject H0: no significant difference.")

    # effect size
    diff, pct = effect_size(BEFORE, AFTER)
    print(f"\n--- Effect Size ---")
    print(f"Mean difference: {diff:.2f} units")
    print(f"Percentage increase: {pct:.2f} %")

    # bootstrap CI
    boot_diffs, ci_bounds = bootstrap_ci(BEFORE, AFTER, ci=90)
    print(f"\n--- Bootstrap 90 % CI ---")
    print(f"[{ci_bounds[0]:.2f}, {ci_bounds[1]:.2f}]")

    _, ci95 = bootstrap_ci(BEFORE, AFTER, ci=95)
    print(f"\n--- Bootstrap 95 % CI ---")
    print(f"[{ci95[0]:.2f}, {ci95[1]:.2f}]")

    # visualisation
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # permutation distribution
    ax = axes[0]
    ax.hist(perm_diffs, bins=40, edgecolor="white", alpha=0.7)
    ax.axvline(obs_diff, color="red", linestyle="--", linewidth=2,
               label=f"Observed diff = {obs_diff:.2f}")
    ax.set_xlabel("Permuted mean difference")
    ax.set_ylabel("Frequency")
    ax.set_title("Permutation Test Distribution")
    ax.legend()

    # bootstrap distribution
    ax = axes[1]
    ax.hist(boot_diffs, bins=40, edgecolor="white", alpha=0.7, color="seagreen")
    ax.axvline(ci_bounds[0], color="orange", linestyle="--",
               label=f"90 % CI [{ci_bounds[0]:.1f}, {ci_bounds[1]:.1f}]")
    ax.axvline(ci_bounds[1], color="orange", linestyle="--")
    ax.set_xlabel("Bootstrap mean difference")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution of Mean Difference")
    ax.legend()

    plt.tight_layout()
    plt.savefig("resampling_shoe_sales.png", dpi=150)
    plt.show()
    print("\nFigure saved: resampling_shoe_sales.png")


if __name__ == "__main__":
    main()
