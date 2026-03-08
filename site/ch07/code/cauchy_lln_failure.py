"""
Cauchy Distribution — When the Law of Large Numbers Fails
==========================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

The Cauchy distribution has no finite mean (E[|X|] = infinity).
As a result, the sample mean of i.i.d. Cauchy samples does NOT
converge — averaging more data does not help.

Contrast with the normal distribution, where the sample mean
converges to the true mean as n grows (by the LLN).

Demonstrates:
1. Sample mean trajectories: Cauchy vs Normal
2. Histogram of sample means for various n (Cauchy stays spread)
3. QQ-plot showing Cauchy's heavy tails
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def sample_mean_trajectories(dist, n_max=10_000, n_tries=20):
    """Compute running sample means for multiple independent runs."""
    trajectories = []
    for _ in range(n_tries):
        if dist == "cauchy":
            data = np.random.standard_cauchy(n_max)
        else:
            data = np.random.standard_normal(n_max)
        running_mean = np.cumsum(data) / np.arange(1, n_max + 1)
        trajectories.append(running_mean)
    return trajectories


def sample_mean_distributions(dist, n_vals, n_reps=10_000):
    """Compute sample mean distributions for various sample sizes."""
    results = {}
    for n in n_vals:
        if dist == "cauchy":
            data = np.random.standard_cauchy((n_reps, n))
        else:
            data = np.random.standard_normal((n_reps, n))
        means = data.mean(axis=1)
        results[n] = means
    return results


def main():
    print("=" * 60)
    print("Cauchy Distribution — LLN Failure")
    print("=" * 60)

    n_max = 10_000
    n_tries = 20

    print("\n  Cauchy: no finite mean → sample mean does NOT converge")
    print("  Normal: E[X] = 0 → sample mean converges to 0")

    # Trajectories
    cauchy_traj = sample_mean_trajectories("cauchy", n_max, n_tries)
    normal_traj = sample_mean_trajectories("normal", n_max, n_tries)

    # Distributions of sample mean for different n
    n_vals = [100, 1000, 10_000]
    cauchy_dists = sample_mean_distributions("cauchy", n_vals)
    normal_dists = sample_mean_distributions("normal", n_vals)

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ns = np.arange(1, n_max + 1)

    # Row 1: trajectories
    ax = axes[0, 0]
    for traj in cauchy_traj:
        clipped = np.clip(traj, -50, 50)
        ax.semilogx(ns, clipped, lw=0.7, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", lw=2, label="E[X] undefined")
    ax.set_xlabel("n")
    ax.set_ylabel("Sample mean")
    ax.set_title("Cauchy: Sample Mean Trajectories")
    ax.set_ylim(-50, 50)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for traj in normal_traj:
        ax.semilogx(ns, traj, lw=0.7, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", lw=2, label="E[X] = 0")
    ax.set_xlabel("n")
    ax.set_ylabel("Sample mean")
    ax.set_title("Normal: Sample Mean Trajectories")
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)

    # Row 1 col 3: QQ plot
    ax = axes[0, 2]
    cauchy_sample = np.random.standard_cauchy(1000)
    stats.probplot(cauchy_sample, dist="norm", plot=ax)
    ax.set_title("Cauchy vs Normal Q-Q Plot")
    ax.get_lines()[0].set_markerfacecolor("coral")
    ax.get_lines()[0].set_markersize(3)

    # Row 2: histograms of sample means for different n
    for col, n in enumerate(n_vals):
        ax = axes[1, col]
        c_means = np.clip(cauchy_dists[n], -20, 20)
        n_means = normal_dists[n]

        ax.hist(c_means, bins=80, density=True, alpha=0.6,
                color="coral", edgecolor="white", label="Cauchy")
        ax.hist(n_means, bins=50, density=True, alpha=0.6,
                color="steelblue", edgecolor="white", label="Normal")
        ax.set_title(f"Sample Mean Distribution (n = {n})")
        ax.set_xlabel("Sample mean value")
        ax.set_ylabel("Density")
        ax.set_xlim(-5, 5)
        ax.legend(fontsize=9)

        # Print std of sample means
        print(f"\n  n = {n}:")
        c_full = cauchy_dists[n]
        print(f"    Cauchy sample-mean IQR: "
              f"{np.percentile(c_full, 75) - np.percentile(c_full, 25):.3f}")
        print(f"    Normal sample-mean std: {n_means.std():.4f} "
              f"(theory: {1/np.sqrt(n):.4f})")

    plt.suptitle("Cauchy vs Normal: The Law of Large Numbers "
                 "Requires Finite Mean",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("cauchy_lln_failure.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: cauchy_lln_failure.png")


if __name__ == "__main__":
    main()
