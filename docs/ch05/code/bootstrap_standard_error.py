"""
Bootstrap Standard Error
=========================
Adapted from intro2stats "Distributions" notebook.

Standard error measures how far a sample statistic (e.g. the mean)
is likely to be from the true population value.  Here we estimate it
via bootstrap resampling:

    SE(x̄) ≈ std of bootstrap means

We compare the bootstrap SE against the classical formula
    SE = s / √n
and against a direct squared-error approach.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic price data (inspired by CA weed HighQ Jan 2015) ───
PRICES = np.array([
    245.02, 244.88, 244.76, 244.65, 244.53, 244.42, 244.30,
    244.18, 244.08, 243.97, 243.85, 243.74, 243.63, 243.52,
    243.40, 243.28, 243.17, 243.06, 242.95, 242.83, 242.72,
    242.61, 242.49, 242.38, 242.27, 242.15, 242.04, 241.93,
    241.81, 241.70, 241.59,
])

NUM_BOOTSTRAP = 10_000


# ── Squared-error approach (from the notebook) ─────────────
def bootstrap_se_squared_error(data, n_boot=NUM_BOOTSTRAP):
    """
    For each bootstrap sample, compute (bootstrap_mean - sample_mean)².
    SE ≈ sqrt( mean of those squared errors ).
    """
    actual_mean = data.mean()
    sq_errors = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        sq_errors[i] = (sample.mean() - actual_mean) ** 2
    return np.sqrt(sq_errors.mean()), sq_errors


# ── Standard bootstrap SE ──────────────────────────────────
def bootstrap_se(data, n_boot=NUM_BOOTSTRAP):
    """SE = std of bootstrap means."""
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means[i] = sample.mean()
    return means.std(ddof=1), means


# ── Main ────────────────────────────────────────────────────
def main():
    data = PRICES
    n = len(data)
    print("=" * 60)
    print("Bootstrap Standard Error")
    print("=" * 60)
    print(f"\n  Sample size   : {n}")
    print(f"  Sample mean   : {data.mean():.4f}")
    print(f"  Sample std    : {data.std(ddof=1):.4f}")

    # classical formula
    se_classical = data.std(ddof=1) / np.sqrt(n)
    print(f"\n  Classical SE  : {se_classical:.4f}   (s / sqrt(n))")

    # bootstrap SE
    se_boot, boot_means = bootstrap_se(data)
    print(f"  Bootstrap SE  : {se_boot:.4f}   (std of {NUM_BOOTSTRAP:,} bootstrap means)")

    # squared-error approach
    se_sq, _ = bootstrap_se_squared_error(data)
    print(f"  Squared-err SE: {se_sq:.4f}   (sqrt of mean squared error)")

    # varying sample sizes
    print(f"\n  --- SE vs sample size (bootstrap) ---")
    for sub_n in [5, 10, 15, 20, 25, 31]:
        sub = data[:sub_n]
        se_b, _ = bootstrap_se(sub, n_boot=5000)
        se_c = sub.std(ddof=1) / np.sqrt(sub_n)
        print(f"    n={sub_n:>2}  classical={se_c:.4f}  bootstrap={se_b:.4f}")

    # visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(boot_means, bins=40, edgecolor="white", alpha=0.7)
    ax.axvline(data.mean(), color="red", linestyle="--",
               label=f"Sample mean = {data.mean():.2f}")
    ax.set_xlabel("Bootstrap sample mean")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrap Distribution of x̄  (SE = {se_boot:.3f})")
    ax.legend(fontsize=9)

    ax = axes[1]
    sizes = np.arange(5, n + 1)
    se_vals = [data[:k].std(ddof=1) / np.sqrt(k) for k in sizes]
    ax.plot(sizes, se_vals, marker="o", markersize=4)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("SE (classical)")
    ax.set_title("Standard Error Decreases with n")

    plt.tight_layout()
    plt.savefig("bootstrap_standard_error.png", dpi=150)
    plt.show()
    print("\nFigure saved: bootstrap_standard_error.png")


if __name__ == "__main__":
    main()
