"""
MLE for Geometric and Poisson Distributions — Parametric vs Nonparametric
==========================================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

Compares parametric maximum-likelihood models with the nonparametric
empirical PMF for two common discrete distributions:
1. Geometric — models consecutive successes before a failure
2. Poisson   — models count of events in a fixed interval

For each, we:
  - Generate synthetic data (train/test split)
  - Compute the MLE parameter
  - Compare test-set fit of parametric vs nonparametric models
  - Visualise the log-likelihood surface
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── 1. Geometric Distribution MLE ────────────────────────────
def geometric_mle_demo(n_train=1000, n_test=1000, p_true=0.12):
    """
    Geometric(p): P(X = k) = (1-p) * p^k for k = 0, 1, 2, ...
    (number of successes before first failure)
    MLE: p_hat = sample_mean / (1 + sample_mean), or equivalently
    for scipy's convention: p_hat = 1 / (1 + mean_of_successes)
    """
    # Generate streaks: number of successes before failure
    train = np.random.geometric(1 - p_true, n_train) - 1  # 0-indexed
    test = np.random.geometric(1 - p_true, n_test) - 1

    # MLE
    p_hat = train.mean() / (1 + train.mean())
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    # Parametric PMF
    pmf_param = (1 - p_hat) * p_hat ** k_vals

    # Empirical PMF (nonparametric)
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # Log-likelihood surface
    theta_grid = np.linspace(0.01, 0.99, 200)
    n_success = train.sum()
    n_fail = len(train)
    ll = n_success * np.log(theta_grid) + n_fail * np.log(1 - theta_grid)

    # Test errors
    err_param = np.sqrt(np.mean((pmf_param[:k_max] - pmf_test[:k_max])**2))
    err_nonparam = np.sqrt(np.mean((pmf_train[:k_max] - pmf_test[:k_max])**2))

    return {
        "k_vals": k_vals, "pmf_param": pmf_param, "pmf_train": pmf_train,
        "pmf_test": pmf_test, "p_true": p_true, "p_hat": p_hat,
        "theta_grid": theta_grid, "ll": ll,
        "err_param": err_param, "err_nonparam": err_nonparam,
    }


# ── 2. Poisson Distribution MLE ─────────────────────────────
def poisson_mle_demo(n_train=200, n_test=200, lam_true=4.5):
    """
    Poisson(lambda): P(X = k) = e^{-lam} * lam^k / k!
    MLE: lambda_hat = sample mean
    """
    train = np.random.poisson(lam_true, n_train)
    test = np.random.poisson(lam_true, n_test)

    lam_hat = train.mean()
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    pmf_param = stats.poisson.pmf(k_vals, lam_hat)
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # Log-likelihood surface
    lam_grid = np.linspace(0.5, 10, 200)
    ll = train.sum() * np.log(lam_grid) - n_train * lam_grid

    err_param = np.sqrt(np.mean((pmf_param[:k_max] - pmf_test[:k_max])**2))
    err_nonparam = np.sqrt(np.mean((pmf_train[:k_max] - pmf_test[:k_max])**2))

    return {
        "k_vals": k_vals, "pmf_param": pmf_param, "pmf_train": pmf_train,
        "pmf_test": pmf_test, "lam_true": lam_true, "lam_hat": lam_hat,
        "lam_grid": lam_grid, "ll": ll,
        "err_param": err_param, "err_nonparam": err_nonparam,
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MLE: Geometric & Poisson — Parametric vs Nonparametric")
    print("=" * 60)

    geo = geometric_mle_demo()
    poi = poisson_mle_demo()

    print("\n--- Geometric Distribution ---")
    print(f"  True p = {geo['p_true']:.3f}")
    print(f"  MLE p_hat = {geo['p_hat']:.4f}")
    print(f"  Test RMSE — parametric: {geo['err_param']:.5f}, "
          f"nonparametric: {geo['err_nonparam']:.5f}")

    print("\n--- Poisson Distribution ---")
    print(f"  True lambda = {poi['lam_true']:.2f}")
    print(f"  MLE lambda_hat = {poi['lam_hat']:.4f}")
    print(f"  Test RMSE — parametric: {poi['err_param']:.5f}, "
          f"nonparametric: {poi['err_nonparam']:.5f}")

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # Row 1: Geometric
    ax = axes[0, 0]
    ax.plot(geo["theta_grid"], geo["ll"], "k-", lw=2)
    ax.axvline(geo["p_hat"], color="red", linestyle="--",
               label=f"MLE = {geo['p_hat']:.3f}")
    ax.axvline(geo["p_true"], color="blue", linestyle=":",
               label=f"True = {geo['p_true']:.3f}")
    ax.set_xlabel("theta")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("Geometric: Log-Likelihood")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(geo["k_vals"], geo["pmf_param"], "o", ms=8, mfc="white",
            mec="black", mew=1.5, label="Parametric (MLE)")
    ax.plot(geo["k_vals"], geo["pmf_train"], "o", ms=4, color="black",
            label="Empirical PMF (train)")
    ax.set_xlabel("k (consecutive successes)")
    ax.set_ylabel("P(X = k)")
    ax.set_title("Geometric: Train Fit")
    ax.legend(fontsize=9)

    ax = axes[0, 2]
    ax.plot(geo["k_vals"], geo["pmf_param"], "o", ms=8, mfc="white",
            mec="black", mew=1.5, label="Parametric (MLE)")
    ax.plot(geo["k_vals"], geo["pmf_test"], "o", ms=4, color="red",
            label="Empirical PMF (test)")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    ax.set_title("Geometric: Test Fit")
    ax.legend(fontsize=9)

    # Row 2: Poisson
    ax = axes[1, 0]
    ax.plot(poi["lam_grid"], poi["ll"], "k-", lw=2)
    ax.axvline(poi["lam_hat"], color="red", linestyle="--",
               label=f"MLE = {poi['lam_hat']:.2f}")
    ax.axvline(poi["lam_true"], color="blue", linestyle=":",
               label=f"True = {poi['lam_true']:.2f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Log-likelihood (kernel)")
    ax.set_title("Poisson: Log-Likelihood")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.bar(poi["k_vals"], poi["pmf_param"], color="white", edgecolor="black",
           lw=1.5, label="Parametric (MLE)")
    ax.plot(poi["k_vals"], poi["pmf_train"], "ko", ms=5,
            label="Empirical PMF (train)")
    ax.set_xlabel("k (event count)")
    ax.set_ylabel("P(X = k)")
    ax.set_title("Poisson: Train Fit")
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.bar(poi["k_vals"], poi["pmf_param"], color="white", edgecolor="black",
           lw=1.5, label="Parametric (MLE)")
    ax.plot(poi["k_vals"], poi["pmf_test"], "ro", ms=5,
            label="Empirical PMF (test)")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    ax.set_title("Poisson: Test Fit")
    ax.legend(fontsize=9)

    plt.suptitle("Maximum Likelihood Estimation: "
                 "Parametric vs Nonparametric Comparison",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("geometric_poisson_mle.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: geometric_poisson_mle.png")


if __name__ == "__main__":
    main()
