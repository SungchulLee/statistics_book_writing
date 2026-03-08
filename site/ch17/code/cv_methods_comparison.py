"""
Cross-Validation Methods Comparison
=====================================
Adapted from ISL (Introduction to Statistical Learning) Chapter 5 Lab.

Compares three cross-validation strategies for model selection:
1. Validation set approach (single train/test split)
2. Leave-One-Out CV (LOOCV)
3. k-Fold CV (k = 5, 10)

Applied to polynomial regression of increasing degree to select
the optimal model complexity, demonstrating:
- Bias-variance tradeoff in CV estimates
- Computational cost differences
- Stability of the selected degree across methods
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score, LeaveOneOut, KFold,
)
import time

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic data ───────────────────────────────────────────
def generate_data(n=200):
    """y = sin(x) + 0.3*x + noise (true relationship is non-linear)."""
    x = np.random.uniform(-3, 3, n)
    y = np.sin(x) + 0.3 * x + np.random.normal(0, 0.5, n)
    return x.reshape(-1, 1), y


def poly_pipeline(degree):
    """Create a polynomial regression pipeline."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression()),
    ])


# ── Validation set approach ──────────────────────────────────
def validation_set_mse(X, y, degrees, n_splits=10):
    """
    Repeat random train/test splits and average MSE per degree.
    This shows the high variance of a single split.
    """
    n = len(y)
    n_train = int(0.5 * n)
    all_mses = {d: [] for d in degrees}

    for _ in range(n_splits):
        perm = np.random.permutation(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        for d in degrees:
            model = poly_pipeline(d).fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            mse = np.mean((y[test_idx] - pred) ** 2)
            all_mses[d].append(mse)

    return all_mses


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Cross-Validation Methods Comparison")
    print("=" * 60)

    X, y = generate_data(n=200)
    degrees = range(1, 11)

    # 1. Validation set (multiple random splits)
    print("\n--- 1. Validation Set Approach (10 random splits) ---")
    val_mses = validation_set_mse(X, y, degrees, n_splits=10)
    val_means = [np.mean(val_mses[d]) for d in degrees]
    val_stds = [np.std(val_mses[d]) for d in degrees]
    best_val = int(np.argmin(val_means)) + 1
    print(f"  Best degree: {best_val} (MSE = {val_means[best_val-1]:.4f})")

    # 2. LOOCV
    print("\n--- 2. Leave-One-Out CV ---")
    loo = LeaveOneOut()
    loocv_mses = []
    t0 = time.time()
    for d in degrees:
        scores = cross_val_score(poly_pipeline(d), X, y,
                                 cv=loo, scoring="neg_mean_squared_error")
        loocv_mses.append(-scores.mean())
    loo_time = time.time() - t0
    best_loo = int(np.argmin(loocv_mses)) + 1
    print(f"  Best degree: {best_loo} (MSE = {loocv_mses[best_loo-1]:.4f})")
    print(f"  Time: {loo_time:.2f}s")

    # 3. k-Fold CV (k=5 and k=10)
    kfold_results = {}
    for k in [5, 10]:
        print(f"\n--- 3. {k}-Fold CV ---")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        kf_mses = []
        kf_stds = []
        t0 = time.time()
        for d in degrees:
            scores = cross_val_score(poly_pipeline(d), X, y,
                                     cv=kf,
                                     scoring="neg_mean_squared_error")
            kf_mses.append(-scores.mean())
            kf_stds.append(scores.std())
        kf_time = time.time() - t0
        best_kf = int(np.argmin(kf_mses)) + 1
        print(f"  Best degree: {best_kf} "
              f"(MSE = {kf_mses[best_kf-1]:.4f})")
        print(f"  Time: {kf_time:.2f}s")
        kfold_results[k] = {"mses": kf_mses, "stds": kf_stds,
                            "best": best_kf}

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: validation set variability
    ax = axes[0, 0]
    for d in degrees:
        ax.scatter([d] * len(val_mses[d]), val_mses[d],
                   color="gray", s=20, alpha=0.5)
    ax.plot(list(degrees), val_means, "ro-", lw=2, label="Mean MSE")
    ax.fill_between(list(degrees),
                    [m - s for m, s in zip(val_means, val_stds)],
                    [m + s for m, s in zip(val_means, val_stds)],
                    alpha=0.2, color="red")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Test MSE")
    ax.set_title("Validation Set (50/50 split, 10 repeats)")
    ax.legend(fontsize=9)

    # Panel 2: LOOCV
    ax = axes[0, 1]
    ax.plot(list(degrees), loocv_mses, "bo-", lw=2, ms=7)
    ax.axvline(best_loo, color="blue", linestyle=":", alpha=0.5)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("LOOCV MSE")
    ax.set_title(f"LOOCV (best degree = {best_loo})")

    # Panel 3: k-fold comparison
    ax = axes[1, 0]
    ax.plot(list(degrees), loocv_mses, "bo-", lw=1.5, ms=5,
            label="LOOCV", alpha=0.7)
    for k, res in kfold_results.items():
        ax.plot(list(degrees), res["mses"], "s-", lw=1.5, ms=5,
                label=f"{k}-fold CV")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("CV MSE")
    ax.set_title("All CV Methods Compared")
    ax.legend(fontsize=9)

    # Panel 4: true function vs best polynomial
    ax = axes[1, 1]
    x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_true = np.sin(x_plot.ravel()) + 0.3 * x_plot.ravel()
    ax.scatter(X.ravel(), y, s=10, alpha=0.4, color="gray",
               label="Data")
    ax.plot(x_plot, y_true, "k-", lw=2, label="True function")

    for d, color in [(1, "red"), (best_loo, "blue"), (10, "green")]:
        model = poly_pipeline(d).fit(X, y)
        ax.plot(x_plot, model.predict(x_plot), "--", lw=1.5,
                color=color, label=f"Degree {d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted Polynomials")
    ax.legend(fontsize=8)
    ax.set_ylim(y.min() - 1, y.max() + 1)

    plt.suptitle("Cross-Validation Methods for Model Selection",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("cv_methods_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: cv_methods_comparison.png")


if __name__ == "__main__":
    main()
