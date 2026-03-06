"""
Best Subset and Stepwise Feature Selection
============================================
Adapted from ISL (Introduction to Statistical Learning) Chapter 6 Lab.

Demonstrates three model selection strategies for linear regression:
1. Best subset selection — evaluate all 2^p subsets (exhaustive)
2. Forward stepwise — start empty, greedily add the best predictor
3. Backward stepwise — start full, greedily remove the worst predictor

Each method selects the best k-variable model for k = 1, ..., p.
We then compare them using:
  - Training RSS
  - Validation-set RSS
  - AIC / BIC criteria
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic data ───────────────────────────────────────────
def generate_data(n=200, p=8):
    """
    y = 3*x1 + 1.5*x2 - 2*x3 + 0.8*x4 + noise
    Features x5..x8 are irrelevant (noise predictors).
    """
    X = np.random.randn(n, p)
    true_beta = np.array([3.0, 1.5, -2.0, 0.8, 0, 0, 0, 0])
    y = X @ true_beta + np.random.normal(0, 2, n)
    names = [f"x{i+1}" for i in range(p)]
    return X, y, names, true_beta


# ── Best subset selection ────────────────────────────────────
def best_subset(X, y, max_k=None):
    """For each k, find the k-variable model with lowest training RSS."""
    n, p = X.shape
    if max_k is None:
        max_k = p
    results = {}
    for k in range(1, max_k + 1):
        best_rss = np.inf
        best_features = None
        for combo in combinations(range(p), k):
            model = LinearRegression().fit(X[:, combo], y)
            rss = np.sum((y - model.predict(X[:, combo])) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_features = combo
        results[k] = {"features": best_features, "rss": best_rss}
    return results


# ── Forward stepwise ─────────────────────────────────────────
def forward_stepwise(X, y):
    """Greedily add the predictor that most reduces RSS."""
    n, p = X.shape
    selected = []
    remaining = list(range(p))
    results = {}
    for k in range(1, p + 1):
        best_rss = np.inf
        best_feature = None
        for f in remaining:
            trial = selected + [f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_feature = f
        selected.append(best_feature)
        remaining.remove(best_feature)
        results[k] = {"features": tuple(selected), "rss": best_rss}
    return results


# ── Backward stepwise ────────────────────────────────────────
def backward_stepwise(X, y):
    """Start with all predictors, greedily remove the least useful."""
    n, p = X.shape
    current = list(range(p))
    results = {}
    # Full model
    model = LinearRegression().fit(X[:, current], y)
    results[p] = {
        "features": tuple(current),
        "rss": np.sum((y - model.predict(X[:, current])) ** 2),
    }
    for k in range(p - 1, 0, -1):
        best_rss = np.inf
        best_remove = None
        for f in current:
            trial = [x for x in current if x != f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_remove = f
        current.remove(best_remove)
        results[k] = {"features": tuple(current), "rss": best_rss}
    return results


# ── Validation-set evaluation ────────────────────────────────
def validation_rss(X_train, y_train, X_test, y_test, features):
    """Compute test RSS for a given feature set."""
    model = LinearRegression().fit(X_train[:, features], y_train)
    pred = model.predict(X_test[:, features])
    return np.sum((y_test - pred) ** 2)


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Best Subset and Stepwise Feature Selection")
    print("=" * 60)

    X, y, names, true_beta = generate_data(n=200, p=8)

    # Split into train / validation
    n_train = 140
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Run all three methods on training data
    best = best_subset(X_train, y_train)
    fwd = forward_stepwise(X_train, y_train)
    bwd = backward_stepwise(X_train, y_train)

    # Validation RSS
    ks = range(1, 9)
    val_best = [validation_rss(X_train, y_train, X_val, y_val,
                               best[k]["features"]) for k in ks]
    val_fwd = [validation_rss(X_train, y_train, X_val, y_val,
                              fwd[k]["features"]) for k in ks]
    val_bwd = [validation_rss(X_train, y_train, X_val, y_val,
                              bwd[k]["features"]) for k in ks]

    # Print results
    print(f"\n  True model: y = {' + '.join(f'{b}*{n}' for b, n in zip(true_beta, names) if b != 0)}")

    print("\n--- Best k-Variable Model (Forward Stepwise) ---")
    for k in ks:
        feats = [names[i] for i in fwd[k]["features"]]
        print(f"  k={k}: {', '.join(feats):40s}  "
              f"train RSS={fwd[k]['rss']:8.1f}  val RSS={val_fwd[k-1]:8.1f}")

    best_k_fwd = int(np.argmin(val_fwd)) + 1
    best_k_bwd = int(np.argmin(val_bwd)) + 1
    best_k_sub = int(np.argmin(val_best)) + 1
    print(f"\n  Optimal k (validation): "
          f"best-subset={best_k_sub}, forward={best_k_fwd}, "
          f"backward={best_k_bwd}")

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: training RSS
    ax = axes[0]
    ax.plot(list(ks), [best[k]["rss"] for k in ks], "ko-",
            label="Best subset", ms=6)
    ax.plot(list(ks), [fwd[k]["rss"] for k in ks], "bs--",
            label="Forward", ms=5)
    ax.plot(list(ks), [bwd[k]["rss"] for k in ks], "r^:",
            label="Backward", ms=5)
    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel("Training RSS")
    ax.set_title("Training RSS vs Model Size")
    ax.legend(fontsize=9)

    # Panel 2: validation RSS
    ax = axes[1]
    ax.plot(list(ks), val_best, "ko-", label="Best subset", ms=6)
    ax.plot(list(ks), val_fwd, "bs--", label="Forward", ms=5)
    ax.plot(list(ks), val_bwd, "r^:", label="Backward", ms=5)
    ax.axvline(best_k_sub, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel("Validation RSS")
    ax.set_title("Validation RSS vs Model Size")
    ax.legend(fontsize=9)

    # Panel 3: selected features heatmap
    ax = axes[2]
    methods = {"Best Subset": best, "Forward": fwd, "Backward": bwd}
    method_names = list(methods.keys())
    optimal_ks = [best_k_sub, best_k_fwd, best_k_bwd]
    matrix = np.zeros((3, 8))
    for i, (mname, res) in enumerate(methods.items()):
        k_opt = optimal_ks[i]
        for f in res[k_opt]["features"]:
            matrix[i, f] = 1
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{m} (k={k})" for m, k in
                        zip(method_names, optimal_ks)], fontsize=9)
    ax.set_title("Selected Features (optimal k)")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, "x" if matrix[i, j] else "",
                    ha="center", va="center", fontsize=12,
                    color="white" if matrix[i, j] else "lightgray")

    plt.tight_layout()
    plt.savefig("subset_stepwise_selection.png", dpi=150)
    plt.show()
    print("\nFigure saved: subset_stepwise_selection.png")


if __name__ == "__main__":
    main()
