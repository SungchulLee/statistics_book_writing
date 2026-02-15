#!/usr/bin/env python3
# ======================================================================
# 21_model_selection_01_aic_bic_cv.py
# ======================================================================
# Demonstrate model selection using AIC, BIC, and cross-validation
# for choosing the best subset of predictors in linear regression.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 4 — Regression and Prediction).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)


def aic(n, rss, k):
    """Akaike Information Criterion: AIC = n ln(RSS/n) + 2k."""
    return n * np.log(rss / n) + 2 * k


def bic(n, rss, k):
    """Bayesian Information Criterion: BIC = n ln(RSS/n) + k ln(n)."""
    return n * np.log(rss / n) + k * np.log(n)


def cv_mse(X, y, folds=5):
    """5-fold cross-validated MSE for OLS regression (no sklearn)."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    mses = []
    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        pred = X_va @ beta
        mses.append(np.mean((y_va - pred) ** 2))
    return np.mean(mses)


def main():
    print("Model Selection Comparison")
    print("=" * 55)

    # ── Generate data with 8 predictors, only 3 truly relevant ──
    n, p_total = 200, 8
    X_raw = np.random.randn(n, p_total)
    beta_true = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
    y = X_raw @ beta_true + np.random.randn(n) * 2

    # ── Forward selection by AIC / BIC ──
    remaining = list(range(p_total))
    selected = []
    aic_history, bic_history = [], []

    for step in range(p_total):
        best_score, best_j = np.inf, None
        for j in remaining:
            cols = selected + [j]
            X_cand = np.column_stack([np.ones(n), X_raw[:, cols]])
            beta = np.linalg.lstsq(X_cand, y, rcond=None)[0]
            rss = np.sum((y - X_cand @ beta) ** 2)
            score = aic(n, rss, len(cols) + 1)
            if score < best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)

        X_sel = np.column_stack([np.ones(n), X_raw[:, selected]])
        beta = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        rss = np.sum((y - X_sel @ beta) ** 2)
        k = len(selected) + 1
        aic_history.append(aic(n, rss, k))
        bic_history.append(bic(n, rss, k))

    # ── Cross-validation MSE for each model size ──
    cv_history = []
    for size in range(1, p_total + 1):
        cols = selected[:size]
        X_cv = np.column_stack([np.ones(n), X_raw[:, cols]])
        cv_history.append(cv_mse(X_cv, y))

    # ── Print results ──
    best_aic = int(np.argmin(aic_history)) + 1
    best_bic = int(np.argmin(bic_history)) + 1
    best_cv = int(np.argmin(cv_history)) + 1
    print(f"  Best model size by AIC : {best_aic} predictors")
    print(f"  Best model size by BIC : {best_bic} predictors")
    print(f"  Best model size by CV  : {best_cv} predictors")
    print(f"  True relevant predictors: 3")

    # ── Plot ──
    sizes = np.arange(1, p_total + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(sizes, aic_history, 'o-', label='AIC')
    ax1.plot(sizes, bic_history, 's--', label='BIC')
    ax1.axvline(best_aic, color='C0', alpha=0.3, linestyle=':')
    ax1.axvline(best_bic, color='C1', alpha=0.3, linestyle=':')
    ax1.set_xlabel('Number of predictors')
    ax1.set_ylabel('Information criterion')
    ax1.set_title('AIC and BIC vs Model Size')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(sizes, cv_history, 'D-', color='C2')
    ax2.axvline(best_cv, color='C2', alpha=0.3, linestyle=':')
    ax2.set_xlabel('Number of predictors')
    ax2.set_ylabel('5-fold CV MSE')
    ax2.set_title('Cross-Validation MSE vs Model Size')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
