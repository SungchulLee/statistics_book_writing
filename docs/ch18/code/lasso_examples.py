#!/usr/bin/env python3
# ======================================================================
# 18_lasso_01_regularization_path_and_cv.py
# ======================================================================
# Lasso (L1) regression:
#   1. Coordinate-descent Lasso solver (pure NumPy).
#   2. Regularisation path — coefficients vs log(lambda).
#   3. Cross-validated lambda selection.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 4 — Regression and Prediction).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def soft_threshold(rho, lam):
    """Soft-thresholding operator for coordinate descent."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    return 0.0


def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Lasso regression via coordinate descent.

    Parameters
    ----------
    X   : (n, p) design matrix (should be standardised).
    y   : (n,)   response vector.
    lam : float  L1 penalty parameter.

    Returns
    -------
    beta : (p,) coefficient vector.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = y - X @ beta + X[:, j] * beta[j]
            rho_j = X[:, j] @ r_j / n
            beta[j] = soft_threshold(rho_j, lam)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def lasso_path(X, y, lambdas):
    """Compute coefficient path over a grid of lambda values."""
    coefs = []
    for lam in lambdas:
        beta = lasso_cd(X, y, lam)
        coefs.append(beta.copy())
    return np.array(coefs)


def cv_lasso(X, y, lambdas, folds=5):
    """K-fold cross-validated MSE for each lambda."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    cv_mse = np.zeros(len(lambdas))

    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        for i, lam in enumerate(lambdas):
            beta = lasso_cd(X_tr, y_tr, lam)
            pred = X_va @ beta
            cv_mse[i] += np.mean((y_va - pred) ** 2)
    return cv_mse / folds


def main():
    print("Lasso Regression")
    print("=" * 55)

    # ── Generate data: 10 predictors, only 3 truly relevant ──
    n, p = 150, 10
    X_raw = np.random.randn(n, p)
    beta_true = np.array([4.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0])
    y = X_raw @ beta_true + np.random.randn(n) * 2

    # Standardise columns
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std

    # ── 1. Regularisation path ──
    lambdas = np.logspace(1, -2, 60)
    path = lasso_path(X, y, lambdas)

    # ── 2. Cross-validated lambda selection ──
    lambdas_cv = np.logspace(1, -2, 30)
    mse_cv = cv_lasso(X, y, lambdas_cv)
    best_idx = int(np.argmin(mse_cv))
    best_lam = lambdas_cv[best_idx]

    beta_best = lasso_cd(X, y, best_lam)
    n_nonzero = np.sum(np.abs(beta_best) > 1e-8)

    print(f"\n1. Best lambda (5-fold CV):  {best_lam:.4f}")
    print(f"   Non-zero coefficients:    {n_nonzero}  (true: 3)")
    print(f"   Min CV MSE:               {mse_cv[best_idx]:.3f}")
    print(f"\n   Fitted coefficients:")
    for j in range(p):
        marker = " *" if np.abs(beta_true[j]) > 0 else ""
        print(f"     beta[{j}] = {beta_best[j]:+.3f}  (true: {beta_true[j]:+.1f}){marker}")

    # ── Plots ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Regularisation path
    for j in range(p):
        lw = 2 if np.abs(beta_true[j]) > 0 else 0.8
        ax1.plot(np.log10(lambdas), path[:, j], linewidth=lw,
                 label=f'x{j}' if np.abs(beta_true[j]) > 0 else None)
    ax1.axvline(np.log10(best_lam), color='grey', linestyle=':', alpha=0.6)
    ax1.set_xlabel('log10(lambda)')
    ax1.set_ylabel('Coefficient value')
    ax1.set_title('Lasso Regularisation Path')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # CV MSE
    ax2.plot(np.log10(lambdas_cv), mse_cv, 'o-', markersize=4)
    ax2.axvline(np.log10(best_lam), color='red', linestyle='--',
                label=f'Best lambda = {best_lam:.3f}')
    ax2.set_xlabel('log10(lambda)')
    ax2.set_ylabel('5-Fold CV MSE')
    ax2.set_title('Cross-Validated Lambda Selection')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
