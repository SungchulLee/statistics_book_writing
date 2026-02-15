#!/usr/bin/env python3
# ======================================================================
# 13_wls_01_weighted_least_squares.py
# ======================================================================
# Weighted Least Squares (WLS) regression:
#   1. OLS vs WLS when variance increases with x (heteroscedasticity).
#   2. Residual plots for OLS and WLS.
#   3. Comparison of standard errors.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 4 — Regression and Prediction).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def ols_fit(X, y):
    """Ordinary least squares: beta = (X'X)^{-1} X'y."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def wls_fit(X, y, w):
    """
    Weighted least squares: beta = (X'WX)^{-1} X'Wy.

    Parameters
    ----------
    X : (n, p) design matrix.
    y : (n,)   response.
    w : (n,)   positive weights (inversely proportional to variance).
    """
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta


def main():
    print("Weighted Least Squares Regression")
    print("=" * 55)

    # ── Generate heteroscedastic data ──
    n = 120
    x = np.random.uniform(1, 10, n)
    # Variance grows linearly with x
    sigma = 0.5 + 1.5 * x
    y = 3.0 + 2.0 * x + np.random.normal(0, sigma)

    X = np.column_stack([np.ones(n), x])

    # ── 1. OLS ──
    beta_ols = ols_fit(X, y)
    y_hat_ols = X @ beta_ols
    resid_ols = y - y_hat_ols
    rss_ols = np.sum(resid_ols ** 2)

    # ── 2. WLS with weights = 1 / sigma^2 ──
    # In practice we estimate sigma; here we use the known form
    w = 1.0 / sigma ** 2
    beta_wls = wls_fit(X, y, w)
    y_hat_wls = X @ beta_wls
    resid_wls = y - y_hat_wls
    rss_wls = np.sum(w * resid_wls ** 2)

    # ── Standard errors ──
    # OLS SE (homoscedastic formula)
    s2_ols = rss_ols / (n - 2)
    se_ols = np.sqrt(np.diag(s2_ols * np.linalg.inv(X.T @ X)))

    # WLS SE
    W = np.diag(w)
    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    se_wls = np.sqrt(np.diag(XtWX_inv))

    print(f"\n  {'':12s} {'Intercept':>12s} {'Slope':>12s}")
    print(f"  {'True':12s} {'3.000':>12s} {'2.000':>12s}")
    print(f"  {'OLS coef':12s} {beta_ols[0]:>12.3f} {beta_ols[1]:>12.3f}")
    print(f"  {'OLS SE':12s} {se_ols[0]:>12.3f} {se_ols[1]:>12.3f}")
    print(f"  {'WLS coef':12s} {beta_wls[0]:>12.3f} {beta_wls[1]:>12.3f}")
    print(f"  {'WLS SE':12s} {se_wls[0]:>12.3f} {se_wls[1]:>12.3f}")

    # ── Plots ──
    x_line = np.linspace(0, 11, 100)
    X_line = np.column_stack([np.ones(100), x_line])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Data + fitted lines
    ax = axes[0]
    ax.scatter(x, y, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
    ax.plot(x_line, X_line @ beta_ols, 'b-', linewidth=2, label='OLS')
    ax.plot(x_line, X_line @ beta_wls, 'r--', linewidth=2, label='WLS')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('OLS vs WLS Fit')
    ax.legend(fontsize=9)

    # OLS residuals
    ax = axes[1]
    ax.scatter(y_hat_ols, resid_ols, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xlabel('OLS Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('OLS Residuals (fan shape)')

    # WLS weighted residuals
    ax = axes[2]
    weighted_resid = np.sqrt(w) * resid_wls
    ax.scatter(y_hat_wls, weighted_resid, alpha=0.5, s=20,
               edgecolors='k', linewidths=0.3, color='C3')
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xlabel('WLS Fitted values')
    ax.set_ylabel('Weighted residuals')
    ax.set_title('WLS Weighted Residuals (stabilised)')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
