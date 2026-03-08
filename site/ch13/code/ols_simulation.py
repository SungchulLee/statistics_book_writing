"""
OLS Simulation — Monte Carlo with Projection Matrices
=======================================================
Adapted from Basic-Statistics-With-Python linregfunc.py and
merged with concepts from the existing ols_regression_output.py.

Demonstrates OLS estimation from a linear-algebra perspective:
  - Data generation  (X, beta, u, y = X @ beta + u)
  - Normal-equation estimator  beta_hat = (X'X)^{-1} X'y
  - Projection matrices  P = X(X'X)^{-1}X'  and  M = I - P
  - ANOVA decomposition   TSS = ESS + RSS
  - Unbiased variance estimator  s^2
  - Covariance matrix of beta_hat
  - Monte-Carlo verification of unbiasedness
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Data generation ────────────────────────────────────────
def gen_X(n, k):
    """Random design matrix [1 | X_2 ... X_k] of shape (n, k)."""
    const = np.ones((n, 1))
    X_indep = np.random.randn(n, k - 1)
    return np.hstack([const, X_indep])


def gen_beta(params):
    """Column vector of true coefficients."""
    return np.array(params).reshape(-1, 1)


def gen_u(n, sigma=1.0):
    """Column vector of disturbances ~ N(0, sigma^2)."""
    return np.random.randn(n, 1) * sigma


# ── OLS estimator ──────────────────────────────────────────
def ols(y, X):
    """beta_hat = (X'X)^{-1} X'y"""
    return np.linalg.inv(X.T @ X) @ X.T @ y


# ── Projection matrices ────────────────────────────────────
def proj_P(X):
    """Projection matrix P = X (X'X)^{-1} X'  (projects onto col(X))."""
    return X @ np.linalg.inv(X.T @ X) @ X.T


def proj_M(X):
    """Annihilator matrix M = I - P  (projects onto col(X)^perp)."""
    return np.eye(X.shape[0]) - proj_P(X)


# ── ANOVA decomposition ────────────────────────────────────
def anova_decomposition(y, X, beta_hat):
    """Return (TSS, ESS, RSS) where TSS = ESS + RSS (centered)."""
    y_bar = y.mean()
    y_hat = X @ beta_hat
    TSS = float(np.sum((y - y_bar) ** 2))
    ESS = float(np.sum((y_hat - y_bar) ** 2))
    RSS = float(np.sum((y - y_hat) ** 2))
    return TSS, ESS, RSS


# ── Variance and covariance ────────────────────────────────
def s_squared(resid, k):
    """Unbiased estimator of sigma^2: s^2 = e'e / (n - k)."""
    return float(np.sum(resid ** 2) / (len(resid) - k))


def cov_beta_hat(resid, k, X):
    """Var(beta_hat) = s^2 (X'X)^{-1}."""
    s2 = s_squared(resid, k)
    cov = s2 * np.linalg.inv(X.T @ X)
    return cov


# ── Single simulation ──────────────────────────────────────
def run_one_ols(n=100, beta_true=[2, 3, -1], sigma=1.0):
    """Generate data and return OLS results as a dict."""
    k = len(beta_true)
    X = gen_X(n, k)
    beta = gen_beta(beta_true)
    u = gen_u(n, sigma)
    y = X @ beta + u

    bhat = ols(y, X)
    resid = y - X @ bhat
    TSS, ESS, RSS = anova_decomposition(y, X, bhat)
    R2 = ESS / TSS
    s2 = s_squared(resid, k)
    cov = cov_beta_hat(resid, k, X)

    return {
        "beta_hat": bhat.flatten(),
        "resid": resid,
        "TSS": TSS, "ESS": ESS, "RSS": RSS,
        "R2": R2, "s2": s2,
        "se": np.sqrt(cov.diagonal()),
        "P": proj_P(X), "M": proj_M(X),
    }


# ── Monte Carlo verification ───────────────────────────────
def monte_carlo(n=100, beta_true=[2, 3, -1], sigma=1.0, n_sim=5000):
    """Repeat OLS n_sim times; return array of beta_hat estimates."""
    k = len(beta_true)
    estimates = np.empty((n_sim, k))
    for i in range(n_sim):
        res = run_one_ols(n, beta_true, sigma)
        estimates[i] = res["beta_hat"]
    return estimates


# ── Main ────────────────────────────────────────────────────
def main():
    beta_true = [2, 3, -1]
    n, sigma = 200, 2.0
    k = len(beta_true)

    print("=" * 60)
    print("OLS Simulation — Projection Matrices & Monte Carlo")
    print("=" * 60)

    # single run
    res = run_one_ols(n, beta_true, sigma)
    print(f"\n--- Single OLS (n={n}, sigma={sigma}) ---")
    print(f"  True beta:  {beta_true}")
    print(f"  beta_hat:   {np.round(res['beta_hat'], 4)}")
    print(f"  Std errors: {np.round(res['se'], 4)}")
    print(f"  R-squared:  {res['R2']:.4f}")
    print(f"  s^2:        {res['s2']:.4f}  (true sigma^2 = {sigma**2})")
    print(f"  TSS = {res['TSS']:.2f}  ESS = {res['ESS']:.2f}  RSS = {res['RSS']:.2f}")

    # projection matrix properties
    P = res["P"]
    M = res["M"]
    print(f"\n--- Projection Matrix Properties ---")
    print(f"  P is idempotent (P^2 == P): {np.allclose(P @ P, P)}")
    print(f"  M is idempotent (M^2 == M): {np.allclose(M @ M, M)}")
    print(f"  P is symmetric:             {np.allclose(P, P.T)}")
    print(f"  M is symmetric:             {np.allclose(M, M.T)}")
    print(f"  trace(P) = {np.trace(P):.1f}  (should be k = {k})")
    print(f"  trace(M) = {np.trace(M):.1f}  (should be n-k = {n-k})")

    # Monte Carlo
    n_sim = 5000
    print(f"\n--- Monte Carlo ({n_sim} replications) ---")
    estimates = monte_carlo(n, beta_true, sigma, n_sim)
    mc_mean = estimates.mean(axis=0)
    mc_std = estimates.std(axis=0, ddof=1)
    for j in range(k):
        print(f"  beta_{j}: true={beta_true[j]:>5},  "
              f"MC mean={mc_mean[j]:>7.4f},  MC std={mc_std[j]:.4f}")

    # visualisation
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4))
    for j in range(k):
        ax = axes[j]
        ax.hist(estimates[:, j], bins=40, density=True,
                edgecolor="white", alpha=0.7)
        ax.axvline(beta_true[j], color="red", linestyle="--", linewidth=2,
                   label=f"True = {beta_true[j]}")
        ax.axvline(mc_mean[j], color="blue", linestyle=":", linewidth=2,
                   label=f"MC mean = {mc_mean[j]:.3f}")
        ax.set_xlabel(f"beta_{j}")
        ax.set_title(f"Sampling Distribution of beta_{j}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("ols_simulation.png", dpi=150)
    plt.show()
    print("\nFigure saved: ols_simulation.png")


if __name__ == "__main__":
    main()
