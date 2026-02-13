"""
Chapter 18: Regularization Techniques — Code Examples
======================================================
Ridge, Lasso, Elastic Net comparison with cross-validation,
regularization paths, and coefficient analysis.
"""

import numpy as np
from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                   RidgeCV, LassoCV, ElasticNetCV)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# 1. Generate Synthetic Data with Multicollinearity
# =============================================================================

def generate_data(n=200, p=20, s=5, rho=0.8, noise=1.0):
    """
    Generate regression data with correlated predictors.
    - n: samples, p: predictors, s: true nonzero coefficients
    - rho: correlation between adjacent predictors
    - noise: std of error term
    """
    # Correlated design matrix (Toeplitz structure)
    Sigma = np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    # True sparse coefficients
    beta_true = np.zeros(p)
    beta_true[:s] = np.array([3, -2, 1.5, -1, 0.5])

    y = X @ beta_true + noise * np.random.randn(n)
    return X, y, beta_true


X, y, beta_true = generate_data()
n, p = X.shape

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data: n={n}, p={p}, true nonzero={np.sum(beta_true != 0)}")
print(f"True β: {beta_true}")


# =============================================================================
# 2. Fit Ridge, Lasso, Elastic Net with CV
# =============================================================================

alphas = np.logspace(-4, 2, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_scaled, y)

lasso_cv = LassoCV(n_alphas=100, cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)

enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                       n_alphas=100, cv=5, max_iter=10000)
enet_cv.fit(X_scaled, y)

print("\n--- Cross-Validated Results ---")
print(f"{'Method':<15} {'Best λ':>10} {'R²':>8} {'Nonzero':>8}")
print("-" * 45)
for name, model in [("Ridge", ridge_cv), ("Lasso", lasso_cv), ("Elastic Net", enet_cv)]:
    r2 = model.score(X_scaled, y)
    nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    lam = model.alpha_
    extra = f" (l1_ratio={model.l1_ratio_})" if hasattr(model, 'l1_ratio_') else ""
    print(f"{name:<15} {lam:>10.6f} {r2:>8.4f} {nonzero:>8}{extra}")


# =============================================================================
# 3. Coefficient Comparison
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, name, coefs in [
    (axes[0], "True", beta_true),
    (axes[1], "Ridge", ridge_cv.coef_),
    (axes[2], "Lasso", lasso_cv.coef_),
    (axes[3], "Elastic Net", enet_cv.coef_),
]:
    colors = ['#d32f2f' if abs(c) > 1e-6 else '#90a4ae' for c in coefs]
    ax.bar(range(p), coefs, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_title(name)
    ax.set_xlabel("Feature index")
    ax.axhline(0, color='black', linewidth=0.5)

axes[0].set_ylabel("Coefficient value")
plt.tight_layout()
plt.savefig("coefficient_comparison.png", dpi=150)
plt.show()


# =============================================================================
# 4. Regularization Paths
# =============================================================================

def plot_regularization_paths(X, y):
    """Plot coefficient paths for Ridge and Lasso."""
    alphas_path = np.logspace(-3, 3, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ridge path
    ridge_coefs = []
    for a in alphas_path:
        model = Ridge(alpha=a)
        model.fit(X, y)
        ridge_coefs.append(model.coef_.copy())
    ridge_coefs = np.array(ridge_coefs)

    for j in range(min(p, 10)):
        axes[0].plot(np.log10(alphas_path), ridge_coefs[:, j],
                     linewidth=1.5, label=f"β_{j}" if j < 5 else None)
    axes[0].set_xlabel("log₁₀(λ)")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Ridge Regularization Path")
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].legend(fontsize=8)

    # Lasso path
    lasso_coefs = []
    alphas_lasso = np.logspace(-4, 1, 200)
    for a in alphas_lasso:
        model = Lasso(alpha=a, max_iter=10000)
        model.fit(X, y)
        lasso_coefs.append(model.coef_.copy())
    lasso_coefs = np.array(lasso_coefs)

    for j in range(min(p, 10)):
        axes[1].plot(np.log10(alphas_lasso), lasso_coefs[:, j],
                     linewidth=1.5, label=f"β_{j}" if j < 5 else None)
    axes[1].set_xlabel("log₁₀(λ)")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title("Lasso Regularization Path")
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("regularization_paths.png", dpi=150)
    plt.show()

plot_regularization_paths(X_scaled, y)


# =============================================================================
# 5. Bias-Variance Tradeoff Visualization
# =============================================================================

def bias_variance_regularization(n_sim=500, n=100, p=20, s=5):
    """Show bias-variance tradeoff across lambda for Ridge and Lasso."""
    alphas_test = np.logspace(-3, 2, 30)

    ridge_mse = {a: [] for a in alphas_test}
    lasso_mse = {a: [] for a in alphas_test}

    _, _, beta_true_bv = generate_data(n=2, p=p, s=s)  # just to get beta_true

    for _ in range(n_sim):
        X_sim, y_sim, _ = generate_data(n=n, p=p, s=s)
        X_sim = StandardScaler().fit_transform(X_sim)

        for a in alphas_test:
            ridge = Ridge(alpha=a).fit(X_sim, y_sim)
            lasso = Lasso(alpha=a, max_iter=5000).fit(X_sim, y_sim)
            ridge_mse[a].append(np.sum((ridge.coef_ - beta_true_bv)**2))
            lasso_mse[a].append(np.sum((lasso.coef_ - beta_true_bv)**2))

    ridge_avg = [np.mean(ridge_mse[a]) for a in alphas_test]
    lasso_avg = [np.mean(lasso_mse[a]) for a in alphas_test]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log10(alphas_test), ridge_avg, 'o-', label="Ridge", linewidth=2)
    ax.plot(np.log10(alphas_test), lasso_avg, 's-', label="Lasso", linewidth=2)
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("MSE of coefficient estimates")
    ax.set_title("Bias-Variance Tradeoff: Ridge vs Lasso")
    ax.legend()
    plt.tight_layout()
    plt.savefig("bv_tradeoff_regularization.png", dpi=150)
    plt.show()

bias_variance_regularization()


# =============================================================================
# 6. Soft Thresholding Visualization
# =============================================================================

def plot_shrinkage_operators(lam=1.0):
    """Compare Ridge, Lasso, and hard thresholding shrinkage."""
    z = np.linspace(-4, 4, 500)

    ridge = z / (1 + lam)
    lasso = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
    hard = z * (np.abs(z) > lam)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z, z, 'k--', alpha=0.3, label="OLS (no shrinkage)")
    ax.plot(z, ridge, linewidth=2, label=f"Ridge (λ={lam})")
    ax.plot(z, lasso, linewidth=2, label=f"Lasso (λ={lam})")
    ax.plot(z, hard, linewidth=2, label=f"Hard threshold (λ={lam})")
    ax.set_xlabel(r"OLS estimate $\hat{\beta}^{OLS}$")
    ax.set_ylabel(r"Regularized estimate $\hat{\beta}$")
    ax.set_title("Shrinkage Operators (Orthonormal Design)")
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("shrinkage_operators.png", dpi=150)
    plt.show()

plot_shrinkage_operators()
