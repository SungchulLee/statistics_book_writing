"""
Regularization Techniques: Ridge, Lasso, and Elastic Net
=========================================================
Demonstrates Ridge, Lasso, and Elastic Net regression with
scikit-learn, including cross-validation for hyperparameter tuning,
coefficient path visualization, and comparison across methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. Generate Synthetic Data with Multicollinearity
# ============================================================

np.random.seed(42)
n_samples, n_features, n_informative = 200, 20, 5

X, y, true_coef = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=10,
    coef=True,
    random_state=42
)

# Add multicollinearity: duplicate some features with noise
X[:, 5] = X[:, 0] + np.random.normal(0, 0.1, n_samples)
X[:, 6] = X[:, 1] + np.random.normal(0, 0.1, n_samples)
X[:, 7] = X[:, 2] + np.random.normal(0, 0.1, n_samples)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
split = int(0.8 * n_samples)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

print("=" * 60)
print("Dataset: {} samples, {} features ({} informative)".format(
    n_samples, n_features, n_informative))
print("Multicollinearity added to features 5, 6, 7")
print("=" * 60)


# ============================================================
# 2. Ridge Regression with Cross-Validation
# ============================================================

print("\n--- Ridge Regression ---")

alphas_ridge = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas_ridge, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha (lambda): {ridge_cv.alpha_:.4f}")
y_pred_ridge = ridge_cv.predict(X_test)
print(f"Test MSE:  {mean_squared_error(y_test, y_pred_ridge):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Non-zero coefficients: {np.sum(ridge_cv.coef_ != 0)} / {n_features}")


# ============================================================
# 3. Lasso Regression with Cross-Validation
# ============================================================

print("\n--- Lasso Regression ---")

lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

print(f"Best alpha (lambda): {lasso_cv.alpha_:.4f}")
y_pred_lasso = lasso_cv.predict(X_test)
print(f"Test MSE:  {mean_squared_error(y_test, y_pred_lasso):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)} / {n_features}")


# ============================================================
# 4. Elastic Net with Cross-Validation
# ============================================================

print("\n--- Elastic Net ---")

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    alphas=np.logspace(-3, 1, 100),
    cv=5,
    random_state=42,
    max_iter=10000
)
enet_cv.fit(X_train, y_train)

print(f"Best alpha (lambda): {enet_cv.alpha_:.4f}")
print(f"Best l1_ratio (alpha): {enet_cv.l1_ratio_:.2f}")
y_pred_enet = enet_cv.predict(X_test)
print(f"Test MSE:  {mean_squared_error(y_test, y_pred_enet):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred_enet):.4f}")
print(f"Non-zero coefficients: {np.sum(enet_cv.coef_ != 0)} / {n_features}")


# ============================================================
# 5. Coefficient Comparison
# ============================================================

print("\n--- Coefficient Comparison ---")
print(f"{'Feature':<10} {'True':>10} {'Ridge':>10} {'Lasso':>10} {'ElasticNet':>12}")
print("-" * 55)
for i in range(n_features):
    print(f"x_{i:<7} {true_coef[i]:>10.2f} {ridge_cv.coef_[i]:>10.2f} "
          f"{lasso_cv.coef_[i]:>10.2f} {enet_cv.coef_[i]:>12.2f}")


# ============================================================
# 6. Visualization: Coefficient Paths
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ridge coefficient path
alphas_path = np.logspace(-2, 4, 200)
ridge_coefs = []
for a in alphas_path:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)
ridge_coefs = np.array(ridge_coefs)

axes[0].semilogx(alphas_path, ridge_coefs)
axes[0].axvline(ridge_cv.alpha_, color='k', linestyle='--', label=f'CV best: {ridge_cv.alpha_:.2f}')
axes[0].set_xlabel('Alpha (lambda)')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_title('Ridge Regression Coefficient Path')
axes[0].legend()

# Lasso coefficient path
alphas_lasso = np.logspace(-3, 2, 200)
lasso_coefs = []
for a in alphas_lasso:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_coefs.append(lasso.coef_)
lasso_coefs = np.array(lasso_coefs)

axes[1].semilogx(alphas_lasso, lasso_coefs)
axes[1].axvline(lasso_cv.alpha_, color='k', linestyle='--', label=f'CV best: {lasso_cv.alpha_:.2f}')
axes[1].set_xlabel('Alpha (lambda)')
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('Lasso Regression Coefficient Path')
axes[1].legend()

# Elastic Net coefficient path
alphas_enet = np.logspace(-3, 2, 200)
enet_coefs = []
for a in alphas_enet:
    enet = ElasticNet(alpha=a, l1_ratio=enet_cv.l1_ratio_, max_iter=10000)
    enet.fit(X_train, y_train)
    enet_coefs.append(enet.coef_)
enet_coefs = np.array(enet_coefs)

axes[2].semilogx(alphas_enet, enet_coefs)
axes[2].axvline(enet_cv.alpha_, color='k', linestyle='--',
                label=f'CV best: {enet_cv.alpha_:.2f} (l1={enet_cv.l1_ratio_:.1f})')
axes[2].set_xlabel('Alpha (lambda)')
axes[2].set_ylabel('Coefficient Value')
axes[2].set_title('Elastic Net Coefficient Path')
axes[2].legend()

plt.tight_layout()
plt.savefig('regularization_coefficient_paths.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: regularization_coefficient_paths.png")


# ============================================================
# 7. Summary Comparison
# ============================================================

print("\n" + "=" * 60)
print("Summary Comparison")
print("=" * 60)
results = {
    'Ridge': (mean_squared_error(y_test, y_pred_ridge),
              r2_score(y_test, y_pred_ridge),
              np.sum(ridge_cv.coef_ != 0)),
    'Lasso': (mean_squared_error(y_test, y_pred_lasso),
              r2_score(y_test, y_pred_lasso),
              np.sum(lasso_cv.coef_ != 0)),
    'Elastic Net': (mean_squared_error(y_test, y_pred_enet),
                    r2_score(y_test, y_pred_enet),
                    np.sum(enet_cv.coef_ != 0)),
}

print(f"{'Method':<15} {'MSE':>10} {'R²':>10} {'Non-zero':>10}")
print("-" * 48)
for method, (mse, r2, nz) in results.items():
    print(f"{method:<15} {mse:>10.4f} {r2:>10.4f} {nz:>10}")
