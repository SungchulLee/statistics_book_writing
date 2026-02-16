"""
Cross-Validation for Polynomial Model Selection
================================================

This script demonstrates three resampling approaches for selecting the
optimal polynomial degree:

1. Validation Set Approach: Single random split, shows high variability
2. Leave-One-Out Cross-Validation (LOOCV): Computationally expensive, lowest variance
3. k-Fold Cross-Validation: Practical balance between bias and variance

Each method estimates test error by averaging over multiple training/test splits.
The results show how validation set approach (single split) can be misleading,
while k-fold CV provides stable estimates with modest computational cost.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (cross_val_score, LeaveOneOut, KFold,
                                      ShuffleSplit)

# =============================================================================
# Generate synthetic regression data with true nonlinear relationship
# =============================================================================

np.random.seed(42)

n = 200
X = np.random.uniform(1, 10, n)
# True relationship: quadratic with noise
y = 5 + 2 * X - 0.3 * X**2 + np.random.normal(0, 2, n)

X_2d = X.reshape(-1, 1)

# =============================================================================
# Define polynomial degrees to test
# =============================================================================

degrees = np.arange(1, 11)  # Test polynomials from degree 1 to 10
n_splits = 10

# =============================================================================
# 1. VALIDATION SET APPROACH
# =============================================================================

print("=" * 75)
print("1. VALIDATION SET APPROACH (Single Random Split)")
print("=" * 75)

# We'll perform multiple validation set approaches to show variability
n_validations = 20
val_mse_multiple = np.zeros((n_validations, len(degrees)))

for run in range(n_validations):
    # Random split: 80% training, 20% validation
    val_size = int(0.2 * n)
    indices = np.random.permutation(n)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    X_train, X_val = X_2d[train_idx], X_2d[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    for i, degree in enumerate(degrees):
        # Fit polynomial
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        # Train and evaluate
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_val_poly)

        mse = np.mean((y_val - y_pred) ** 2)
        val_mse_multiple[run, i] = mse

# Average and std across validation runs
val_mse_mean = val_mse_multiple.mean(axis=0)
val_mse_std = val_mse_multiple.std(axis=0)

print(f"Performed {n_validations} random validation set splits (80/20)")
print(f"Average MSE across splits for each degree:")
for degree, mse, std in zip(degrees, val_mse_mean, val_mse_std):
    print(f"  Degree {degree:2d}: MSE = {mse:.3f} (std = {std:.3f})")

best_degree_val = degrees[np.argmin(val_mse_mean)]
print(f"\nBest degree (validation approach): {best_degree_val}")

# =============================================================================
# 2. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# =============================================================================

print("\n" + "=" * 75)
print("2. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print("=" * 75)

loo = LeaveOneOut()
loocv_mse = np.zeros(len(degrees))

for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)

    model = LinearRegression()
    # Negative MSE scores (sklearn convention: higher is better)
    scores = cross_val_score(model, X_poly, y, cv=loo,
                             scoring='neg_mean_squared_error')
    loocv_mse[i] = -scores.mean()

print(f"Performed LOOCV (n={n} folds, one sample per fold)")
print(f"MSE for each degree:")
for degree, mse in zip(degrees, loocv_mse):
    print(f"  Degree {degree:2d}: MSE = {mse:.3f}")

best_degree_loocv = degrees[np.argmin(loocv_mse)]
print(f"\nBest degree (LOOCV): {best_degree_loocv}")

# =============================================================================
# 3. k-FOLD CROSS-VALIDATION (k=10)
# =============================================================================

print("\n" + "=" * 75)
print("3. k-FOLD CROSS-VALIDATION (k=10)")
print("=" * 75)

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold_mse = np.zeros(len(degrees))
kfold_std = np.zeros(len(degrees))

for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=kfold,
                             scoring='neg_mean_squared_error')
    kfold_mse[i] = -scores.mean()
    kfold_std[i] = scores.std()

print(f"Performed {n_splits}-fold CV (each fold ~{n//n_splits} samples)")
print(f"Mean MSE and std across folds:")
for degree, mse, std in zip(degrees, kfold_mse, kfold_std):
    print(f"  Degree {degree:2d}: MSE = {mse:.3f} (std = {std:.3f})")

best_degree_kfold = degrees[np.argmin(kfold_mse)]
print(f"\nBest degree (k-fold CV): {best_degree_kfold}")

# =============================================================================
# VISUALIZATION: Compare all three approaches
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- LEFT PANEL: Validation Set (multiple runs) ---
axes[0].scatter(degrees, val_mse_mean, s=80, color='steelblue',
                label='Mean of 20 runs', zorder=3)
axes[0].errorbar(degrees, val_mse_mean, yerr=val_mse_std,
                 fmt='none', ecolor='steelblue', alpha=0.4, capsize=5)
# Plot individual runs as light background
for run in range(n_validations):
    axes[0].plot(degrees, val_mse_multiple[run, :], 'steelblue',
                 alpha=0.1, linewidth=0.8)
axes[0].axvline(x=best_degree_val, color='red', linestyle='--', linewidth=2,
                label=f'Best: degree {best_degree_val}')
axes[0].set_xlabel('Polynomial Degree', fontsize=11)
axes[0].set_ylabel('Mean Squared Error', fontsize=11)
axes[0].set_title('Validation Set Approach\n(20 random 80/20 splits)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(degrees)

# --- MIDDLE PANEL: LOOCV ---
axes[1].plot(degrees, loocv_mse, 'o-', color='darkgreen', linewidth=2.5,
             markersize=8, label='LOOCV')
axes[1].axvline(x=best_degree_loocv, color='red', linestyle='--', linewidth=2,
                label=f'Best: degree {best_degree_loocv}')
axes[1].set_xlabel('Polynomial Degree', fontsize=11)
axes[1].set_ylabel('Mean Squared Error', fontsize=11)
axes[1].set_title(f'Leave-One-Out CV\n({n} folds, minimal variance)',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(degrees)

# --- RIGHT PANEL: k-Fold CV ---
axes[2].errorbar(degrees, kfold_mse, yerr=kfold_std, fmt='o-',
                 color='purple', linewidth=2.5, markersize=8,
                 label=f'{n_splits}-Fold CV', capsize=5, alpha=0.8)
axes[2].axvline(x=best_degree_kfold, color='red', linestyle='--', linewidth=2,
                label=f'Best: degree {best_degree_kfold}')
axes[2].set_xlabel('Polynomial Degree', fontsize=11)
axes[2].set_ylabel('Mean Squared Error', fontsize=11)
axes[2].set_title(f'{n_splits}-Fold Cross-Validation\n(practical balance)',
                  fontsize=11, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(degrees)

plt.tight_layout()
plt.savefig('cv_polynomial_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 75)
print("SUMMARY: COMPARISON OF THREE APPROACHES")
print("=" * 75)

comparison_data = {
    'Method': ['Validation Set', 'LOOCV', f'{n_splits}-Fold CV'],
    'Best Degree': [best_degree_val, best_degree_loocv, best_degree_kfold],
    'Optimal MSE': [val_mse_mean[best_degree_val-1],
                    loocv_mse[best_degree_loocv-1],
                    kfold_mse[best_degree_kfold-1]],
    'Variability': ['High (depends on split)', 'Very low', 'Moderate'],
    'Computation': ['Fast (1 split)', f'Slow ({n} models)', f'Moderate ({n_splits} models)'],
    'Bias': ['Optimistic', 'Unbiased', 'Slightly pessimistic']
}

print("\n{:<20} {:<15} {:<15} {:<25} {:<20}".format(
    'Method', 'Best Degree', 'Optimal MSE', 'Variability', 'Computation'))
print("-" * 95)
for i in range(len(comparison_data['Method'])):
    print("{:<20} {:<15} {:<15.3f} {:<25} {:<20}".format(
        comparison_data['Method'][i],
        str(comparison_data['Best Degree'][i]),
        comparison_data['Optimal MSE'][i],
        comparison_data['Variability'][i],
        comparison_data['Computation'][i]))

print("\nKEY INSIGHTS:")
print("• Validation set approach shows high variability across random splits")
print("• LOOCV has lowest variance but computationally expensive for large datasets")
print("• 10-fold CV provides good balance: stable estimates with reasonable computation")
print("• As polynomial degree increases, test error eventually increases (overfitting)")

print("\n" + "=" * 75)
