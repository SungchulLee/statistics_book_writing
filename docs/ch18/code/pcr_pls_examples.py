#!/usr/bin/env python3
"""
======================================================================
Principal Components Regression (PCR) and Partial Least Squares (PLS)
======================================================================

Demonstrates PCR and PLS regression on housing data with:

  1. PCR: PCA-based dimensionality reduction + regression
  2. PLS: Supervised dimensionality reduction (covariance with response)
  3. Cross-validation for selecting number of components
  4. Comparison of PCR vs PLS vs OLS vs Ridge
  5. Visualizations: scree plot, CV curves, predictions

Source: Adapted from ISLR (Chapter 6 - Linear Model Selection)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure visualization
sns.set_style("whitegrid")
np.random.seed(42)

# ======================================================================
# 1. Load and Prepare Data
# ======================================================================

DATA = Path(__file__).parent.parent.parent / 'data'
HOUSE_CSV = DATA / 'house_sales.csv'

house = pd.read_csv(HOUSE_CSV, sep='\t')

print("=" * 70)
print("PRINCIPAL COMPONENTS REGRESSION (PCR) AND PARTIAL LEAST SQUARES (PLS)")
print("=" * 70)

# Select numeric features only (for simplicity)
numeric_features = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt', 'YrRenovated'
]
outcome = 'AdjSalePrice'

X = house[numeric_features].values
y = house[outcome].values

print(f"\nDataset Information:")
print(f"  Observations: {len(X)}")
print(f"  Features (p): {X.shape[1]}")
print(f"  Feature names: {numeric_features}")

# Standardize features (essential for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)

print(f"\nTarget Variable Statistics:")
print(f"  Mean: ${y.mean():,.0f}")
print(f"  Std:  ${y.std():,.0f}")

# ======================================================================
# 2. OLS Baseline
# ======================================================================

print("\n" + "=" * 70)
print("BASELINE: ORDINARY LEAST SQUARES (OLS)")
print("=" * 70)

ols_model = LinearRegression()
ols_model.fit(X_scaled, y)
ols_pred = ols_model.predict(X_scaled)
ols_rmse = np.sqrt(mean_squared_error(y, ols_pred))
ols_r2 = r2_score(y, ols_pred)

print(f"\nOLS Model (In-sample):")
print(f"  R²:   {ols_r2:.4f}")
print(f"  RMSE: ${ols_rmse:,.0f}")

# ======================================================================
# 3. Principal Components Regression (PCR)
# ======================================================================

print("\n" + "=" * 70)
print("PRINCIPAL COMPONENTS REGRESSION (PCR)")
print("=" * 70)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance explained
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

print(f"\nPCA Results:")
print(f"  Total components: {len(explained_var)}")
print(f"  Variance explained (first 5):")
for i in range(min(5, len(explained_var))):
    print(f"    PC{i+1}: {explained_var[i]:.4f} (cumulative: {cumsum_var[i]:.4f})")

# Cross-validation to select number of components
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
pcr_mse_scores = []

print(f"\nCross-validation for PCR (10-fold CV):")
for M in range(1, X_scaled.shape[1] + 1):
    reg = LinearRegression()
    cv_scores = cross_val_score(
        reg, X_pca[:, :M], y,
        cv=kfold,
        scoring='neg_mean_squared_error'
    )
    mse = -cv_scores.mean()
    pcr_mse_scores.append(mse)

    if M <= 5 or M % 2 == 0:  # Print first 5 and then every other
        print(f"  M={M:2d}: CV MSE = {mse:,.0f} (var explained: {cumsum_var[M-1]:.4f})")

# Find optimal M
M_opt_pcr = np.argmin(pcr_mse_scores) + 1
pcr_cv_rmse = np.sqrt(pcr_mse_scores[M_opt_pcr - 1])

print(f"\nOptimal PCR: M = {M_opt_pcr} components")
print(f"  Variance explained: {cumsum_var[M_opt_pcr-1]:.4f} ({cumsum_var[M_opt_pcr-1]*100:.2f}%)")
print(f"  CV RMSE: ${pcr_cv_rmse:,.0f}")

# Fit final PCR model
pcr_final = LinearRegression()
pcr_final.fit(X_pca[:, :M_opt_pcr], y)
pcr_pred = pcr_final.predict(X_pca[:, :M_opt_pcr])
pcr_rmse = np.sqrt(mean_squared_error(y, pcr_pred))
pcr_r2 = r2_score(y, pcr_pred)

print(f"\nPCR Model Performance (In-sample):")
print(f"  R²:   {pcr_r2:.4f}")
print(f"  RMSE: ${pcr_rmse:,.0f}")

# ======================================================================
# 4. Partial Least Squares (PLS)
# ======================================================================

print("\n" + "=" * 70)
print("PARTIAL LEAST SQUARES (PLS)")
print("=" * 70)

# Cross-validation to select number of components
pls_mse_scores = []

print(f"\nCross-validation for PLS (10-fold CV):")
for M in range(1, X_scaled.shape[1] + 1):
    pls = PLSRegression(n_components=M)
    cv_scores = cross_val_score(
        pls, X_scaled, y,
        cv=kfold,
        scoring='neg_mean_squared_error'
    )
    mse = -cv_scores.mean()
    pls_mse_scores.append(mse)

    if M <= 5 or M % 2 == 0:  # Print first 5 and then every other
        print(f"  M={M:2d}: CV MSE = {mse:,.0f}")

# Find optimal M
M_opt_pls = np.argmin(pls_mse_scores) + 1
pls_cv_rmse = np.sqrt(pls_mse_scores[M_opt_pls - 1])

print(f"\nOptimal PLS: M = {M_opt_pls} components")
print(f"  CV RMSE: ${pls_cv_rmse:,.0f}")

# Fit final PLS model
pls_final = PLSRegression(n_components=M_opt_pls)
pls_final.fit(X_scaled, y)
pls_pred = pls_final.predict(X_scaled)
pls_rmse = np.sqrt(mean_squared_error(y, pls_pred))
pls_r2 = r2_score(y, pls_pred)

print(f"\nPLS Model Performance (In-sample):")
print(f"  R²:   {pls_r2:.4f}")
print(f"  RMSE: ${pls_rmse:,.0f}")

# ======================================================================
# 5. Ridge Regression (for comparison)
# ======================================================================

print("\n" + "=" * 70)
print("RIDGE REGRESSION (for comparison)")
print("=" * 70)

ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=10)
ridge_cv.fit(X_scaled, y)
ridge_pred = ridge_cv.predict(X_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
ridge_r2 = r2_score(y, ridge_pred)
ridge_lambda = ridge_cv.alpha_

print(f"\nOptimal Ridge: λ = {ridge_lambda:.4f}")
print(f"\nRidge Model Performance (In-sample):")
print(f"  R²:   {ridge_r2:.4f}")
print(f"  RMSE: ${ridge_rmse:,.0f}")

# ======================================================================
# 6. Model Comparison
# ======================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'PCR', 'PLS'],
    'Hyperparameter': ['None', f'λ={ridge_lambda:.4f}', f'M={M_opt_pcr}', f'M={M_opt_pls}'],
    'R² (In-sample)': [ols_r2, ridge_r2, pcr_r2, pls_r2],
    'RMSE (In-sample)': [ols_rmse, ridge_rmse, pcr_rmse, pls_rmse],
    'CV RMSE': [
        np.sqrt(cross_val_score(LinearRegression(), X_scaled, y, cv=kfold,
                               scoring='neg_mean_squared_error').mean() * -1),
        ridge_cv_rmse := np.sqrt(-ridge_cv.cv_results_['mean_test_score'].max()),
        pcr_cv_rmse,
        pls_cv_rmse
    ]
})

print(f"\n{comparison.to_string(index=False)}")

# ======================================================================
# 7. Visualizations
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: PCA Variance Explained ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.plot(range(1, len(explained_var) + 1), explained_var, 'o-',
         linewidth=2, markersize=8, color='steelblue')
ax1.axvline(M_opt_pcr, color='red', linestyle='--', linewidth=2,
           label=f'Optimal M={M_opt_pcr}')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained', fontsize=12)
ax1.set_title('Scree Plot: Variance Explained by Each PC', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative variance
ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-',
         linewidth=2, markersize=8, color='darkgreen')
ax2.axhline(0.9, color='orange', linestyle='--', linewidth=1.5,
           label='90% threshold')
ax2.axvline(M_opt_pcr, color='red', linestyle='--', linewidth=2,
           label=f'Optimal M={M_opt_pcr}')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=13)
ax2.set_ylim([0, 1.05])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 2: Cross-Validation Error: PCR vs PLS ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(range(1, len(pcr_mse_scores) + 1), np.sqrt(pcr_mse_scores),
       'o-', linewidth=2, markersize=6, label='PCR', color='steelblue')
ax.plot(range(1, len(pls_mse_scores) + 1), np.sqrt(pls_mse_scores),
       's-', linewidth=2, markersize=6, label='PLS', color='darkgreen')
ax.axvline(M_opt_pcr, color='steelblue', linestyle='--', linewidth=1.5,
          label=f'PCR optimal (M={M_opt_pcr})')
ax.axvline(M_opt_pls, color='darkgreen', linestyle='--', linewidth=1.5,
          label=f'PLS optimal (M={M_opt_pls})')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('CV RMSE', fontsize=12)
ax.set_title('Cross-Validation Error: PCR vs PLS', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 3: Model Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# R² Comparison
models = ['OLS', 'Ridge', 'PCR', 'PLS']
r2_values = [ols_r2, ridge_r2, pcr_r2, pls_r2]
colors = ['blue', 'green', 'orange', 'red']

x_pos = np.arange(len(models))
ax1.bar(x_pos, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('Model Comparison: R² (In-sample)', fontsize=13)
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(r2_values):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

# CV RMSE Comparison
cv_rmse_values = [
    np.sqrt(cross_val_score(LinearRegression(), X_scaled, y, cv=kfold,
                           scoring='neg_mean_squared_error').mean() * -1),
    ridge_cv_rmse,
    pcr_cv_rmse,
    pls_cv_rmse
]

ax2.bar(x_pos, cv_rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylabel('CV RMSE ($)', fontsize=12)
ax2.set_title('Model Comparison: Cross-Validation RMSE', fontsize=13)
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cv_rmse_values):
    ax2.text(i, v + 5000, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# --- Plot 4: Predicted vs Actual ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

predictions = [ols_pred, ridge_pred, pcr_pred, pls_pred]
r2_vals = [ols_r2, ridge_r2, pcr_r2, pls_r2]

for idx, (ax, pred, r2, model_name) in enumerate(zip(axes, predictions, r2_vals, models)):
    ax.scatter(y, pred, alpha=0.3, s=10)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price ($)', fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontsize=11)
    ax.set_title(f'{model_name}: R² = {r2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ======================================================================
# 8. Summary Report
# ======================================================================

print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

summary = f"""
Dataset: King County Housing Prices
  Observations: {len(X)}
  Features: {X.shape[1]}
  Target: Adjusted Sale Price

OLS (All {X.shape[1]} features):
  R²: {ols_r2:.4f}
  RMSE: ${ols_rmse:,.0f}

PCR (M={M_opt_pcr} components, explaining {cumsum_var[M_opt_pcr-1]:.1%} variance):
  R²: {pcr_r2:.4f}
  RMSE: ${pcr_rmse:,.0f}
  Advantage: Handles multicollinearity, reduced overfitting
  Trade-off: Component interpretation, unsupervised

PLS (M={M_opt_pls} components):
  R²: {pls_r2:.4f}
  RMSE: ${pls_rmse:,.0f}
  Advantage: Supervised component selection, often outperforms PCR
  Trade-off: Component interpretation

Ridge Regression (λ={ridge_lambda:.4f}):
  R²: {ridge_r2:.4f}
  RMSE: ${ridge_rmse:,.0f}
  Advantage: All features retained, easy interpretation
  Trade-off: Doesn't select features

Key Insights:
  1. PCR reduced dimensionality from {X.shape[1]} to {M_opt_pcr} while maintaining R²
  2. PLS used only {M_opt_pls} components (supervised selection)
  3. Ridge retains all features with continuous shrinkage
  4. CV RMSE ≈ {{cv_rmse_values[0]:,.0f}} (OLS), {{cv_rmse_values[2]:,.0f}} (PCR), {{cv_rmse_values[3]:,.0f}} (PLS)
  5. PLS generally preferred over PCR when prediction is the goal
"""

print(summary)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
