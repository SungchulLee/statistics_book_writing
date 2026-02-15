#!/usr/bin/env python3
"""
======================================================================
Lasso Regression: Comprehensive Analysis with Regularization Path
======================================================================

Demonstrates Lasso regression on real housing data with:

  1. Regularization path: coefficients vs lambda
  2. Cross-validated lambda selection
  3. Feature importance (non-zero coefficients)
  4. Comparison with OLS and Ridge regression
  5. Model complexity vs prediction accuracy

Source: Adapted from "Practical Statistics for Data Scientists"
        (Chapter 4 — Regression and Prediction)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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

print("="*70)
print("LASSO REGRESSION: HOUSING PRICE PREDICTION")
print("="*70)

# Select features: mix of numeric and categorical
predictors = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'PropertyType', 'NbrLivingUnits',
    'SqFtFinBasement', 'YrBuilt', 'YrRenovated',
    'NewConstruction'
]
outcome = 'AdjSalePrice'

# Create dummy variables for categorical predictors
X = pd.get_dummies(house[predictors], drop_first=True)
X['NewConstruction'] = (X['NewConstruction'] == True).astype(int)
y = house[outcome]

print(f"\nDataset Information:")
print(f"  Total observations: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Feature names: {list(X.columns)}")

# Standardize features (essential for Lasso due to L1 penalty)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nTarget variable statistics:")
print(f"  Mean: ${y.mean():,.0f}")
print(f"  Std:  ${y.std():,.0f}")
print(f"  Min:  ${y.min():,.0f}")
print(f"  Max:  ${y.max():,.0f}")

# ======================================================================
# 2. OLS Baseline Model
# ======================================================================

print("\n" + "="*70)
print("BASELINE: ORDINARY LEAST SQUARES (OLS)")
print("="*70)

ols_model = LinearRegression()
ols_model.fit(X_scaled, y)

ols_pred = ols_model.predict(X_scaled)
ols_rmse = np.sqrt(mean_squared_error(y, ols_pred))
ols_r2 = r2_score(y, ols_pred)
ols_mae = mean_absolute_error(y, ols_pred)

print(f"\nOLS Model Performance (In-sample):")
print(f"  R²:   {ols_r2:.4f}")
print(f"  RMSE: ${ols_rmse:,.0f}")
print(f"  MAE:  ${ols_mae:,.0f}")

# Count non-zero coefficients
n_nonzero_ols = np.sum(np.abs(ols_model.coef_) > 1e-8)
print(f"  Non-zero coefficients: {n_nonzero_ols}/{len(X.columns)}")

# Top 5 features by absolute coefficient
top_features_ols = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ols_model.coef_
}).reindex(pd.DataFrame({'Coefficient': np.abs(ols_model.coef_)})
   .sort_values('Coefficient', ascending=False).index).head(5)
print(f"\nTop 5 OLS Coefficients:")
print(top_features_ols.to_string(index=False))

# ======================================================================
# 3. Regularization Path: Lasso
# ======================================================================

print("\n" + "="*70)
print("LASSO: REGULARIZATION PATH ANALYSIS")
print("="*70)

# Define lambda range
alphas = np.logspace(2, -2, 100)

# Fit Lasso for each alpha
lasso_coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)

print(f"\nLasso path computed for {len(alphas)} lambda values")
print(f"  Lambda range: {alphas.min():.2e} to {alphas.max():.2e}")

# Find lambda where features are selected
n_features_selected = (np.abs(lasso_coefs) > 1e-8).sum(axis=1)
print(f"  Features selected: {n_features_selected.min():.0f} to {n_features_selected.max():.0f}")

# ======================================================================
# 4. Cross-Validated Lambda Selection
# ======================================================================

print("\n" + "="*70)
print("CROSS-VALIDATED LAMBDA SELECTION")
print("="*70)

# Use LassoCV for automatic CV-based lambda selection
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

print(f"\nOptimal lambda (via 5-fold CV):")
print(f"  Lambda: {lasso_cv.alpha_:.4f}")
print(f"  CV MSE: {lasso_cv.mse_path_.mean(axis=1)[np.argmin(lasso_cv.mse_path_.mean(axis=1))]:,.0f}")

# Get predictions and metrics for optimal lambda
lasso_pred = lasso_cv.predict(X_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_pred))
lasso_r2 = r2_score(y, lasso_pred)
lasso_mae = mean_absolute_error(y, lasso_pred)

n_nonzero_lasso = np.sum(np.abs(lasso_cv.coef_) > 1e-8)

print(f"\nLasso Model Performance (Optimal Lambda):")
print(f"  R²:   {lasso_r2:.4f}")
print(f"  RMSE: ${lasso_rmse:,.0f}")
print(f"  MAE:  ${lasso_mae:,.0f}")
print(f"  Non-zero coefficients: {n_nonzero_lasso}/{len(X.columns)}")

# Feature importance
lasso_features = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_cv.coef_,
    'AbsCoef': np.abs(lasso_cv.coef_)
}).sort_values('AbsCoef', ascending=False)

print(f"\nLasso Coefficients (sorted by absolute value):")
print(lasso_features[['Feature', 'Coefficient']].to_string(index=False))

# ======================================================================
# 5. Comparison: OLS vs Lasso vs Ridge
# ======================================================================

print("\n" + "="*70)
print("MODEL COMPARISON: OLS vs LASSO vs RIDGE")
print("="*70)

# Ridge regression
ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=5)
ridge_cv.fit(X_scaled, y)

ridge_pred = ridge_cv.predict(X_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
ridge_r2 = r2_score(y, ridge_pred)
ridge_mae = mean_absolute_error(y, ridge_pred)
ridge_lambda = ridge_cv.alpha_

# Comparison table
comparison = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso'],
    'Lambda': [0, ridge_lambda, lasso_cv.alpha_],
    'R²': [ols_r2, ridge_r2, lasso_r2],
    'RMSE': [ols_rmse, ridge_rmse, lasso_rmse],
    'MAE': [ols_mae, ridge_mae, lasso_mae],
    'Non-zero Coef': [n_nonzero_ols, len(X.columns), n_nonzero_lasso]
})

print(f"\nPerformance Summary (In-sample):")
print(comparison.to_string(index=False))

# Cross-validation performance
print(f"\nCross-Validation Performance (5-fold CV):")
ols_cv_scores = cross_val_score(LinearRegression(), X_scaled, y, cv=5,
                                 scoring='neg_mean_squared_error')
print(f"  OLS CV RMSE: ${np.sqrt(-ols_cv_scores.mean()):,.0f} (+/- ${np.sqrt(ols_cv_scores.std()):,.0f})")
print(f"  Ridge CV RMSE: ${np.sqrt(ridge_cv.cv_results_['mean_test_score'].max()):,.0f}")

# Lasso CV is already computed
lasso_cv_rmse = np.sqrt(lasso_cv.mse_path_.mean(axis=1).min())
print(f"  Lasso CV RMSE: ${lasso_cv_rmse:,.0f}")

# ======================================================================
# 6. Visualizations
# ======================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# --- Plot 1: Regularization Path ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Coefficients vs Lambda
for i, feat in enumerate(X.columns):
    ax1.plot(np.log10(alphas), lasso_coefs[:, i], label=feat if i < 5 else '', alpha=0.8)

ax1.axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--', linewidth=2,
           label=f'Optimal λ={lasso_cv.alpha_:.4f}')
ax1.set_xlabel('log₁₀(Lambda)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Lasso Regularization Path: Coefficients vs Lambda', fontsize=13)
ax1.legend(fontsize=8, loc='best', ncol=2)
ax1.grid(True, alpha=0.3)

# Right: Number of selected features vs Lambda
n_selected = (np.abs(lasso_coefs) > 1e-8).sum(axis=1)
ax2.plot(np.log10(alphas), n_selected, 'o-', linewidth=2, markersize=6)
ax2.axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--', linewidth=2,
           label=f'Optimal λ')
ax2.set_xlabel('log₁₀(Lambda)', fontsize=12)
ax2.set_ylabel('Number of Non-Zero Coefficients', fontsize=12)
ax2.set_title('Feature Selection: Active Features vs Lambda', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 2: Cross-Validation Error ---
fig, ax = plt.subplots(figsize=(10, 6))

# CV mean squared error
cv_mse_mean = lasso_cv.mse_path_.mean(axis=1)
cv_mse_std = lasso_cv.mse_path_.std(axis=1)

ax.plot(np.log10(alphas), np.sqrt(cv_mse_mean), 'o-', linewidth=2, markersize=6,
       label='Mean CV RMSE', color='blue')
ax.fill_between(np.log10(alphas),
                np.sqrt(cv_mse_mean - cv_mse_std),
                np.sqrt(cv_mse_mean + cv_mse_std),
                alpha=0.2, color='blue', label='±1 Std Dev')
ax.axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--', linewidth=2,
          label=f'Optimal λ={lasso_cv.alpha_:.4f}')
ax.set_xlabel('log₁₀(Lambda)', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Cross-Validation Error: Lambda Selection', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 3: Model Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Model performance
models = ['OLS', 'Ridge', 'Lasso']
rmses = [ols_rmse, ridge_rmse, lasso_rmse]
r2s = [ols_r2, ridge_r2, lasso_r2]

x_pos = np.arange(len(models))
ax = axes[0]
ax.bar(x_pos, rmses, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('RMSE ($)', fontsize=12)
ax.set_title('Model Comparison: RMSE', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(rmses):
    ax.text(i, v + 10000, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)

ax = axes[1]
ax.bar(x_pos, r2s, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Model Comparison: R²', fontsize=13)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(r2s):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# --- Plot 4: Predicted vs Actual ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, model_name, y_pred in zip(axes, ['OLS', 'Ridge', 'Lasso'],
                                   [ols_pred, ridge_pred, lasso_pred]):
    ax.scatter(y, y_pred, alpha=0.3, s=10)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price ($)', fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontsize=11)
    r2 = r2_score(y, y_pred)
    ax.set_title(f'{model_name}: R² = {r2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ======================================================================
# 7. Feature Importance Analysis
# ======================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE: LASSO vs OLS")
print("="*70)

# Create comparison dataframe
feature_comparison = pd.DataFrame({
    'Feature': X.columns,
    'OLS': ols_model.coef_,
    'Lasso': lasso_cv.coef_,
    'Lasso_Selected': (np.abs(lasso_cv.coef_) > 1e-8).astype(int)
})

print(f"\nFeatures Selected by Lasso: {feature_comparison[feature_comparison['Lasso_Selected'] == 1]['Feature'].tolist()}")
print(f"Features Eliminated by Lasso: {feature_comparison[feature_comparison['Lasso_Selected'] == 0]['Feature'].tolist()}")

# ======================================================================
# 8. Summary Report
# ======================================================================

print("\n" + "="*70)
print("LASSO REGRESSION SUMMARY REPORT")
print("="*70)

summary = f"""
Dataset: King County Housing Prices
  Observations: {len(X)}
  Features: {X.shape[1]}
  Target: Adjusted Sale Price

OLS Model:
  R²: {ols_r2:.4f}
  RMSE: ${ols_rmse:,.0f}
  All {n_nonzero_ols} features included

Lasso Model (Optimal λ={lasso_cv.alpha_:.4f}):
  R²: {lasso_r2:.4f} (change: {(lasso_r2-ols_r2):+.4f})
  RMSE: ${lasso_rmse:,.0f} (change: ${(lasso_rmse-ols_rmse):+,.0f})
  Features selected: {n_nonzero_lasso}/{len(X.columns)}
  Features eliminated: {len(X.columns) - n_nonzero_lasso}

Top 5 Lasso Features:
"""

for idx, row in lasso_features.head(5).iterrows():
    summary += f"  {row['Feature']}: {row['Coefficient']:+.0f}\n"

summary += f"""
Ridge Model (Optimal λ={ridge_lambda:.4f}):
  R²: {ridge_r2:.4f}
  RMSE: ${ridge_rmse:,.0f}
  All {len(X.columns)} features retained with shrinkage

Interpretation:
  - Lasso achieved similar R² while selecting {n_nonzero_lasso} features
  - Lasso provides interpretable feature selection
  - Eliminated features may be collinear or non-predictive
  - Ridge retains all features with weight shrinkage
"""

print(summary)

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
