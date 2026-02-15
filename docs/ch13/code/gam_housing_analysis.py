#!/usr/bin/env python3
"""
======================================================================
GAM (Generalized Additive Models) Housing Price Analysis
======================================================================

Demonstrates fitting and interpreting Generalized Additive Models for
house price prediction using the King County housing dataset.

Implementations shown:
  1. statsmodels GLMGam with BSplines
  2. pyGAM with automatic lambda selection (grid search)
  3. Visualization of partial dependence plots
  4. Model comparison (GAM vs. Linear vs. Polynomial)

Source: Adapted from "Practical Statistics for Data Scientists"
        (Chapter 4 — Regression and Prediction)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import regression and GAM tools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from pygam import LinearGAM, s, l
from pygam.utils import generate_X_grid

# Set random seed for reproducibility
np.random.seed(42)

# ======================================================================
# 1. Load Data
# ======================================================================

# Adjust path as needed
DATA = Path(__file__).parent.parent.parent / 'data'
HOUSE_CSV = DATA / 'house_sales.csv'

# Load housing data
house = pd.read_csv(HOUSE_CSV, sep='\t')

# Select subset for detailed analysis (Zip Code 98105 for example)
house_98105 = house.loc[house['ZipCode'] == 98105].copy()

print(f"Total samples: {len(house)}")
print(f"Samples in 98105: {len(house_98105)}")
print(f"\nFirst few rows:\n{house_98105[['AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']].head()}")

# ======================================================================
# 2. Fit Models: Linear, Polynomial, and GAM (statsmodels)
# ======================================================================

print("\n" + "="*70)
print("MODEL 1: LINEAR REGRESSION")
print("="*70)

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# Linear model
X_linear = house_98105[predictors].assign(const=1)
model_linear = sm.OLS(house_98105[outcome], X_linear)
result_linear = model_linear.fit()
print(result_linear.summary())

# ------
print("\n" + "="*70)
print("MODEL 2: POLYNOMIAL REGRESSION (degree 2 for SqFtTotLiving)")
print("="*70)

import statsmodels.formula.api as smf

formula_poly = ('AdjSalePrice ~ SqFtTotLiving + np.power(SqFtTotLiving, 2) + ' +
                'SqFtLot + Bathrooms + Bedrooms + BldgGrade')
model_poly = smf.ols(formula=formula_poly, data=house_98105)
result_poly = model_poly.fit()
print(result_poly.summary())

# ------
print("\n" + "="*70)
print("MODEL 3: GAM using statsmodels (GLMGam with BSplines)")
print("="*70)

x_spline = house_98105[predictors]
bs = BSplines(x_spline, df=[10, 3, 3, 3, 3], degree=[3, 2, 2, 2, 2])

# No penalization for comparison (alpha = 0)
alpha = np.array([0] * 5)

formula = ('AdjSalePrice ~ SqFtTotLiving + ' +
           'SqFtLot + Bathrooms + Bedrooms + BldgGrade')

gam_sm = GLMGam.from_formula(formula, data=house_98105, smoother=bs, alpha=alpha)
res_sm = gam_sm.fit()
print(res_sm.summary())

# ------
print("\n" + "="*70)
print("MODEL 4: GAM using pyGAM (with automatic lambda selection)")
print("="*70)

# Prepare data for pyGAM
X_gam = house_98105[predictors].values
y_gam = house_98105[outcome].values

# Fit GAM: smooth spline on SqFtTotLiving, linear terms on others
gam_py = LinearGAM(
    s(0, n_splines=12) +     # SqFtTotLiving: smooth
    l(1) +                   # SqFtLot: linear
    l(2) +                   # Bathrooms: linear
    l(3) +                   # Bedrooms: linear
    l(4)                     # BldgGrade: linear
)

# Grid search for optimal lambda (smoothing parameter)
print("\nPerforming grid search for optimal lambda...")
gam_py.gridsearch(X_gam, y_gam)

print("\nGAM Summary (pyGAM):")
print(gam_py.summary())

# ======================================================================
# 3. Model Comparison
# ======================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Predictions
y_pred_linear = result_linear.fittedvalues
y_pred_poly = result_poly.fittedvalues
y_pred_gam_sm = res_sm.fittedvalues
y_pred_gam_py = gam_py.predict(X_gam)

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'Linear': y_pred_linear,
    'Polynomial': y_pred_poly,
    'GAM (statsmodels)': y_pred_gam_sm,
    'GAM (pyGAM)': y_pred_gam_py
}

comparison = []
for name, pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_gam, pred))
    r2 = r2_score(y_gam, pred)
    comparison.append({
        'Model': name,
        'RMSE': rmse,
        'R²': r2,
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# ======================================================================
# 4. Visualizations
# ======================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# --- Plot 1: Fitted Values Comparison ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(y_gam, y_pred_linear, alpha=0.5, s=20)
axes[0, 0].plot([y_gam.min(), y_gam.max()], [y_gam.min(), y_gam.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'Linear: R² = {r2_score(y_gam, y_pred_linear):.3f}')

axes[0, 1].scatter(y_gam, y_pred_poly, alpha=0.5, s=20)
axes[0, 1].plot([y_gam.min(), y_gam.max()], [y_gam.min(), y_gam.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price')
axes[0, 1].set_ylabel('Predicted Price')
axes[0, 1].set_title(f'Polynomial: R² = {r2_score(y_gam, y_pred_poly):.3f}')

axes[1, 0].scatter(y_gam, y_pred_gam_sm, alpha=0.5, s=20)
axes[1, 0].plot([y_gam.min(), y_gam.max()], [y_gam.min(), y_gam.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Price')
axes[1, 0].set_ylabel('Predicted Price')
axes[1, 0].set_title(f'GAM (statsmodels): R² = {r2_score(y_gam, y_pred_gam_sm):.3f}')

axes[1, 1].scatter(y_gam, y_pred_gam_py, alpha=0.5, s=20)
axes[1, 1].plot([y_gam.min(), y_gam.max()], [y_gam.min(), y_gam.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Price')
axes[1, 1].set_ylabel('Predicted Price')
axes[1, 1].set_title(f'GAM (pyGAM): R² = {r2_score(y_gam, y_pred_gam_py):.3f}')

plt.tight_layout()
plt.show()

# --- Plot 2: Residual Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs Fitted
residuals_gam = y_gam - y_pred_gam_py
axes[0].scatter(y_pred_gam_py, residuals_gam, alpha=0.5, s=20)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('GAM: Residuals vs Fitted')

# Residual histogram
axes[1].hist(residuals_gam, bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('GAM: Residual Distribution')

plt.tight_layout()
plt.show()

# --- Plot 3: Partial Dependence (pyGAM) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

titles = predictors
for i, title in enumerate(titles):
    ax = axes[i]
    XX = gam_py.generate_X_grid(term=i)
    pdep = gam_py.partial_dependence(term=i, X=XX)
    pdep_lower, pdep_upper = gam_py.partial_dependence(term=i, X=XX, width=0.95)[1:]

    ax.plot(XX[:, i], pdep, linewidth=2, label='Mean')
    ax.fill_between(XX[:, i], pdep_lower, pdep_upper, alpha=0.3, label='95% CI')
    ax.set_xlabel(title)
    ax.set_ylabel('Partial Dependence')
    ax.set_title(f'GAM: {title}')
    ax.legend()

# Hide extra subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.show()

# --- Plot 4: statsmodels GAM Partial Residual Plot ---
print("\nGenerating statsmodels GAM partial residual plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, pred in enumerate(predictors):
    ax = axes[i]
    res_sm.plot_partial(i, ax=ax, cpr=True)
    ax.set_title(f'statsmodels GAM: {pred}')

axes[5].set_visible(False)
plt.tight_layout()
plt.show()

# --- Plot 5: Model Complexity (Effective DoF) ---
print("\nModel Effective Degrees of Freedom:")
print(f"  Linear: {len(predictors) + 1}")
print(f"  Polynomial: {len(predictors) + 2}")  # adds one squared term
print(f"  GAM (pyGAM): {gam_py.statistics_['edof']:.2f}")

# ======================================================================
# 6. Prediction Example
# ======================================================================

print("\n" + "="*70)
print("PREDICTION EXAMPLE: House with Specific Characteristics")
print("="*70)

new_house = pd.DataFrame({
    'SqFtTotLiving': [3500],
    'SqFtLot': [12000],
    'Bathrooms': [3.5],
    'Bedrooms': [4],
    'BldgGrade': [10]
})

print(f"\nNew house characteristics:\n{new_house}")

# Predictions
X_new = new_house.assign(const=1)
pred_linear = result_linear.predict(X_new)[0]
pred_poly = result_poly.predict(new_house)[0]
pred_gam_sm = res_sm.predict(X_new)[0]
pred_gam_py = gam_py.predict(new_house[predictors].values)[0]

print(f"\nPredicted prices:")
print(f"  Linear:           ${pred_linear:>12,.0f}")
print(f"  Polynomial:       ${pred_poly:>12,.0f}")
print(f"  GAM (statsmodels):${pred_gam_sm:>12,.0f}")
print(f"  GAM (pyGAM):      ${pred_gam_py:>12,.0f}")

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)
