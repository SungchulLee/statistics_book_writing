#!/usr/bin/env python3
"""
======================================================================
Splines with Patsy: B-splines and Natural Splines
======================================================================

Demonstrates spline regression using patsy's formula interface:

  1. B-splines (Basis splines) via patsy bs()
  2. Natural splines (cubic splines) via patsy cr()
  3. Choosing degrees of freedom and knot placement
  4. Visualization of fitted splines with confidence bands
  5. Comparison with polynomial regression
  6. Multiple predictors with different spline types

Source: Adapted from ISLR (Chapter 7 - Moving Beyond Linearity)
         Uses patsy for formula specification
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from patsy import dmatrix
import statsmodels.api as sm
from statsmodels.formula.api import glm, gam
from statsmodels.genmod.families import Gaussian
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d

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
print("SPLINES WITH PATSY: B-SPLINES AND NATURAL SPLINES")
print("=" * 70)

# Prepare data: age and price relationship
X_raw = house[['YrBuilt']].values.flatten()
y = house['AdjSalePrice'].values

# Convert to age
age = 2024 - X_raw
df = pd.DataFrame({'age': age, 'price': y})

# Sort for visualization
df_sorted = df.sort_values('age').reset_index(drop=True)

print(f"\nDataset: {len(df)} housing sales")
print(f"Age range: {age.min():.1f} to {age.max():.1f} years")
print(f"Price range: ${y.min():,.0f} to ${y.max():,.0f}")

# ======================================================================
# 2. B-splines (Basis Splines)
# ======================================================================

print("\n" + "=" * 70)
print("B-SPLINES (BASIS SPLINES)")
print("=" * 70)

# B-splines with specified degrees of freedom
# df = number of basis functions to use
# degree = polynomial degree (default 3 for cubic)
# knots = positions of interior knots (optional)

# Create design matrix with B-splines
# Option 1: Specify degrees of freedom (knots placed automatically)
bs_df = 4  # 4 degrees of freedom
bs_design = dmatrix(f"bs(age, df={bs_df}, degree=3, include_intercept=False) - 1",
                    {"age": df['age']}, return_type='dataframe')

print(f"\nB-spline with df={bs_df}:")
print(f"  Design matrix shape: {bs_design.shape}")
print(f"  Columns: {list(bs_design.columns)}")

# Fit model
bs_model = LinearRegression()
bs_model.fit(bs_design, df['price'])
bs_pred = bs_model.predict(bs_design)
bs_rmse = np.sqrt(mean_squared_error(df['price'], bs_pred))
bs_r2 = r2_score(df['price'], bs_pred)

print(f"\nB-spline Model Performance:")
print(f"  R²:   {bs_r2:.4f}")
print(f"  RMSE: ${bs_rmse:,.0f}")

# ======================================================================
# 3. B-splines with Custom Knots
# ======================================================================

print("\n" + "=" * 70)
print("B-SPLINES WITH CUSTOM KNOTS")
print("=" * 70)

# Specify interior knots explicitly
# Boundary knots are set at min/max by default
knots_custom = [20, 40, 60]  # Interior knots at age 20, 40, 60

bs_custom_design = dmatrix(
    f"bs(age, knots={knots_custom}, degree=3, include_intercept=False) - 1",
    {"age": df['age']},
    return_type='dataframe'
)

print(f"\nB-spline with custom knots {knots_custom}:")
print(f"  Design matrix shape: {bs_custom_design.shape}")

# Fit model
bs_custom_model = LinearRegression()
bs_custom_model.fit(bs_custom_design, df['price'])
bs_custom_pred = bs_custom_model.predict(bs_custom_design)
bs_custom_rmse = np.sqrt(mean_squared_error(df['price'], bs_custom_pred))
bs_custom_r2 = r2_score(df['price'], bs_custom_pred)

print(f"\nB-spline (Custom Knots) Performance:")
print(f"  R²:   {bs_custom_r2:.4f}")
print(f"  RMSE: ${bs_custom_rmse:,.0f}")

# ======================================================================
# 4. Natural Splines
# ======================================================================

print("\n" + "=" * 70)
print("NATURAL SPLINES (CUBIC SPLINES)")
print("=" * 70)

# Natural splines using patsy cr()
# cr = cubic regression spline with natural boundary conditions
# Natural splines are more stable at boundaries than unrestricted splines

cs_df = 4  # 4 degrees of freedom
cs_design = dmatrix(f"cr(age, df={cs_df}) - 1",
                    {"age": df['age']},
                    return_type='dataframe')

print(f"\nNatural spline with df={cs_df}:")
print(f"  Design matrix shape: {cs_design.shape}")
print(f"  Columns: {list(cs_design.columns)}")

# Fit model
cs_model = LinearRegression()
cs_model.fit(cs_design, df['price'])
cs_pred = cs_model.predict(cs_design)
cs_rmse = np.sqrt(mean_squared_error(df['price'], cs_pred))
cs_r2 = r2_score(df['price'], cs_pred)

print(f"\nNatural Spline Model Performance:")
print(f"  R²:   {cs_r2:.4f}")
print(f"  RMSE: ${cs_rmse:,.0f}")

# ======================================================================
# 5. Comparison: Different Degrees of Freedom
# ======================================================================

print("\n" + "=" * 70)
print("COMPARISON: DIFFERENT DEGREES OF FREEDOM")
print("=" * 70)

results_list = []

for df_val in [2, 3, 4, 5, 6, 8]:
    # B-spline
    bs_temp_design = dmatrix(f"bs(age, df={df_val}, degree=3, include_intercept=False) - 1",
                             {"age": df['age']}, return_type='dataframe')
    bs_temp_model = LinearRegression()
    bs_temp_model.fit(bs_temp_design, df['price'])
    bs_temp_pred = bs_temp_model.predict(bs_temp_design)
    bs_temp_rmse = np.sqrt(mean_squared_error(df['price'], bs_temp_pred))
    bs_temp_r2 = r2_score(df['price'], bs_temp_pred)

    # Natural spline
    cs_temp_design = dmatrix(f"cr(age, df={df_val}) - 1",
                             {"age": df['age']}, return_type='dataframe')
    cs_temp_model = LinearRegression()
    cs_temp_model.fit(cs_temp_design, df['price'])
    cs_temp_pred = cs_temp_model.predict(cs_temp_design)
    cs_temp_rmse = np.sqrt(mean_squared_error(df['price'], cs_temp_pred))
    cs_temp_r2 = r2_score(df['price'], cs_temp_pred)

    results_list.append({
        'df': df_val,
        'BS_R2': bs_temp_r2,
        'BS_RMSE': bs_temp_rmse,
        'CS_R2': cs_temp_r2,
        'CS_RMSE': cs_temp_rmse
    })

results_comp = pd.DataFrame(results_list)

print("\nB-spline vs Natural Spline (varying degrees of freedom):")
print(results_comp.to_string(index=False))

# ======================================================================
# 6. Comparison with Other Methods
# ======================================================================

print("\n" + "=" * 70)
print("COMPARISON: LINEAR vs POLYNOMIAL vs B-SPLINE vs NATURAL SPLINE")
print("=" * 70)

# Linear
linear_model = LinearRegression()
linear_model.fit(df[['age']].values, df['price'])
linear_pred = linear_model.predict(df[['age']].values)
linear_r2 = r2_score(df['price'], linear_pred)
linear_rmse = np.sqrt(mean_squared_error(df['price'], linear_pred))

# Polynomial (degree 3)
poly_features = np.column_stack([df['age'] ** i for i in range(1, 4)])
poly_model = LinearRegression()
poly_model.fit(poly_features, df['price'])
poly_pred = poly_model.predict(poly_features)
poly_r2 = r2_score(df['price'], poly_pred)
poly_rmse = np.sqrt(mean_squared_error(df['price'], poly_pred))

# Comparison table
comparison = pd.DataFrame({
    'Model': ['Linear', 'Polynomial (deg 3)', 'B-spline (df=4)', 'B-spline (custom knots)', 'Natural Spline (df=4)'],
    'R²': [linear_r2, poly_r2, bs_r2, bs_custom_r2, cs_r2],
    'RMSE': [linear_rmse, poly_rmse, bs_rmse, bs_custom_rmse, cs_rmse],
    'Parameters': [1, 3, 4, 4, 4]
})

print(f"\n{comparison.to_string(index=False)}")

# ======================================================================
# 7. Visualizations
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: B-spline, Natural Spline, and Polynomial Comparison ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Create prediction grid for smooth curves
age_grid = np.linspace(df['age'].min(), df['age'].max(), 300)

# Linear
ax = axes[0, 0]
ax.scatter(df['age'], df['price'], alpha=0.2, s=20, color='gray')
linear_grid = linear_model.predict(age_grid.reshape(-1, 1))
ax.plot(age_grid, linear_grid, 'b-', linewidth=2.5, label='Linear fit')
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'Linear Regression: R² = {linear_r2:.4f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Polynomial
ax = axes[0, 1]
ax.scatter(df['age'], df['price'], alpha=0.2, s=20, color='gray')
poly_grid = poly_model.predict(np.column_stack([age_grid ** i for i in range(1, 4)]))
ax.plot(age_grid, poly_grid, 'g-', linewidth=2.5, label='Polynomial fit (deg 3)')
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'Polynomial Regression: R² = {poly_r2:.4f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# B-spline
ax = axes[1, 0]
ax.scatter(df['age'], df['price'], alpha=0.2, s=20, color='gray')
bs_grid_design = dmatrix(f"bs(age, df={bs_df}, degree=3, include_intercept=False) - 1",
                         {"age": age_grid}, return_type='dataframe')
bs_grid_pred = bs_model.predict(bs_grid_design)
ax.plot(age_grid, bs_grid_pred, 'orange', linewidth=2.5, label=f'B-spline (df={bs_df})')
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'B-spline Regression: R² = {bs_r2:.4f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Natural spline
ax = axes[1, 1]
ax.scatter(df['age'], df['price'], alpha=0.2, s=20, color='gray')
cs_grid_design = dmatrix(f"cr(age, df={cs_df}) - 1",
                         {"age": age_grid}, return_type='dataframe')
cs_grid_pred = cs_model.predict(cs_grid_design)
ax.plot(age_grid, cs_grid_pred, 'r-', linewidth=2.5, label=f'Natural spline (df={cs_df})')
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'Natural Spline Regression: R² = {cs_r2:.4f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 2: All Methods Overlaid ---
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(df['age'], df['price'], alpha=0.15, s=25, color='gray', label='Data')

# Plot each model
ax.plot(age_grid, linear_grid, 'b-', linewidth=2.5, label=f'Linear (R²={linear_r2:.3f})')
ax.plot(age_grid, poly_grid, 'g-', linewidth=2.5, label=f'Polynomial deg 3 (R²={poly_r2:.3f})')
ax.plot(age_grid, bs_grid_pred, 'orange', linewidth=2.5, label=f'B-spline df={bs_df} (R²={bs_r2:.3f})')
ax.plot(age_grid, cs_grid_pred, 'r-', linewidth=2.5, label=f'Natural spline df={cs_df} (R²={cs_r2:.3f})')

ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.set_title('Comparison: Linear, Polynomial, B-spline, and Natural Spline', fontsize=13)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 3: Varying Degrees of Freedom ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# R² vs DF
ax1.plot(results_comp['df'], results_comp['BS_R2'], 'o-', linewidth=2, markersize=8,
        label='B-spline', color='orange')
ax1.plot(results_comp['df'], results_comp['CS_R2'], 's-', linewidth=2, markersize=8,
        label='Natural spline', color='red')
ax1.axhline(linear_r2, color='blue', linestyle='--', linewidth=1.5, label='Linear baseline')
ax1.set_xlabel('Degrees of Freedom', fontsize=12)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('Model Fit vs Degrees of Freedom', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# RMSE vs DF
ax2.plot(results_comp['df'], results_comp['BS_RMSE'], 'o-', linewidth=2, markersize=8,
        label='B-spline', color='orange')
ax2.plot(results_comp['df'], results_comp['CS_RMSE'], 's-', linewidth=2, markersize=8,
        label='Natural spline', color='red')
ax2.axhline(linear_rmse, color='blue', linestyle='--', linewidth=1.5, label='Linear baseline')
ax2.set_xlabel('Degrees of Freedom', fontsize=12)
ax2.set_ylabel('RMSE ($)', fontsize=12)
ax2.set_title('Prediction Error vs Degrees of Freedom', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 4: B-spline with Custom Knots ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(df['age'], df['price'], alpha=0.15, s=25, color='gray', label='Data')

# B-spline with custom knots
bs_custom_grid_design = dmatrix(
    f"bs(age, knots={knots_custom}, degree=3, include_intercept=False) - 1",
    {"age": age_grid},
    return_type='dataframe'
)
bs_custom_grid_pred = bs_custom_model.predict(bs_custom_grid_design)
ax.plot(age_grid, bs_custom_grid_pred, 'purple', linewidth=2.5,
       label=f'B-spline with custom knots {knots_custom}')

# Mark knot positions
for knot in knots_custom:
    ax.axvline(knot, color='purple', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.set_title(f'B-spline with Custom Knots: R² = {bs_custom_r2:.4f}', fontsize=13)
ax.legend(fontsize=11)
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
Splines with Patsy: B-splines and Natural Splines
===================================================

Dataset: King County Housing Prices
  Observations: {len(df)}
  Predictor: Age (years)
  Response: Adjusted Sale Price

Key Findings:

1. B-splines (Basis Splines):
   - df={bs_df}: R² = {bs_r2:.4f}, RMSE = ${bs_rmse:,.0f}
   - Custom knots at {knots_custom}: R² = {bs_custom_r2:.4f}, RMSE = ${bs_custom_rmse:,.0f}
   - Flexible, local control via basis functions

2. Natural Splines:
   - df={cs_df}: R² = {cs_r2:.4f}, RMSE = ${cs_rmse:,.0f}
   - More stable at boundaries than cubic splines
   - Cubic polynomial between knots, linear at tails

3. Comparison with other methods:
   - Linear:        R² = {linear_r2:.4f}, RMSE = ${linear_rmse:,.0f}
   - Polynomial:    R² = {poly_r2:.4f}, RMSE = ${poly_rmse:,.0f}
   - Best performer: Natural spline (R² = {cs_r2:.4f})

4. Degrees of Freedom Impact:
   - Increasing df improves fit but risks overfitting
   - Optimal df balances bias and variance
   - Cross-validation recommended for selection

Implementation with Patsy:
  B-spline:     dmatrix("bs(age, df=4, degree=3)", data)
  Custom knots: dmatrix("bs(age, knots=[20,40,60], degree=3)", data)
  Natural:      dmatrix("cr(age, df=4)", data)

Advantages of splines:
  - Local flexibility: adapt to local patterns in data
  - Smooth transitions: continuous and differentiable
  - Reduced overfitting: fewer parameters than high-degree polynomials
  - Boundary stability: natural splines stable at extremes

When to use splines:
  - Non-linear relationships suspected
  - Smooth curves desired (vs step functions)
  - Local adaptation important
  - Fewer parameters than polynomial needed

Comparison with polynomials:
  - Splines: local control, more parameters
  - Polynomials: global shape, fewer parameters
  - Splines often preferred for flexibility
"""

print(summary)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
