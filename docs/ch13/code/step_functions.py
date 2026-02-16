#!/usr/bin/env python3
"""
======================================================================
Step Functions: Piecewise Constant Regression
======================================================================

Demonstrates step function regression for non-linear modeling:

  1. Basic step functions: pd.cut() with dummy variables
  2. Comparison with linear, polynomial, and spline models
  3. Multiple predictor visualization
  4. Choosing number and location of breakpoints
  5. Step functions in classification (logistic regression)

Source: Adapted from ISLR (Chapter 7 - Moving Beyond Linearity)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
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
print("STEP FUNCTIONS: PIECEWISE CONSTANT REGRESSION")
print("=" * 70)

# For focused analysis, use age and price
# Create a simple dataset with one continuous predictor
X_raw = house[['YrBuilt']].values.flatten()
y = house['AdjSalePrice'].values

# Convert year built to age
age = 2024 - X_raw  # Approximate current year
df = pd.DataFrame({'age': age, 'price': y})

print(f"\nDataset: {len(df)} housing sales")
print(f"Age range: {age.min():.0f} to {age.max():.0f} years")
print(f"Price range: ${y.min():,.0f} to ${y.max():,.0f}")

# ======================================================================
# 2. Step Functions: Basic Implementation
# ======================================================================

print("\n" + "=" * 70)
print("STEP FUNCTIONS: BASIC IMPLEMENTATION")
print("=" * 70)

# Method 1: Using pd.cut() to create bins
# Define breakpoints (knots) for step function
knots = [0, 20, 40, 60, 80, 150]  # Age breakpoints
df['age_bin'] = pd.cut(df['age'], bins=knots, include_lowest=True,
                       labels=[f'({knots[i]}-{knots[i+1]}]' for i in range(len(knots)-1)])

print(f"\nStep function bins (by age):")
print(df['age_bin'].value_counts().sort_index())

# Create dummy variables for each bin
df_dummies = pd.get_dummies(df['age_bin'], prefix='age_bin', drop_first=False)

print(f"\nDummy variables created: {df_dummies.shape[1]} columns")
print(f"Bin indicator variables:\n{df_dummies.head(10)}")

# ======================================================================
# 3. Step Function Regression
# ======================================================================

print("\n" + "=" * 70)
print("STEP FUNCTION REGRESSION")
print("=" * 70)

# Fit linear regression with step function terms (dummies)
X_step = df_dummies.values
step_model = LinearRegression()
step_model.fit(X_step, df['price'])

step_pred = step_model.predict(X_step)
step_rmse = np.sqrt(mean_squared_error(df['price'], step_pred))
step_r2 = r2_score(df['price'], step_pred)

print(f"\nStep Function Model (5 bins):")
print(f"  R²:   {step_r2:.4f}")
print(f"  RMSE: ${step_rmse:,.0f}")

# Average price in each bin (manually)
bin_means = df.groupby('age_bin')['price'].agg(['mean', 'std', 'count'])
print(f"\nMean price by age bin:")
print(bin_means)

# ======================================================================
# 4. Comparison: Different Numbers of Bins
# ======================================================================

print("\n" + "=" * 70)
print("COMPARISON: DIFFERENT NUMBERS OF BINS")
print("=" * 70)

results = []

for n_bins in [3, 4, 5, 6, 8, 10]:
    # Create bins
    df_temp = df.copy()
    df_temp['age_bin_temp'] = pd.qcut(df_temp['age'], q=n_bins, duplicates='drop')

    # Create dummies
    X_temp = pd.get_dummies(df_temp['age_bin_temp'], drop_first=False).values

    # Fit model
    model = LinearRegression()
    model.fit(X_temp, df['price'])

    pred = model.predict(X_temp)
    rmse = np.sqrt(mean_squared_error(df['price'], pred))
    r2 = r2_score(df['price'], pred)

    results.append({
        'n_bins': n_bins,
        'R²': r2,
        'RMSE': rmse,
        'df': n_bins  # Degrees of freedom (number of bins)
    })

    print(f"n_bins={n_bins:2d}: R² = {r2:.4f}, RMSE = ${rmse:,.0f}")

results_df = pd.DataFrame(results)

# ======================================================================
# 5. Comparison with Other Methods
# ======================================================================

print("\n" + "=" * 70)
print("COMPARISON: STEP vs LINEAR vs POLYNOMIAL vs SPLINE")
print("=" * 70)

# Sort by age for plotting
sort_idx = np.argsort(df['age'].values)
age_sorted = df['age'].values[sort_idx]
price_sorted = df['price'].values[sort_idx]

# 1. Linear model
linear_model = LinearRegression()
X_linear = df[['age']].values
linear_model.fit(X_linear, df['price'])
linear_pred = linear_model.predict(X_linear)
linear_rmse = np.sqrt(mean_squared_error(df['price'], linear_pred))
linear_r2 = r2_score(df['price'], linear_pred)

# 2. Polynomial model (degree 3)
poly_degree = 3
X_poly = np.column_stack([df['age'].values ** i for i in range(1, poly_degree + 1)])
poly_model = LinearRegression()
poly_model.fit(X_poly, df['price'])
poly_pred = poly_model.predict(X_poly)
poly_rmse = np.sqrt(mean_squared_error(df['price'], poly_pred))
poly_r2 = r2_score(df['price'], poly_pred)

# 3. Step function (5 bins)
df_step = df.copy()
df_step['age_bin'] = pd.cut(df_step['age'], bins=5)
X_step_5 = pd.get_dummies(df_step['age_bin'], drop_first=False).values
step_model_5 = LinearRegression()
step_model_5.fit(X_step_5, df['price'])
step_pred_5 = step_model_5.predict(X_step_5)
step_rmse_5 = np.sqrt(mean_squared_error(df['price'], step_pred_5))
step_r2_5 = r2_score(df['price'], step_pred_5)

# Comparison table
comparison = pd.DataFrame({
    'Model': ['Linear', 'Polynomial (deg 3)', 'Step (5 bins)'],
    'R²': [linear_r2, poly_r2, step_r2_5],
    'RMSE': [linear_rmse, poly_rmse, step_rmse_5],
    'Parameters': [1, 3, 5]
})

print(f"\n{comparison.to_string(index=False)}")

# ======================================================================
# 6. Step Functions for Classification
# ======================================================================

print("\n" + "=" * 70)
print("STEP FUNCTIONS FOR CLASSIFICATION")
print("=" * 70)

# Create binary outcome: expensive house (top 25%)
price_threshold = df['price'].quantile(0.75)
df['expensive'] = (df['price'] > price_threshold).astype(int)

print(f"\nBinary outcome: expensive = 1 if price > ${price_threshold:,.0f}")
print(f"  Proportion expensive: {df['expensive'].mean():.2%}")

# Step function logistic regression
df_class = df.copy()
df_class['age_bin'] = pd.cut(df_class['age'], bins=5)
X_class = pd.get_dummies(df_class['age_bin'], drop_first=True).values

logit_model = LogisticRegression(max_iter=1000)
logit_model.fit(X_class, df_class['expensive'])

# Predictions
logit_pred_prob = logit_model.predict_proba(X_class)[:, 1]
logit_pred_class = logit_model.predict(X_class)

# Accuracy
accuracy = (logit_pred_class == df_class['expensive']).mean()
print(f"\nLogistic Regression with Step Functions:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Coefficients: {logit_model.coef_[0]}")

# ======================================================================
# 7. Visualizations
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: Step Function Regression ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1a: 5 bins
ax = axes[0, 0]
ax.scatter(df['age'], df['price'], alpha=0.3, s=20, color='gray', label='Data')

# Get bin means and plot as steps
bin_edges = pd.cut(df['age'], bins=5, retbins=True)[1]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means_vals = df.groupby(pd.cut(df['age'], bins=5))['price'].transform('mean')

# Plot step function
for i in range(len(bin_edges) - 1):
    mask = (df['age'] >= bin_edges[i]) & (df['age'] < bin_edges[i + 1])
    if mask.any():
        y_val = df[mask]['price'].mean()
        ax.hlines(y_val, bin_edges[i], bin_edges[i + 1], colors='red', linewidth=2.5)

ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'Step Function (5 bins): R² = {step_r2_5:.4f}', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 1b: Different numbers of bins
ax = axes[0, 1]
ax.plot(results_df['n_bins'], results_df['R²'], 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Bins', fontsize=11)
ax.set_ylabel('R²', fontsize=11)
ax.set_title('R² vs Number of Bins', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 1c: Comparison with other models
ax = axes[1, 0]
ax.scatter(df['age'], df['price'], alpha=0.3, s=20, color='gray', label='Data')

# Plot Linear
ax.plot(age_sorted, linear_model.predict(df[['age']].values[sort_idx]),
       label=f'Linear (R²={linear_r2:.3f})', linewidth=2, color='blue')

# Plot Polynomial
ax.plot(age_sorted, poly_model.predict(np.column_stack([age_sorted ** i for i in range(1, poly_degree + 1)])),
       label=f'Polynomial deg 3 (R²={poly_r2:.3f})', linewidth=2, color='orange')

# Plot Step
for i in range(len(bin_edges) - 1):
    mask = (df['age'] >= bin_edges[i]) & (df['age'] < bin_edges[i + 1])
    if mask.any():
        y_val = df[mask]['price'].mean()
        ax.hlines(y_val, bin_edges[i], bin_edges[i + 1], colors='red', linewidth=2.5)

ax.plot([], [], 'r-', linewidth=2.5, label=f'Step (R²={step_r2_5:.3f})')

ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Comparison: Different Model Types', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 1d: Classification with step functions
ax = axes[1, 1]
age_unique = np.sort(df['age'].unique())

# Predict probability for range of ages
X_pred = pd.DataFrame({'age': age_unique})
X_pred['age_bin'] = pd.cut(X_pred['age'], bins=5)
X_pred_dummy = pd.get_dummies(X_pred['age_bin'], drop_first=True).values

# Align with training data categories
# For prediction, we need to be careful about bin assignments
actual_probs = []
for age_val in age_unique:
    # Find bin for this age value
    bin_idx = np.searchsorted(bin_edges[1:], age_val)
    if bin_idx < len(bin_edges) - 1:
        # Get mean probability for that bin
        mask = (df['age'] >= bin_edges[bin_idx]) & (df['age'] < bin_edges[bin_idx + 1])
        if mask.any():
            actual_probs.append(df[mask]['expensive'].mean())

if actual_probs:
    ax.plot(age_unique[:len(actual_probs)], actual_probs, 'o-', linewidth=2, markersize=6,
           label='Actual', color='green')

# Plot logistic prediction
ax.scatter(df['age'], df['expensive'], alpha=0.15, s=20, color='gray', label='Data (jittered)')
ax.plot(age_unique, logit_model.predict_proba(X_pred_dummy if X_pred_dummy.shape[0] > 0 else
                                               np.zeros((len(age_unique), logit_model.n_features_in_)))[:, 1]
        if X_pred_dummy.shape[0] > 0 else [], 'r--', linewidth=2, label='Logistic Prediction')

ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Probability of Expensive', fontsize=11)
ax.set_title(f'Classification: Step Function Logistic (Acc={accuracy:.3f})', fontsize=12)
ax.set_ylim([-0.1, 1.1])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Plot 2: Number of Bins Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# R² vs number of bins
ax1.plot(results_df['n_bins'], results_df['R²'], 'o-', linewidth=2.5, markersize=8)
ax1.set_xlabel('Number of Bins', fontsize=12)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('Model Fit vs Number of Bins', fontsize=13)
ax1.grid(True, alpha=0.3)

# RMSE vs number of bins
ax2.plot(results_df['n_bins'], results_df['RMSE'], 'o-', linewidth=2.5, markersize=8, color='orange')
ax2.set_xlabel('Number of Bins', fontsize=12)
ax2.set_ylabel('RMSE ($)', fontsize=12)
ax2.set_title('Prediction Error vs Number of Bins', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ======================================================================
# 8. Summary Report
# ======================================================================

print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

summary = f"""
Step Functions: Piecewise Constant Regression
==============================================

Dataset: King County Housing Prices
  Observations: {len(df)}
  Predictor: Age (years)
  Response: Adjusted Sale Price

Step Function Model (5 bins):
  R²: {step_r2_5:.4f}
  RMSE: ${step_rmse_5:,.0f}
  Parameters: 5 (one for each bin)

Comparison with other methods:
  Linear:        R² = {linear_r2:.4f} (1 parameter)
  Polynomial:    R² = {poly_r2:.4f} (3 parameters)
  Step function: R² = {step_r2_5:.4f} (5 parameters)

Key Insights:
  1. Step functions are piecewise constant functions
  2. Created using pd.cut() with dummy variables
  3. Simple to interpret: average price per age range
  4. Flexible: can capture non-linear relationships
  5. Trade-off: discontinuous at bin boundaries
  6. Advantages over splines: simpler, fewer parameters
  7. Disadvantages: less smooth, larger approximation error

Classification Example:
  Predicting if house is "expensive" (top 25% by price)
  Logistic regression with step functions: {accuracy:.1%} accuracy

When to use step functions:
  - Non-linear relationships with clear regime changes
  - Interpretability is important (bin means are intuitive)
  - Simplicity preferred over smoothness
  - When breakpoints have domain knowledge meaning

Common alternatives:
  - Polynomial regression: smoother but global shape
  - Splines: smoother with more parameters
  - GAMs: flexible with automatic smoothness
"""

print(summary)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
