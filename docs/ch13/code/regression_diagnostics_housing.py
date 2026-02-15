#!/usr/bin/env python3
"""
======================================================================
Comprehensive Regression Diagnostics: Housing Price Data
======================================================================

Demonstrates thorough diagnostic analysis for linear regression using
the King County housing dataset. Covers:

  1. Studentized residuals for outlier detection
  2. Cook's distance for influence detection
  3. Hat values (leverage) analysis
  4. Comparison of models with/without influential points
  5. Heteroskedasticity diagnosis
  6. Partial residual plots

Source: Adapted from "Practical Statistics for Data Scientists"
        (Chapter 4 — Regression and Prediction)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configure visualization
sns.set_style("whitegrid")
np.random.seed(42)

# ======================================================================
# 1. Load Data and Fit Baseline Model
# ======================================================================

# Adjust path as needed
DATA = Path(__file__).parent.parent.parent / 'data'
HOUSE_CSV = DATA / 'house_sales.csv'

house = pd.read_csv(HOUSE_CSV, sep='\t')

# Focus on Zip Code 98105 for clearer diagnostics
house_98105 = house.loc[house['ZipCode'] == 98105].copy()

print("="*70)
print("REGRESSION DIAGNOSTICS: HOUSING PRICES (ZIP 98105)")
print("="*70)
print(f"\nDataset: {len(house_98105)} properties")
print(f"Features: SqFtTotLiving, SqFtLot, Bathrooms, Bedrooms, BldgGrade")

# Fit baseline model
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

X = house_98105[predictors].assign(const=1)
y = house_98105[outcome]

model = sm.OLS(y, X)
results = model.fit()

print(f"\nBaseline Model R²: {results.rsquared:.4f}")
print(f"Baseline Model RMSE: {np.sqrt(results.mse_resid):.0f}")
print(f"\nFull OLS Summary:\n{results.summary()}")

# ======================================================================
# 2. Outlier and Influence Analysis
# ======================================================================

print("\n" + "="*70)
print("OUTLIER AND INFLUENCE ANALYSIS")
print("="*70)

# Get influence diagnostics
influence = OLSInfluence(results)

# Extract key diagnostics
studentized_resids = influence.resid_studentized_internal
hat_values = influence.hat_matrix_diag
cooks_dist, cooks_pval = influence.cooks_distance

print(f"\nStudentized Residuals Statistics:")
print(f"  Min: {studentized_resids.min():.3f}")
print(f"  Max: {studentized_resids.max():.3f}")
print(f"  Outlier threshold (|t| > 2.5): {(np.abs(studentized_resids) > 2.5).sum()} points")

print(f"\nHat Values (Leverage) Statistics:")
print(f"  Min: {hat_values.min():.4f}")
print(f"  Max: {hat_values.max():.4f}")
print(f"  Mean: {hat_values.mean():.4f}")
print(f"  High leverage (> 3*mean): {(hat_values > 3*hat_values.mean()).sum()} points")

print(f"\nCook's Distance Statistics:")
print(f"  Min: {cooks_dist.min():.4f}")
print(f"  Max: {cooks_dist.max():.4f}")
print(f"  Threshold (4/n = 4/{len(house_98105)} = {4/len(house_98105):.4f}): {(cooks_dist > 4/len(house_98105)).sum()} points")

# --- Identify and examine extreme points ---
print("\n" + "-"*70)
print("Extreme Observations")
print("-"*70)

# Largest studentized residuals
idx_extreme_resid = np.abs(studentized_resids).argsort()[-3:][::-1]
print("\nTop 3 Most Extreme Residuals:")
for idx in idx_extreme_resid:
    print(f"  Index {idx}: Studentized Residual = {studentized_resids.iloc[idx]:.3f}, "
          f"Cook's D = {cooks_dist[idx]:.4f}")
    print(f"    {house_98105.iloc[idx][predictors].to_dict()}")
    print(f"    Actual: ${house_98105.iloc[idx][outcome]:,.0f}, "
          f"Fitted: ${results.fittedvalues.iloc[idx]:,.0f}")

# Largest Cook's distances
idx_influential = np.argsort(cooks_dist)[-3:][::-1]
print("\nTop 3 Most Influential Observations (Cook's D):")
for idx in idx_influential:
    print(f"  Index {idx}: Cook's D = {cooks_dist[idx]:.4f}, "
          f"Studentized Resid = {studentized_resids.iloc[idx]:.3f}")
    print(f"    {house_98105.iloc[idx][predictors].to_dict()}")

# ======================================================================
# 3. Effect of Removing Influential Points
# ======================================================================

print("\n" + "="*70)
print("IMPACT OF REMOVING INFLUENTIAL OBSERVATIONS")
print("="*70)

# Remove points with Cook's distance > 0.08 (chosen threshold)
threshold_cooks = 0.08
mask_keep = cooks_dist < threshold_cooks

print(f"\nRemoving {(~mask_keep).sum()} observations with Cook's D > {threshold_cooks}")

house_filtered = house_98105.loc[mask_keep]
X_filtered = house_filtered[predictors].assign(const=1)
y_filtered = house_filtered[outcome]

model_filtered = sm.OLS(y_filtered, X_filtered)
results_filtered = model_filtered.fit()

# Comparison table
comparison = pd.DataFrame({
    'Original': results.params,
    'Influential Removed': results_filtered.params,
})

print(f"\nParameter Estimates Comparison:")
print(comparison)

# Calculate percent changes
print(f"\nPercent Change in Coefficients:")
pct_change = ((results_filtered.params - results.params) / np.abs(results.params) * 100)
print(pct_change)

# --- Model quality metrics ---
print(f"\nModel Quality Metrics:")
print(f"{'Metric':<20} {'Original':<15} {'Filtered':<15} {'Change':<10}")
print("-"*60)

orig_r2 = results.rsquared
filt_r2 = results_filtered.rsquared
print(f"{'R²':<20} {orig_r2:<15.4f} {filt_r2:<15.4f} {(filt_r2-orig_r2):<10.4f}")

orig_rmse = np.sqrt(results.mse_resid)
filt_rmse = np.sqrt(results_filtered.mse_resid)
print(f"{'RMSE':<20} {orig_rmse:<15.0f} {filt_rmse:<15.0f} {(filt_rmse-orig_rmse):<10.0f}")

orig_aic = results.aic
filt_aic = results_filtered.aic
print(f"{'AIC':<20} {orig_aic:<15.0f} {filt_aic:<15.0f} {(filt_aic-orig_aic):<10.0f}")

# ======================================================================
# 4. Heteroskedasticity Analysis
# ======================================================================

print("\n" + "="*70)
print("HETEROSKEDASTICITY ANALYSIS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Residuals vs Fitted Values
axes[0].scatter(results.fittedvalues, np.abs(results.resid), alpha=0.5, s=30)
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Absolute Residuals')
axes[0].set_title('Heteroskedasticity Check: Abs(Residuals) vs Fitted')
axes[0].grid(True, alpha=0.3)

# Add LOWESS smoothing
sns.regplot(x=results.fittedvalues, y=np.abs(results.resid),
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'},
            lowess=True, ax=axes[1])
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Absolute Residuals')
axes[1].set_title('Heteroskedasticity Check with LOWESS Smoother')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Breusch-Pagan test for heteroskedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(results.resid, X)
print(f"\nBreusch-Pagan Test for Heteroskedasticity:")
print(f"  Test Statistic: {bp_test[0]:.4f}")
print(f"  p-value: {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print(f"  Conclusion: Evidence of heteroskedasticity (p < 0.05)")
else:
    print(f"  Conclusion: No significant heteroskedasticity (p >= 0.05)")

# ======================================================================
# 5. Influential Observations Plot
# ======================================================================

print("\n" + "="*70)
print("INFLUENCE PLOT: Hat Values vs Studentized Residuals")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

# Create influence plot
colors = cooks_dist / cooks_dist.max()  # Color by Cook's distance
scatter = ax.scatter(hat_values, studentized_resids,
                    s=1000*cooks_dist, c=colors, cmap='RdYlBu_r',
                    alpha=0.6, edgecolors='black', linewidth=0.5)

# Add threshold lines
ax.axhline(-2.5, linestyle='--', color='red', alpha=0.5, label='Outlier threshold (±2.5)')
ax.axhline(2.5, linestyle='--', color='red', alpha=0.5)
ax.axvline(2*hat_values.mean(), linestyle='--', color='blue', alpha=0.5,
          label=f'High leverage (2×mean={2*hat_values.mean():.4f})')

ax.set_xlabel('Hat Values (Leverage)', fontsize=12)
ax.set_ylabel('Studentized Residuals', fontsize=12)
ax.set_title('Influence Plot: Bubble Size = Cook\'s Distance', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Cook's Distance (normalized)", fontsize=10)

plt.tight_layout()
plt.show()

# ======================================================================
# 6. Partial Residual Plots
# ======================================================================

print("\n" + "="*70)
print("PARTIAL RESIDUAL PLOTS (CCPR Plots)")
print("="*70)

fig = plt.figure(figsize=(14, 10))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
plt.tight_layout()
plt.show()

# ======================================================================
# 7. Residual Diagnostics
# ======================================================================

print("\n" + "="*70)
print("RESIDUAL DISTRIBUTION ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histogram of residuals
axes[0, 0].hist(results.resid, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Residuals')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Residuals')

# Q-Q plot
sm.qqplot(results.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot: Residuals vs Normal Distribution')

# Residuals over index (time-like ordering)
axes[1, 0].scatter(range(len(results.resid)), results.resid, alpha=0.5, s=20)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Observation Index')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals vs Observation Order')

# ACF (partial autocorrelation)
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(results.resid, lags=30, ax=axes[1, 1])
axes[1, 1].set_title('Partial Autocorrelation of Residuals')

plt.tight_layout()
plt.show()

# Normality test
from scipy import stats

shapiro_stat, shapiro_p = stats.shapiro(results.resid)
print(f"\nShapiro-Wilk Test for Normality:")
print(f"  Test Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print(f"  Conclusion: Residuals deviate from normality (p < 0.05)")
else:
    print(f"  Conclusion: Residuals appear normally distributed (p >= 0.05)")

# ======================================================================
# 8. Summary Report
# ======================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY REPORT")
print("="*70)

summary_report = f"""
Dataset Information:
  Total observations: {len(house_98105)}
  Predictors: {len(predictors)}
  Response: {outcome}

Model Performance:
  R²: {results.rsquared:.4f}
  Adjusted R²: {results.rsquared_adj:.4f}
  RMSE: ${np.sqrt(results.mse_resid):,.0f}
  AIC: {results.aic:.1f}

Outlier Analysis:
  Observations with |studentized resid| > 2.5: {(np.abs(studentized_resids) > 2.5).sum()}
  Observations with high leverage (> 3×mean): {(hat_values > 3*hat_values.mean()).sum()}
  Observations with high influence (Cook's D > 0.08): {(cooks_dist > 0.08).sum()}

Impact of Influential Observations:
  Observations removed: {(~mask_keep).sum()}
  R² change: {filt_r2 - orig_r2:+.4f}
  RMSE change: ${filt_rmse - orig_rmse:+,.0f}
  Largest coefficient change: {pct_change.abs().max():.1f}%

Assumption Checks:
  Breusch-Pagan test p-value: {bp_test[1]:.4f} {'✗ Heteroskedasticity detected' if bp_test[1] < 0.05 else '✓ No heteroskedasticity'}
  Shapiro-Wilk test p-value: {shapiro_p:.4f} {'✗ Non-normality' if shapiro_p < 0.05 else '✓ Normal residuals'}

Recommendations:
  1. Investigate {(~mask_keep).sum()} high-influence points for data quality issues
  2. Consider robust regression if heteroskedasticity is severe
  3. Check if residual non-normality impacts prediction intervals
  4. Consider Box-Cox transformation for the response if needed
"""

print(summary_report)

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
