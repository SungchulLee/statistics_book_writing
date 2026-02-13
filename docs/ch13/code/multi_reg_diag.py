"""
Multiple Regression and Diagnostics — Complete Implementation

This script demonstrates:
  1. Fitting a multiple linear regression model
  2. Checking multicollinearity (VIF)
  3. Residual analysis (residuals vs fitted, Q-Q plot)
  4. Identifying influential points (Cook's distance, leverage)
  5. Model selection with AIC and BIC
  6. Handling assumption violations (log transformation)
  7. Interaction terms and polynomial regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# 1. Load Dataset
# ============================================================
# Using California Housing (Boston dataset is deprecated)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

print("Dataset shape:", df.shape)
print(df.head())
print()

# ============================================================
# 2. Fit Multiple Linear Regression
# ============================================================
features = ['MedInc', 'AveRooms', 'AveOccup']
X = add_constant(df[features])
y = df['PRICE']

model = OLS(y, X).fit()
print("=" * 60)
print("MULTIPLE LINEAR REGRESSION SUMMARY")
print("=" * 60)
print(model.summary())
print()

# ============================================================
# 3. Multicollinearity — Variance Inflation Factor (VIF)
# ============================================================
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]

print("=" * 60)
print("VARIANCE INFLATION FACTORS")
print("=" * 60)
print(vif_data.to_string(index=False))
print()

# ============================================================
# 4. Residual Analysis
# ============================================================
y_pred = model.predict(X)
residuals = y - y_pred

# 4a. Residuals vs Fitted Values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted Values')

# 4b. Q-Q Plot
sm.qqplot(residuals, line='45', ax=axes[1])
axes[1].set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: residual_analysis.png")
print()

# ============================================================
# 5. Influential Points
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
influence_plot(model, ax=ax, criterion='cooks')
ax.set_title('Influence Plot (Cook\'s Distance)')
plt.tight_layout()
plt.savefig('influence_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: influence_plot.png")
print()

# Cook's Distance summary
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
print(f"Observations with Cook's D > 1: {np.sum(cooks_d > 1)}")
print(f"Observations with Cook's D > 4/n: {np.sum(cooks_d > 4/len(y))}")
print()

# ============================================================
# 6. Model Selection — AIC and BIC
# ============================================================
# Compare models with different feature sets
feature_sets = {
    'Model 1 (MedInc only)': ['MedInc'],
    'Model 2 (MedInc, AveRooms)': ['MedInc', 'AveRooms'],
    'Model 3 (MedInc, AveRooms, AveOccup)': ['MedInc', 'AveRooms', 'AveOccup'],
    'Model 4 (All features)': list(housing.feature_names),
}

print("=" * 60)
print("MODEL COMPARISON — AIC and BIC")
print("=" * 60)

results = []
for name, feats in feature_sets.items():
    X_temp = add_constant(df[feats])
    m = OLS(y, X_temp).fit()
    results.append({
        'Model': name,
        'Predictors': len(feats),
        'R²': f"{m.rsquared:.4f}",
        'Adj. R²': f"{m.rsquared_adj:.4f}",
        'AIC': f"{m.aic:.1f}",
        'BIC': f"{m.bic:.1f}",
    })

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))
print()

# ============================================================
# 7. Handling Assumption Violations — Log Transformation
# ============================================================
# Apply log transformation to the dependent variable
df['LOG_PRICE'] = np.log(df['PRICE'])
y_log = df['LOG_PRICE']

X_base = add_constant(df[features])
model_log = OLS(y_log, X_base).fit()

print("=" * 60)
print("LOG-TRANSFORMED MODEL SUMMARY")
print("=" * 60)
print(model_log.summary())
print()

# ============================================================
# 8. Interaction Terms
# ============================================================
df['MedInc_x_AveRooms'] = df['MedInc'] * df['AveRooms']
features_interact = features + ['MedInc_x_AveRooms']

X_interact = add_constant(df[features_interact])
model_interact = OLS(y, X_interact).fit()

print("=" * 60)
print("MODEL WITH INTERACTION TERM")
print("=" * 60)
print(f"AIC (no interaction):   {model.aic:.1f}")
print(f"AIC (with interaction): {model_interact.aic:.1f}")
print(f"BIC (no interaction):   {model.bic:.1f}")
print(f"BIC (with interaction): {model_interact.bic:.1f}")
print()

# ============================================================
# 9. Polynomial Regression
# ============================================================
X_medinc = df[['MedInc']].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_medinc)
X_poly_sm = add_constant(X_poly)

model_poly = OLS(y, X_poly_sm).fit()

print("=" * 60)
print("POLYNOMIAL REGRESSION (degree=2) — MedInc")
print("=" * 60)
print(model_poly.summary())
print()

# Plot polynomial fit
X_plot = np.linspace(df['MedInc'].min(), df['MedInc'].max(), 200).reshape(-1, 1)
X_plot_poly = add_constant(poly.transform(X_plot))
y_plot = model_poly.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], y, alpha=0.1, s=5, label='Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Quadratic Fit')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Polynomial Regression: Price vs Median Income')
plt.legend()
plt.tight_layout()
plt.savefig('polynomial_fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: polynomial_fit.png")
