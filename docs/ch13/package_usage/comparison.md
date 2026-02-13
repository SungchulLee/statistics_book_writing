# sklearn vs statsmodels Comparison

## Overview

Python offers two primary libraries for regression modeling: **scikit-learn** (`sklearn`) and **statsmodels**. They serve different purposes and are best suited for different workflows.

---

## At a Glance

| Feature | `statsmodels` | `sklearn` |
|---|---|---|
| Primary focus | Statistical inference | Prediction and machine learning |
| Model summary | Detailed (coefficients, $p$-values, $R^2$, AIC, BIC) | Minimal (must compute manually) |
| Hypothesis testing | Built-in ($t$-tests, $F$-tests, Wald tests) | Not available |
| Confidence intervals | Built-in for coefficients and predictions | Not available |
| Diagnostics | Extensive (VIF, influence plots, residual tests) | Limited |
| Cross-validation | Not built-in | Built-in (`cross_val_score`, pipelines) |
| Regularization | Limited | Ridge, Lasso, ElasticNet built-in |
| Intercept handling | Must add manually with `add_constant()` | Automatic (default `fit_intercept=True`) |
| Formula interface | Yes (`smf.ols('y ~ x1 + x2', data=df)`) | No |

---

## When to Use Each

### Use `statsmodels` when:

- You need **coefficient inference**: $p$-values, confidence intervals, significance tests.
- You are performing **model diagnostics**: residual analysis, heteroscedasticity tests, multicollinearity checks.
- You want to compare models using **AIC or BIC**.
- You need a **detailed regression summary table** for reporting.
- You are working in a **classical statistics** or **econometrics** context.

### Use `sklearn` when:

- The primary goal is **prediction** and generalization to new data.
- You need **cross-validation** and **train/test splitting**.
- You are building a **pipeline** with preprocessing steps (scaling, encoding, feature selection).
- You need **regularized models** (Ridge, Lasso, ElasticNet).
- You want a consistent API across different model types (regression, classification, clustering).

---

## Side-by-Side Example

### statsmodels

```python
import statsmodels.api as sm
import pandas as pd

X = sm.add_constant(df[['RM', 'LSTAT', 'PTRATIO']])
y = df['PRICE']

model = sm.OLS(y, X).fit()
print(model.summary())

# Access specific results
print(f"R²: {model.rsquared:.4f}")
print(f"Adj. R²: {model.rsquared_adj:.4f}")
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")
print(model.pvalues)
print(model.conf_int())
```

### sklearn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['PRICE']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(f"R²: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## Using Both Together

In practice, many analysts use both libraries in the same project:

1. **Explore and diagnose** with `statsmodels`: fit an OLS model, examine the summary, check VIF, test residual assumptions.
2. **Predict and validate** with `sklearn`: use cross-validation, regularized models, and pipelines for production-ready prediction.

```python
# Step 1: Statistical analysis with statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_sm = sm.add_constant(df[['RM', 'LSTAT', 'PTRATIO']])
model_sm = sm.OLS(df['PRICE'], X_sm).fit()
print(model_sm.summary())

# Check VIF
vif = pd.DataFrame({
    'Feature': X_sm.columns,
    'VIF': [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
})
print(vif)

# Step 2: Prediction pipeline with sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

X_sk = df[['RM', 'LSTAT', 'PTRATIO']]
cv_scores = cross_val_score(pipe, X_sk, df['PRICE'], cv=5, scoring='r2')
print(f"Ridge CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## Summary

`statsmodels` excels at statistical inference and diagnostics, while `sklearn` excels at prediction and model deployment. The two libraries are complementary, and using both in a regression workflow provides the most complete analysis.
