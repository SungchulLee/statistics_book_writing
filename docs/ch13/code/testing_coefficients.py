"""
Testing Coefficients in Linear Regression
==========================================
Demonstrates hypothesis tests (t-tests), p-values, and confidence intervals
for regression coefficients using statsmodels.
"""

import numpy as np
import statsmodels.api as sm

# ============================================================
# Example 1: Basic Hypothesis Testing for Coefficients
# ============================================================
print("=" * 70)
print("Example 1: Hypothesis Testing for Regression Coefficients")
print("=" * 70)

np.random.seed(0)
X = np.random.rand(100, 2)  # Two independent variables
y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)  # Dependent variable

# Add a constant (intercept)
X_const = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X_const)
results = model.fit()

# Full summary includes t-tests, p-values, and confidence intervals
print(results.summary())

# ============================================================
# Example 2: Extracting p-values and Confidence Intervals
# ============================================================
print("\n" + "=" * 70)
print("Example 2: p-values and Confidence Intervals")
print("=" * 70)

np.random.seed(42)
X = np.random.rand(100, 1)  # Predictor: advertising budget
y = 2.5 * X[:, 0] + np.random.randn(100)  # Dependent variable: sales

X_const = sm.add_constant(X)

model = sm.OLS(y, X_const)
results = model.fit()

print(results.summary())

# Extract p-values and confidence intervals
p_values = results.pvalues
confidence_intervals = results.conf_int()

print("\nP-values for the coefficients:")
print(p_values)

print("\n95% Confidence Intervals for the coefficients:")
print(confidence_intervals)

# ============================================================
# Example 3: Multiple Predictors and Significance Interpretation
# ============================================================
print("\n" + "=" * 70)
print("Example 3: Interpreting Significance with Multiple Predictors")
print("=" * 70)

np.random.seed(42)
study_hours = np.random.rand(100) * 10   # Predictor 1
sleep_hours = np.random.rand(100) * 8    # Predictor 2
exam_scores = (5 + 2.5 * study_hours
               - 1.5 * sleep_hours
               + np.random.randn(100) * 2)  # Outcome

X = np.column_stack((study_hours, sleep_hours))
X_const = sm.add_constant(X)

model = sm.OLS(exam_scores, X_const)
results = model.fit()

print(results.summary())

# Interpretation helper
print("\n--- Significance Interpretation ---")
for i, name in enumerate(["Intercept", "Study Hours", "Sleep Hours"]):
    pval = results.pvalues[i]
    ci = results.conf_int().iloc[i]
    status = "Significant" if pval < 0.05 else "Not Significant"
    print(f"{name}: coef={results.params[i]:.4f}, "
          f"p={pval:.4f} ({status}), "
          f"95% CI=({ci[0]:.4f}, {ci[1]:.4f})")
