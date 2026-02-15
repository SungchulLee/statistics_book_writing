# Regularization Path

## Overview

The **regularization path** traces how model coefficients change as the regularization parameter (lambda, $\lambda$) varies from large (high regularization, sparse solutions) to small (low regularization, close to OLS). Understanding the regularization path is essential for:

1. **Visualizing the bias-variance tradeoff** — See how coefficients shrink as regularization increases
2. **Feature selection** — Identify which features are selected at different regularization levels
3. **Understanding model complexity** — Observe how many features are "active" (non-zero) at each lambda
4. **Choosing the optimal lambda** — Combined with cross-validation, select the best regularization strength

---

## The Regularization Path for Lasso

For Lasso regression, the regularization problem is:

$$\text{minimize} \quad \frac{1}{2n}\|y - X\beta\|^2_2 + \lambda \|\beta\|_1$$

As $\lambda$ increases:
- More coefficients are shrunk exactly to zero
- The model becomes sparser (fewer non-zero coefficients)
- Bias increases, but variance decreases
- Predictive error follows a U-shaped curve (minimum at optimal $\lambda$)

---

## Key Properties of the Lasso Path

### Feature Activation Order

The Lasso path exhibits a **soft-thresholding** property where:

1. At very high $\lambda$, all coefficients are zero
2. As $\lambda$ decreases, features enter the model (become non-zero) one by one
3. The order of entry reflects feature importance under L1 regularization
4. At very small $\lambda$, all features are active (close to OLS solution)

### Monotonicity of Feature Selection

Once a feature becomes non-zero in the Lasso path, it typically remains non-zero as $\lambda$ decreases (with some exceptions for highly correlated predictors). This creates a "homotopy" structure useful for computation.

### Degrees of Freedom

At any point on the path, the effective degrees of freedom (eDoF) is:

$$\text{eDoF}(\lambda) = \text{number of non-zero coefficients}$$

This exact relationship (unlike Ridge regression) makes Lasso particularly useful for model selection.

---

## Computing the Regularization Path

### Algorithm: Coordinate Descent with Warm Starts

Modern implementations use **coordinate descent** with "warm starts":

```
Initialize: β = 0

For λ in decreasing order (λ_max to λ_min):
    Initialize β from previous solution (warm start)
    For each coordinate j:
        Compute residual: r_j = y - X β + X_j β_j
        Apply soft-thresholding: β_j = soft_threshold(X_j^T r_j / n, λ)
    Continue until convergence
```

The warm start leverages the previous solution to speed computation—solutions at nearby lambdas are similar, so we start close to the optimum.

### Complexity

- **Single lambda**: O(np) per iteration
- **Full path** (M lambdas): O(M × np × iterations) if computed sequentially
- **Efficient implementation**: Solutions for all lambdas in nearly the time of solving one problem!

---

## Practical Interpretation of the Path

### Example: Housing Prices

Consider predicting house prices with 12 features. The Lasso path might look like:

| λ (scaled) | # Non-Zero | Selected Features | RMSE |
|---|---|---|---|
| 10.0 | 0 | (none) | 340,000 |
| 5.0 | 1 | SqFtTotLiving | 280,000 |
| 2.0 | 4 | SqFtTotLiving, BldgGrade, YrBuilt, Bathrooms | 240,000 |
| 1.0 | 7 | + SqFtLot, Bedrooms, NbrLivingUnits | 225,000 |
| 0.5 | 10 | + SqFtFinBasement, YrRenovated | 220,000 |
| 0.1 | 12 | All features (approaching OLS) | 218,000 |

**Insights:**
- The most important predictor (SqFtTotLiving) enters first
- Building grade and year built are the next most important
- Less important features (YrRenovated) enter at small lambda
- RMSE continues decreasing, but improvements slow after 7 features

### Cross-Validation on the Path

Rather than optimizing each lambda separately, compute CV error for each $\lambda$ on the precomputed path:

```
For each lambda on the path:
    For each CV fold:
        Fit model on training fold
        Predict on validation fold
        Record error
    Average errors across folds
Select lambda with minimum CV error
```

This is much faster than refitting at each lambda!

---

## Visualization of the Path

### Plot 1: Coefficients vs Lambda

```python
import matplotlib.pyplot as plt
import numpy as np

# After computing lasso_coefs (shape: [n_lambdas, n_features])
fig, ax = plt.subplots(figsize=(10, 6))

for j in range(n_features):
    ax.plot(np.log10(lambdas), lasso_coefs[:, j], label=feature_names[j])

ax.axvline(np.log10(lambda_opt), color='red', linestyle='--', label='Optimal λ')
ax.set_xlabel('log₁₀(Lambda)')
ax.set_ylabel('Coefficient Value')
ax.set_title('Lasso Regularization Path')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

This plot reveals:
- Feature selection: which features are active at each lambda
- Shrinkage direction: how coefficients change
- Sparsity: the order and timing of feature entry

### Plot 2: Cross-Validation Error

```python
# Plot CV error across the path
fig, ax = plt.subplots(figsize=(10, 6))

cv_mean = cv_mse_path.mean(axis=1)
cv_std = cv_mse_path.std(axis=1)

ax.plot(np.log10(lambdas), np.sqrt(cv_mean), 'o-', label='CV RMSE')
ax.fill_between(np.log10(lambdas),
                 np.sqrt(cv_mean - cv_std),
                 np.sqrt(cv_mean + cv_std),
                 alpha=0.2)
ax.axvline(np.log10(lambda_opt), color='red', linestyle='--', label='Optimal λ')
ax.set_xlabel('log₁₀(Lambda)')
ax.set_ylabel('Cross-Validation RMSE')
ax.set_title('Lambda Selection via Cross-Validation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

This reveals the optimal regularization strength: typically a minimum around λ = 1-10 before CV error increases due to excessive regularization.

---

## Comparing Regularization Methods: Paths and Trade-offs

### Lasso vs Ridge vs Elastic Net

| Property | Lasso (L1) | Ridge (L2) | Elastic Net |
|---|---|---|---|
| Path sparsity | Yes (exact zeros) | No (all non-zero) | Partial (grouped zeros) |
| Feature selection | Automatic | Manual (thresholding) | Automatic (grouping) |
| Computational cost | Medium (warm start) | Low (closed form) | Medium |
| High-correlation | Picks one feature | Keeps all, shrinks | Balanced selection |
| Interpretability | High (sparse) | Medium | High |

---

## Practical Guidelines

### Choosing Lambda via Cross-Validation

**Standard approach (1-SE rule):**
1. Compute CV error for each lambda
2. Find λ* with minimum CV error
3. Often use λ = λ_1SE: the largest lambda within 1 standard error of minimum
   - Provides simpler model with similar CV error
   - More conservative against overfitting

```python
# Find lambda_1SE
best_idx = np.argmin(cv_mse)
lambda_1se = lambdas[np.where(cv_mse <= cv_mse[best_idx] + cv_std[best_idx])[0][-1]]
```

### Interpreting the Path for Model Selection

1. **Number of features** — For sparse interpretability, choose lambda where only 5-15 features are active
2. **Stability** — Prefer lambdas where small changes don't drastically alter the model
3. **Domain knowledge** — Features should align with domain understanding; if not, investigate multicollinearity
4. **Prediction vs interpretation** — Higher lambda (fewer features) may sacrifice accuracy for simplicity

### Computational Considerations

- **Modern packages** (scikit-learn, glmnet): Compute path for ~100 lambdas nearly as fast as single fit
- **Warm starts**: Essential for efficiency; don't solve each lambda independently
- **Standardization**: Always standardize features before fitting Lasso; coefficients are not comparable on different scales

---

## Summary

The regularization path provides a complete picture of the bias-variance tradeoff:

- **Visualizes** how coefficients and feature selection change with regularization
- **Enables** efficient computation via warm starts and coordinate descent
- **Guides** lambda selection through cross-validation
- **Supports** interpretability by revealing feature importance and stability
- **Bridges** the gap between complex models (all features) and simple models (few features)

Understanding the regularization path transforms lambda selection from a "black box" into an informed, principled choice aligned with your data and goals.
