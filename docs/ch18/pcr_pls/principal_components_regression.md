# Principal Components Regression (PCR)

## Overview

**Principal Components Regression (PCR)** combines dimensionality reduction with regression. Instead of regressing the response directly on all predictors, PCR first extracts principal components (linear combinations of predictors that capture most variance) and then regresses the response on these components.

PCR is particularly valuable when:
- **Multicollinearity** is severe (high correlation among predictors)
- **$p$ is large** relative to $n$ (many more predictors than observations)
- **Interpretability** is less critical than prediction accuracy

---

## The PCR Algorithm

### Step 1: Standardize Predictors

Standardize each predictor to have mean 0 and standard deviation 1:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$

This is essential because PCA is sensitive to scale. Without standardization, predictors with large variances dominate the components.

### Step 2: Compute Principal Components

Apply PCA to the standardized predictors $X_{\text{scaled}}$ to compute principal components:

$$Z_k = X_{\text{scaled}} V_k$$

where:
- $V_k$ is the matrix of eigenvectors (loadings) of the covariance matrix
- $Z_k$ is the matrix of the first $k$ principal components
- Each principal component is a linear combination: $Z_j = \sum_{i=1}^p v_{ij} X_i$

The components are ordered by the variance they explain:
$$\text{Var}(Z_1) \geq \text{Var}(Z_2) \geq \cdots \geq \text{Var}(Z_p)$$

### Step 3: Regress on Principal Components

Perform standard linear regression using the first $M$ principal components as predictors:

$$y = \beta_0 + \beta_1 Z_1 + \beta_2 Z_2 + \cdots + \beta_M Z_M + \epsilon$$

where $M \leq p$ is selected by cross-validation (Step 4).

### Step 4: Choose $M$ via Cross-Validation

The number of components $M$ is a tuning parameter:

- **Too few components** ($M$ small): Underfitting; lose information from excluded predictors
- **Too many components** ($M$ close to $p$): Overfitting; noisy components increase variance
- **Optimal $M^*$**: Minimizes cross-validation error

Use **$k$-fold cross-validation**:

1. For each candidate $M \in \{1, 2, \ldots, p\}$:
   - For each fold:
     - Fit PCA on training fold (compute components)
     - Fit regression on first $M$ components
     - Predict on validation fold
   - Compute average CV error

2. Select $\hat{M} = \arg\min_M \text{CV}(M)$

---

## Advantages and Disadvantages

### Advantages

1. **Handles multicollinearity** — Uncorrelated components eliminate multicollinearity problems
2. **Works when $p > n$** — Dimensionality reduction makes regression feasible
3. **Automatic feature combination** — Components are data-driven linear combinations of all predictors
4. **Computational efficiency** — Solving least squares on $M < p$ predictors is faster than alternatives
5. **Reduces overfitting** — Using fewer components acts as implicit regularization

### Disadvantages

1. **Unsupervised dimension reduction** — PCA ignores the response $y$; components may not align with predicting $y$
   - **Contrast with PLS**: Partial Least Squares uses the response to guide component construction
2. **Loss of interpretability** — Components are linear combinations of original predictors; harder to interpret
3. **Standardization required** — Must standardize predictors; predictions can be sensitive to scaling choices
4. **Model complexity** — Must store the loading matrix $V$ to apply model to new data
5. **Not for feature selection** — All original features may be used, even if only a few truly matter

---

## PCR vs. Ridge Regression

Both PCR and Ridge regression address multicollinearity, but differ fundamentally:

| Aspect | PCR | Ridge |
|--------|-----|-------|
| **Approach** | Unsupervised dimension reduction (PCA) | Shrinkage of all coefficients |
| **Components retained** | Only first $M$ components | All predictors, shrunk |
| **Parameter** | Number of components $M$ | Regularization strength $\lambda$ |
| **Bias-variance** | Drops components (bias), retains $M$ (variance) | Shrinks all coefficients (bias ↑, variance ↓) |
| **When to use** | $p$ large, severe multicollinearity, $p > n$ | Moderate multicollinearity, moderate $p$ |

**Key insight**: Ridge uses a continuous shrinkage mechanism, while PCR uses a discrete selection mechanism.

---

## PCR vs. Partial Least Squares (PLS)

Both are dimensionality reduction methods for regression, but differ in how they construct components:

| Aspect | PCR | PLS |
|--------|-----|-----|
| **Component construction** | Unsupervised (PCA): maximize variance of $X$ | Supervised: maximize covariance of $X$ and $y$ |
| **Components aligned with** | Explaining variance in predictors | Predicting the response |
| **Typical performance** | Depends on PCA alignment with $y$ | Often better when components should predict $y$ |
| **Interpretability** | Same limitation: linear combinations of $X$ | Same limitation: linear combinations of $X$ |

In practice, PLS often outperforms PCR because it uses information about $y$ when building components.

---

## Mathematical Details

### Variance Explained

The proportion of variance explained by the first $k$ principal components is:

$$\frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{p} \lambda_j}$$

where $\lambda_j$ are the eigenvalues of the covariance matrix $\text{Cov}(X_{\text{scaled}})$, ordered from largest to smallest.

A scree plot visualizes this: it shows eigenvalues (or cumulative variance explained) vs. component number. An "elbow" indicates the number of components capturing most variation.

### Regression Coefficients in Original Scale

PCR estimates coefficients in terms of principal components:

$$\hat{\beta}_{\text{PCR}} = V_M \hat{\gamma}$$

where:
- $V_M$ is the $p \times M$ matrix of loadings for the first $M$ components
- $\hat{\gamma}$ is the regression coefficients on the components

To make predictions on new data with original features $x_{\text{new}}$:

$$\hat{y}_{\text{new}} = \hat{\beta}_0 + x_{\text{new}}^T \hat{\beta}_{\text{PCR}}$$

---

## Python Implementation

### Step-by-Step Example

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Load data
X = pd.DataFrame(...)  # Features
y = pd.Series(...)     # Response

# Step 1: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Check variance explained
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"Variance explained by each component:\n{pca.explained_variance_ratio_}")
print(f"Cumulative variance:\n{cumsum_var}")

# Step 3 & 4: Choose M via cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = []

for M in range(1, X_scaled.shape[1] + 1):
    # Fit regression on first M components
    reg = LinearRegression()
    cv_score = cross_val_score(
        reg, X_pca[:, :M], y,
        cv=kfold,
        scoring='neg_mean_squared_error'
    )
    mse = -cv_score.mean()
    mse_scores.append(mse)
    print(f"M={M:2d}: CV MSE = {mse:,.0f}")

# Find optimal M
M_opt = np.argmin(mse_scores) + 1
print(f"\nOptimal number of components: M = {M_opt}")
print(f"Variance explained: {cumsum_var[M_opt-1]:.4f} ({cumsum_var[M_opt-1]*100:.2f}%)")

# Fit final model
pcr_model = LinearRegression()
pcr_model.fit(X_pca[:, :M_opt], y)

# Make predictions
y_pred = pcr_model.predict(X_pca[:, :M_opt])
```

### Visualization Example

```python
# Plot 1: Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Variance explained by each component
ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'o-', linewidth=2, markersize=6)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained')
ax1.set_title('Scree Plot')
ax1.grid(True, alpha=0.3)

# Cumulative variance
ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', linewidth=2, markersize=6)
ax2.axhline(0.9, color='red', linestyle='--', label='90% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained')
ax2.set_title('Cumulative Variance Explained')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 2: CV error vs number of components
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(mse_scores) + 1), np.sqrt(mse_scores), 'o-', linewidth=2, markersize=6)
ax.axvline(M_opt, color='red', linestyle='--', label=f'Optimal M = {M_opt}')
ax.set_xlabel('Number of Components (M)')
ax.set_ylabel('CV RMSE')
ax.set_title('Cross-Validation Error vs Number of Components')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## When to Use PCR

PCR is a good choice when:

1. **Multicollinearity is severe** — Components are uncorrelated
2. **$p$ is very large** — Dimensionality reduction needed
3. **$p > n$** — OLS is infeasible; PCR makes regression possible
4. **Interpretability of original features is not critical** — Willing to work with components
5. **Prediction accuracy is the primary goal** — Doesn't require understanding individual features

Consider alternatives if:
- **Feature selection is important** — Use Lasso or elastic net instead
- **Interpretability is critical** — Linear regression with a subset of features may be preferable
- **PLS might work better** — If the goal is prediction (PLS uses the response in component construction)

---

## Comparison with Other Methods

| Method | Multicollinearity | Feature Selection | Interpretability | When to Use |
|--------|------------------|-------------------|-----------------|-------------|
| **OLS** | Poor | No | High | Small $p$, low correlation |
| **Ridge** | Good | No | High | Moderate $p$, moderate correlation |
| **Lasso** | Good | Yes | High | Feature selection important |
| **Elastic Net** | Good | Yes | High | Balance of Ridge + Lasso |
| **PCR** | Excellent | No | Low | Large $p$ or $p > n$ |
| **PLS** | Excellent | No | Low | Large $p$, prediction focus |

---

## Summary

Principal Components Regression combines the unsupervised dimensionality reduction of PCA with linear regression:

1. **Standardize** predictors
2. **Extract** principal components (linear combinations of predictors)
3. **Select** the number of components $M$ via cross-validation
4. **Regress** response on the first $M$ components

PCR effectively addresses multicollinearity and high-dimensionality, making it valuable for prediction in challenging settings where $p$ is large or correlation among predictors is severe. However, the loss of interpretability and unsupervised nature of component selection are trade-offs to consider.
