# Partial Least Squares (PLS)

## Overview

**Partial Least Squares (PLS)** is a dimensionality reduction method that, unlike PCA (used in PCR), constructs components by considering the relationship between predictors and the response. PLS finds linear combinations of predictors that are highly correlated with the response, making it particularly useful when prediction is the goal.

### Key Distinction from PCR

| Aspect | PCR | PLS |
|--------|-----|-----|
| **Component construction** | Unsupervised: maximize variance in $X$ | Supervised: maximize covariance of $X$ with $y$ |
| **Objective** | Explain variance in predictors | Explain variance in predictors **and** correlation with response |
| **When it excels** | When predictor variance aligns with response | When small number of predictors drive response |
| **Typical outcome** | Often requires many components | Often fewer components than PCR |

PLS is valuable when:
- **Prediction accuracy** is the primary goal
- **Multicollinearity** among predictors is severe
- **$p$ is large relative to $n$** ($p > n$ or $p \approx n$)
- The response depends on a **small number of latent directions** in predictor space

---

## The PLS Algorithm

### Step 1: Standardize Predictors and Response

Standardize both predictors and response:

$$X_{\text{scaled}} = \frac{X - \mu_X}{\sigma_X}, \quad y_{\text{centered}} = y - \bar{y}$$

This ensures all variables are on comparable scales and simplifies interpretation.

### Step 2: Construct Latent Components Iteratively

Unlike PCR, which extracts all principal components at once, PLS constructs components sequentially by maximizing the covariance between $X$ and $y$.

**First Component ($T_1, U_1$):**

The first PLS component is the direction in $X$ space most correlated with $y$:

$$T_1 = X_{\text{scaled}} w_1$$

where the weight vector $w_1$ maximizes the covariance:

$$w_1 = \arg\max_w \text{Cov}(X_{\text{scaled}} w, y_{\text{centered}})$$

subject to $\|w_1\| = 1$ (unit norm constraint).

In practice, $w_1$ is proportional to $X_{\text{scaled}}^T y_{\text{centered}}$ (the correlation between each predictor and response).

**Subsequent Components ($T_m, U_m$ for $m > 1$):**

1. Regress $X$ on the current component $T_{m-1}$: compute residuals $X^{(m)} = X - \hat{X}$
2. Regress $y$ on $T_{m-1}$: compute residuals $y^{(m)} = y - \hat{y}$
3. Construct $w_m$ to maximize covariance between residual $X^{(m)}$ and residual $y^{(m)}$
4. Form new component: $T_m = X^{(m)} w_m$

This iterative deflation ensures components capture variance in $y$ not explained by previous components.

### Step 3: Regress on PLS Components

Perform regression on the first $M$ PLS components:

$$y = \beta_0 + \beta_1 T_1 + \beta_2 T_2 + \cdots + \beta_M T_M + \epsilon$$

### Step 4: Select Number of Components via Cross-Validation

Use **$k$-fold cross-validation** to choose the optimal number of components $M$:

1. For each $M \in \{1, 2, \ldots, p\}$:
   - For each fold: fit PLS with $M$ components on training data
   - Predict on validation fold and record error
   - Average prediction error across folds

2. Select $\hat{M} = \arg\min_M \text{CV}(M)$

---

## Mathematical Details

### PLS Weight Vector

The weight vector for the $m$-th component solves:

$$w_m = \frac{X_{\text{residual}}^T y_{\text{residual}}}{\|X_{\text{residual}}^T y_{\text{residual}}\|}$$

where the residuals are orthogonal to all previous components. This differs fundamentally from PCA, which uses the eigenvectors of the covariance matrix.

### NIPALS Algorithm

The **Non-linear Iterative Partial Least Squares (NIPALS)** algorithm is the standard computational approach:

```
1. Initialize: X̃ = X (scaled), ỹ = y (centered)
2. For m = 1 to M:
   a. w_m = X̃'ỹ / ||X̃'ỹ||        (compute weight)
   b. t_m = X̃ w_m                   (compute component)
   c. β_m = (t_m' ỹ) / (t_m' t_m)   (regress y on t_m)
   d. ỹ = ỹ - β_m t_m               (deflate y residuals)
   e. p_m = X̃' t_m / (t_m' t_m)    (compute loading)
   f. X̃ = X̃ - t_m p_m'             (deflate X residuals)
3. β_global = [β_1, β_2, ..., β_M]
```

---

## Advantages and Disadvantages

### Advantages

1. **Supervised dimensionality reduction** — Components are constructed to predict $y$, not just explain variance in $X$
2. **Fewer components needed** — Often requires fewer components than PCR because components are selected based on predictive power
3. **Handles high-dimensionality** — Works when $p > n$ without the need for variable selection
4. **Handles multicollinearity** — Components are uncorrelated, eliminating multicollinearity issues
5. **Interpretable loadings** — Component loadings show how original variables contribute
6. **Computational efficiency** — NIPALS algorithm is iterative and computationally efficient

### Disadvantages

1. **Loss of interpretability** — Like PCR, components are linear combinations; harder to interpret than original variables
2. **Standardization sensitivity** — Results sensitive to predictor scaling; standardization is essential
3. **Limited extrapolation** — Predictions unreliable outside training data range
4. **Model complexity** — Must store loadings and weights; not as simple as OLS
5. **Requires tuning** — Must select optimal number of components via cross-validation
6. **Theory less established** — Fewer asymptotic results compared to OLS or Ridge

---

## PLS in Chemometrics and Beyond

PLS originated in chemometrics (spectroscopy analysis) and is widely used when:

- **High-dimensional spectroscopic data** with many wavelengths ($p >> n$)
- **Batch process monitoring** with multiple sensors predicting product quality
- **Drug discovery** with molecular descriptors predicting biological activity
- **Marketing research** with survey responses predicting sales

### Variants

1. **PLS-DA (Discriminant Analysis)** — PLS for classification; constructs components that separate classes
2. **Multi-response PLS** — Extends to multiple responses $Y$ (instead of single $y$)
3. **Orthogonal PLS** — Constructs orthogonal components for improved interpretability

---

## PLS vs. PCR vs. Ridge vs. Lasso

| Method | Dimension Reduction | Component Selection | Use Case |
|--------|-------------------|-------------------|----------|
| **PCR** | Yes (unsupervised) | Top $M$ by variance | When predictor variance important |
| **PLS** | Yes (supervised) | Top $M$ by covariance with $y$ | When prediction goal, $p >> n$ |
| **Ridge** | No (continuous shrinkage) | All predictors, shrunk | Moderate $p$, some multicollinearity |
| **Lasso** | No (sparse shrinkage) | Automatic feature selection | Feature selection important |

**Decision rule:**
- **$p$ large or $p > n$?** → PCR or PLS
- **Multicollinearity but moderate $p$?** → Ridge
- **Feature selection important?** → Lasso or Elastic Net
- **Prediction focus with $p >> n$?** → PLS (usually beats PCR)

---

## Python Implementation

### Step-by-Step Example

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Load data
X = pd.DataFrame(...)  # Features
y = pd.Series(...)     # Response

# Step 1: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2 & 3: Cross-validation for optimal M
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = []

for M in range(1, X_scaled.shape[1] + 1):
    pls = PLSRegression(n_components=M)
    cv_score = cross_val_score(
        pls, X_scaled, y,
        cv=kfold,
        scoring='neg_mean_squared_error'
    )
    mse = -cv_score.mean()
    mse_scores.append(mse)
    print(f"M={M:2d}: CV MSE = {mse:,.0f}")

# Find optimal M
M_opt = np.argmin(mse_scores) + 1
print(f"\nOptimal number of components: M = {M_opt}")

# Step 4: Fit final model
pls_model = PLSRegression(n_components=M_opt)
pls_model.fit(X_scaled, y)

# Make predictions
y_pred = pls_model.predict(X_scaled)

# Examine loadings (importance of each predictor)
loadings = pls_model.x_weights_  # Predictor weights
print(f"PLS loadings (first component):\n{loadings[:, 0]}")
```

### Visualization Example

```python
# Plot: CV error vs number of components
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(mse_scores) + 1), np.sqrt(mse_scores), 'o-',
        linewidth=2, markersize=8)
ax.axvline(M_opt, color='red', linestyle='--', label=f'Optimal M = {M_opt}')
ax.set_xlabel('Number of Components (M)')
ax.set_ylabel('Cross-Validation RMSE')
ax.set_title('PLS: Component Selection via Cross-Validation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Scatterplot: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y, y_pred, alpha=0.5, s=30)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted (PLS)')
ax.set_title(f'PLS Model: M = {M_opt} components')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## When to Use PLS

**Use PLS when:**
1. Prediction accuracy is the primary goal
2. $p$ is large (high-dimensional data)
3. $p > n$ (more predictors than observations)
4. Multicollinearity is severe among predictors
5. You want to avoid feature selection but maintain low complexity
6. Data comes from physical/chemical measurements (PLS's original domain)

**Consider alternatives if:**
- **Interpretability critical** — Linear models may be better
- **$p$ small and multicollinearity moderate** — Ridge or Lasso
- **Feature selection important** — Lasso or Elastic Net
- **Component interpretation essential** — PCA-based methods less suitable

---

## Practical Considerations

### Preprocessing

1. **Standardization is essential** — PLS is covariance-based; scale all predictors and response
2. **Outlier detection** — Outliers can dominate covariance structure
3. **Missing data** — Impute or remove; PLS doesn't handle missingness directly

### Hyperparameter Tuning

- **Number of components**: Always use cross-validation (default: 10-fold)
- **Scaling**: Consider robust scaling if outliers present
- **Centering**: Always center response and predictors

### Model Validation

- **Separate test set**: Report performance on held-out test data, not CV RMSE
- **Residual analysis**: Check for patterns indicating model misspecification
- **Prediction intervals**: Standard errors on predictions for uncertainty quantification

---

## Summary

Partial Least Squares is a powerful supervised dimensionality reduction technique that:

1. **Constructs components** that maximize covariance of predictors with response
2. **Requires fewer components** than PCR because components are selected for prediction
3. **Handles multicollinearity** by creating uncorrelated latent variables
4. **Works in high-dimensional settings** ($p > n$ or $p >> n$)
5. **Balances flexibility and interpretability** between linear models and non-parametric methods

In practice, when the goal is **prediction in high-dimensional settings with multicollinearity**, PLS often outperforms PCR because its supervised component selection aligns better with the prediction objective. However, interpretability trade-offs remain; for transparent, actionable insights on individual predictors, regularized linear methods (Ridge, Lasso) may be preferable despite their performance disadvantages in extreme high-dimensionality.
