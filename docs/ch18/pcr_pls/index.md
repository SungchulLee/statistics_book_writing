# Dimensionality Reduction: PCR and PLS

## Overview

When the number of predictors $p$ is large (especially when $p > n$, more predictors than observations), or when severe multicollinearity exists among predictors, coefficient shrinkage methods (Ridge, Lasso) become less efficient. **Dimensionality reduction methods** provide an alternative: instead of shrinking coefficients, they construct a smaller number of latent variables (principal components or PLS components) and regress on these.

## When to Use Dimensionality Reduction

Dimensionality reduction is valuable when:

1. **$p$ is very large** — Many more predictors than observations ($p > n$ or $p >> n$)
2. **Severe multicollinearity** — Predictors are highly correlated, making OLS estimates unstable
3. **Prediction is the goal** — Interpretability of individual coefficients is less critical
4. **Computational efficiency** — Reducing dimensionality improves computational speed
5. **Data comes from measurements** — E.g., spectroscopy, sensor arrays where all dimensions capture noise

## Key Methods

### Principal Components Regression (PCR)

**Unsupervised dimensionality reduction**: Constructs principal components by maximizing variance in the predictor space $X$, then regresses the response on these components.

**Algorithm**:
1. Standardize predictors
2. Perform PCA to extract principal components
3. Select optimal number of components via cross-validation
4. Regress response on selected components

**Advantages**:
- Eliminates multicollinearity (components are uncorrelated)
- Handles $p > n$ settings naturally
- Simple, well-understood method
- PCA components have clear interpretation in terms of variance

**Disadvantages**:
- Unsupervised: PCA maximizes $X$ variance, not covariance with $y$
- May require many components to capture response variation
- Loss of original feature interpretability
- Components are linear combinations of all predictors

**When to use**: When predictor variance aligns with response variation, or as a baseline for comparison.

---

### Partial Least Squares (PLS)

**Supervised dimensionality reduction**: Constructs components by maximizing covariance between predictors and response, balancing explaining variance in $X$ with predicting $y$.

**Algorithm**:
1. Standardize predictors and response
2. Iteratively construct components that maximize covariance of $X$ with $y$
3. Select optimal number of components via cross-validation
4. Regress response on selected components

**Advantages**:
- Supervised: Components are chosen to predict response
- Often requires fewer components than PCR (more efficient)
- Excellent for high-dimensional prediction problems
- Components respect covariance structure between $X$ and $y$
- Originated in chemometrics; proven in practice

**Disadvantages**:
- Components still lack direct interpretability
- Theory less developed than OLS (fewer asymptotic results)
- Standardization required and can affect results
- Must tune number of components

**When to use**: When prediction accuracy is the goal and $p$ is large; typically outperforms PCR.

---

## Comparison: PCR vs PLS

| Aspect | PCR | PLS |
|--------|-----|-----|
| **Objective** | Maximize variance in $X$ | Maximize covariance of $X$ and $y$ |
| **Component selection** | Unsupervised | Supervised |
| **Typical # components** | Many (to capture $y$ variation) | Fewer (aligned with $y$) |
| **Computational complexity** | O(min(n,p)³) for PCA | O(min(n,p)²) per component |
| **When it excels** | Variance in $X$ important | Predicting $y$ is goal |
| **Interpretability** | Same loss as PLS | Same loss as PCR |

**Rule of thumb**: In practice, PLS often outperforms PCR because its supervised component selection aligns better with the prediction objective.

---

## Comparison with Shrinkage Methods

| Method | Type | Sparsity | Interpretability | High-dim Capability |
|--------|------|----------|------------------|-------------------|
| **Ridge** | Shrinkage | Dense | High | Good (when $p >> n$) |
| **Lasso** | Shrinkage | Sparse | High | Good (automatic selection) |
| **Elastic Net** | Shrinkage | Sparse | High | Good (balanced) |
| **PCR** | Dimension reduction | N/A | Low | Excellent |
| **PLS** | Dimension reduction | N/A | Low | Excellent |

**Decision tree**:
- **Need feature selection?** → Lasso or Elastic Net
- **Want all features + stability?** → Ridge
- **$p$ very large or $p > n$?** → PCR or PLS
- **Prediction focus, high-dimensionality?** → PLS
- **Need interpretability?** → Ridge or Lasso

---

## Contents

- **Principal Components Regression (PCR)** — Unsupervised approach using PCA for dimensionality reduction
- **Partial Least Squares (PLS)** — Supervised approach maximizing covariance with response

## Code Examples

See `code/pcr_pls_examples.py` for:
- Full implementation of PCR with cross-validation
- Full implementation of PLS with cross-validation
- Comprehensive model comparison
- Visualizations (scree plots, CV curves, predictions)
- Real-world housing price prediction example

## Practical Guidance

### When to Use PCR or PLS

**Choose PCR/PLS if**:
- $p > n$ (more predictors than observations)
- Multicollinearity is severe
- $p$ is large (50+ predictors) even if $p < n$
- You have spectroscopic or sensor data
- Prediction accuracy is paramount
- Interpretability of individual coefficients is not critical

**Choose Ridge/Lasso if**:
- Feature selection is important
- You need to understand which predictors matter
- $p$ is moderate (< 50) and $p < n$
- Interpretability is crucial
- You can afford computational cost of fitting multiple models

### Cross-Validation Strategy

Both PCR and PLS require selecting the number of components via cross-validation:

```python
from sklearn.model_selection import cross_val_score

# Test different numbers of components
for n_components in range(1, min(n_samples, n_predictors) + 1):
    cv_scores = cross_val_score(model, X_scaled, y, cv=10,
                                scoring='neg_mean_squared_error')
    mse = -cv_scores.mean()
```

Always use a held-out test set to report final performance; don't trust CV error alone.

---

## Further Reading

### Key References
- Hastie, Tibshirani, Wainwright (2015). *Statistical Learning with Sparsity* — Chapter on dimensionality reduction
- ISLR (2013). Chapter 6 — Linear Model Selection and Regularization
- Geladi & Kowalski (1986). Partial least-squares regression — original chemometrics paper

### Related Topics
- **Regularization** — Ridge, Lasso, Elastic Net (shrinkage alternatives)
- **Splines and GAMs** — Non-linear modeling with smoothness penalties (Chapter 7)
- **Model Selection** — Cross-validation, information criteria

---

## Summary

Dimensionality reduction methods (PCR and PLS) offer powerful alternatives to shrinkage when:

1. **The predictor space is very high-dimensional** ($p >> n$ or $p > n$)
2. **Multicollinearity is severe** across many predictors
3. **Prediction accuracy** is prioritized over interpretability

**Key distinction**: PLS's supervised component construction usually outperforms PCR's unsupervised approach, making PLS the preferred choice when prediction is the goal. Choose PCR only when variance in predictor space (not covariance with response) is the primary concern.
