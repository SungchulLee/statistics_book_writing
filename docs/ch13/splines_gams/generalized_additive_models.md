# Generalized Additive Models (GAMs)

## Overview

**Generalized Additive Models (GAMs)** extend linear regression by allowing smooth, non-parametric functions of predictors instead of assuming linear relationships. GAMs provide a flexible middle ground between rigid linear models and overly complex black-box methods like neural networks.

The key innovation of GAMs is replacing linear terms with smooth functions while maintaining interpretability:

**Linear Regression:**
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

**Generalized Additive Model:**
$$Y = \beta_0 + f_1(X_1) + f_2(X_2) + \cdots + f_p(X_p) + \epsilon$$

where each $f_j$ is a smooth function (typically a spline) learned from the data.

---

## Why Use GAMs?

GAMs address several limitations of linear regression:

1. **Non-linear relationships** — Many real-world relationships are curved. GAMs capture these automatically without manually specifying polynomial degrees.

2. **Different smoothness per variable** — Each predictor can have its own degree of smoothing, controlled by a penalty parameter (lambda).

3. **Interpretability** — Unlike neural networks, each smooth function $f_j(X_j)$ can be visualized and interpreted independently. The additive structure means effects don't interact by default.

4. **Automatic overfitting control** — Regularization prevents spurious wiggles in the smooth functions while maintaining flexibility.

5. **Uncertainty quantification** — Unlike tree-based methods, GAMs provide confidence bands around predictions through standard errors.

---

## Mathematical Formulation

### The Basic GAM

For a continuous response, the Gaussian GAM is:

$$Y = \beta_0 + \sum_{j=1}^{p} f_j(X_j) + \epsilon, \quad \epsilon \sim N(0, \sigma^2)$$

### Smooth Functions Using Splines

The smooth functions $f_j$ are typically represented as linear combinations of basis functions:

$$f_j(X_j) = \sum_{k=1}^{K_j} b_{jk}(X_j) \cdot c_{jk}$$

where:
- $b_{jk}$ are basis functions (e.g., B-splines, thin-plate splines)
- $c_{jk}$ are coefficients learned from data
- $K_j$ is the number of basis functions for variable $j$

### Regularization: Penalized Estimation

To avoid overfitting while allowing flexibility, GAMs use a roughness penalty:

$$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p} f_j(x_{ij})\right)^2 + \sum_{j=1}^{p} \lambda_j \int [f_j''(x)]^2 dx$$

where:
- The first term is the sum of squared residuals
- $\lambda_j$ controls the smoothness of the $j$-th function: larger $\lambda_j$ results in smoother (less wiggly) functions
- The integral term measures the "roughness" (second derivative squared)

### Degrees of Freedom and Effective DoF

Unlike linear regression where degrees of freedom equal the number of parameters, GAMs have **effective degrees of freedom (eDoF)** accounting for the smoothness penalty:

$$\text{eDoF}_j = \text{tr}(S_j)$$

where $S_j$ is a matrix depending on the basis and the penalty. The total model complexity is:

$$\text{eDoF}_{\text{total}} = 1 + \sum_{j=1}^{p} \text{eDoF}_j$$

This allows model comparison using the same criteria (AIC, BIC) as linear regression, with eDoF replacing traditional parameter counts.

---

## Fitting GAMs: The Backfitting Algorithm

The most common fitting approach is **backfitting**, an iterative algorithm:

1. Initialize: $\hat{f}_j^{(0)} = 0$ for all $j$, and $\hat{\beta}_0 = \bar{y}$

2. For iteration $t$:
   - For each $j = 1, 2, \ldots, p$:
     - Compute partial residuals: $r_{-j} = y - \hat{\beta}_0 - \sum_{k \neq j} \hat{f}_k(X_k)$
     - Fit a smooth function to $(X_j, r_{-j})$ with penalty $\lambda_j$: $\hat{f}_j^{(t)} = S(r_{-j} | X_j, \lambda_j)$

3. Repeat until convergence (coefficients stabilize)

This decomposes the fitting problem into univariate smoothing problems, making GAMs computationally efficient compared to fitting high-dimensional non-parametric functions directly.

---

## Types of Smooth Terms

### Linear Terms
A linear term is included as:
$$f_j(X_j) = \beta_j X_j$$

This has no smoothness penalty and is useful when a relationship is genuinely linear.

### Spline Terms (s)
Represented by basis functions with a smoothness penalty:
$$f_j(X_j) = \sum_{k=1}^{K_j} b_{jk}(X_j) c_{jk} + \lambda_j \int [f_j''(x)]^2 dx$$

Common choices:
- **Cubic B-splines**: Smooth, locally-supported, computationally efficient
- **Thin-plate splines**: Optimal in a smoothness sense, but computationally expensive

The `df` (degrees of freedom) parameter controls the flexibility: larger `df` allows more wiggles.

### Cyclic Splines
For periodic data (e.g., time of day, day of week), cyclic splines enforce $f(0) = f(1)$ (or appropriate boundaries).

---

## Python Implementation

### Using statsmodels

The `statsmodels.gam` module provides GAM fitting:

```python
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import matplotlib.pyplot as plt

# Sample data
n = 500
np.random.seed(42)
X = np.random.uniform(0, 10, (n, 3))
y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.normal(0, 0.5, n))

# Create DataFrame
df = pd.DataFrame({
    'y': y,
    'x0': X[:, 0],
    'x1': X[:, 1],
    'x2': X[:, 2]
})

# Define smooth basis
x_spline = df[['x0', 'x1', 'x2']]
bs = BSplines(x_spline, df=[10, 3, 3], degree=[3, 2, 2])

# Fit GAM
formula = 'y ~ x0 + x1 + x2'
gam = GLMGam.from_formula(formula, data=df, smoother=bs)
results = gam.fit()

print(results.summary())
```

### Using pyGAM

The `pygam` library provides a more user-friendly interface with automatic lambda selection via grid search:

```python
from pygam import LinearGAM, s, l
from pygam.utils import generate_X_grid

# Fit GAM with smoothing spline on x0 and linear terms on x1, x2
gam = LinearGAM(s(0, n_splines=12) + l(1) + l(2))

# Automatic lambda selection via grid search
gam.gridsearch(X, y)

# Print summary
print(gam.summary())

# Visualize partial dependence
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i in range(3):
    XX = gam.generate_X_grid(term=i)
    ax = axes[i]
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.fill_between(XX[:, i],
                     gam.partial_dependence(term=i, X=XX, width=0.95)[1],
                     gam.partial_dependence(term=i, X=XX, width=0.95)[2],
                     alpha=0.3)
    ax.set_xlabel(f'x{i}')
    ax.set_ylabel(f'f{i}(x{i})')
    ax.set_title(f'Partial Dependence: x{i}')

plt.tight_layout()
plt.show()
```

---

## Model Selection for GAMs

### Smoothing Parameter Selection

The smoothing parameters $\lambda_j$ control the bias-variance tradeoff:

- **Larger $\lambda_j$** → smoother function (high bias, low variance)
- **Smaller $\lambda_j$** → wigglier function (low bias, high variance)

Common selection methods:

1. **Generalized Cross-Validation (GCV)**: Balances fit and complexity, computationally efficient
2. **UBRE (Un-biased Risk Estimator)**: Similar to AIC, works well for Gaussian responses
3. **Automatic grid search**: pyGAM automatically searches a grid of lambda values

### Comparing GAMs

Once smooth functions are estimated, compare models using:

- **Deviance** (residual sum of squares for Gaussian responses)
- **Effective DoF** (reflects model complexity)
- **AIC/BIC** with eDoF in place of parameter count:
  $$\text{AIC} = -2 \log L + 2 \cdot \text{eDoF}$$

### Adjusting Model Complexity

Control overall model complexity via:

1. **Degrees of freedom per term** (`df` parameter): fewer basis functions → smoother
2. **Global smoothing penalty**: multiply all lambdas by a constant
3. **Model formula**: include only relevant smooth terms

---

## Advantages and Disadvantages

### Advantages

- **Flexibility**: Captures non-linear relationships without manual specification
- **Interpretability**: Individual smooth functions are visualizable and understandable
- **Automatic smoothness selection**: Many algorithms optimize smoothing parameters automatically
- **Uncertainty quantification**: Provides confidence bands and standard errors
- **Efficiency**: Backfitting makes fitting scalable to moderate dimensions
- **Fairness**: Additive structure avoids interactions by default, making effects comparable

### Disadvantages

- **Curse of dimensionality**: Performance degrades as the number of predictors increases beyond 10-15 (though more efficient than non-parametric methods)
- **Assumption of additivity**: Interactions must be explicitly included; more complex for high-order interactions
- **Smoothing parameter selection**: Can be sensitive to the choice of lambda; grid search adds computation
- **Interpretability trade-off**: More complex relationships are harder to summarize than simple parametric forms
- **Software dependence**: Results may vary slightly across implementations (statsmodels vs. pyGAM vs. R's mgcv)

---

## Practical Example: Housing Prices

Consider predicting house prices from multiple features. A GAM allows different degrees of smoothness for each predictor:

```python
from pygam import LinearGAM, s, l
import pandas as pd

# Load housing data
house = pd.read_csv('house_sales.csv', sep='\t')

# Select predictors
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
X = house[predictors].values
y = house['AdjSalePrice'].values

# Fit GAM: smooth spline on square footage, linear terms on others
gam = LinearGAM(
    s(0, n_splines=12) +     # SqFtTotLiving: smooth (likely non-linear)
    l(1) +                   # SqFtLot: linear
    l(2) +                   # Bathrooms: linear
    l(3) +                   # Bedrooms: linear
    l(4)                     # BldgGrade: linear
)

gam.gridsearch(X, y)
print(gam.summary())

# Predict on new data
new_house = pd.DataFrame({
    'SqFtTotLiving': [3000],
    'SqFtLot': [10000],
    'Bathrooms': [3.5],
    'Bedrooms': [4],
    'BldgGrade': [10]
})
prediction = gam.predict(new_house[predictors].values)
print(f"Predicted price: ${prediction[0]:,.0f}")

# Visualize the smooth term
fig, ax = plt.subplots(figsize=(6, 4))
XX = gam.generate_X_grid(term=0)
ax.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
ax.set_xlabel('Square Feet (Living)')
ax.set_ylabel('Contribution to Price')
ax.set_title('GAM: Non-Linear Effect of Square Footage')
plt.tight_layout()
plt.show()
```

---

## Summary

Generalized Additive Models provide a powerful, interpretable approach to non-linear regression:

- **Flexible smooth functions** replace rigid linear terms while maintaining additivity
- **Automatic smoothing** via regularization prevents overfitting
- **Individual visualization** of each effect aids interpretation
- **Practical tools** (statsmodels, pyGAM) make GAMs accessible for real applications
- **Trade-offs** between flexibility and interpretability make GAMs ideal when moderate non-linearity is expected

When data suggests non-linear relationships and interpretation is important, GAMs offer an excellent balance between the simplicity of linear regression and the flexibility of fully non-parametric methods.
