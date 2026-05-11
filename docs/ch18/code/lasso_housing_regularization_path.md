# Lasso Housing Regularization Path

## Overview

This page applies Lasso regression to real housing data (King County house sales), tracing the
full regularization path from the null model ($\hat{\beta} = 0$) to the OLS solution. We use
cross-validation to select the optimal $\lambda$, compare performance against OLS and Ridge,
and interpret which housing features survive Lasso's variable selection.

## Problem Setup

We model the adjusted sale price as a linear function of housing characteristics:

$$
\text{AdjSalePrice} = \beta_0 + \beta_1 \cdot \text{SqFtTotLiving} + \beta_2 \cdot \text{SqFtLot} + \cdots + \varepsilon.
$$

The predictors include both numeric features (square footage, bathrooms, year built) and
categorical features (property type) that are one-hot encoded. All features are standardized
before fitting.

## Code: Data Loading and Preparation

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV

DATA = Path(__file__).parent.parent.parent / 'data'
house = pd.read_csv(DATA / 'house_sales.csv', sep='\t')

predictors = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'PropertyType', 'NbrLivingUnits',
    'SqFtFinBasement', 'YrBuilt', 'YrRenovated', 'NewConstruction'
]
outcome = 'AdjSalePrice'

X = pd.get_dummies(house[predictors], drop_first=True)
X['NewConstruction'] = X['NewConstruction'].astype(int)
y = house[outcome]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

Standardization is essential because the $L_1$ penalty $\lambda\|\beta\|_1$ penalizes all
coefficients equally; without it, predictors on different scales would be penalized
disproportionately.

## Code: OLS Baseline

```python
ols_model = LinearRegression().fit(X_scaled, y)
ols_pred = ols_model.predict(X_scaled)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ols_rmse = np.sqrt(mean_squared_error(y, ols_pred))
ols_r2 = r2_score(y, ols_pred)
n_nonzero_ols = np.sum(np.abs(ols_model.coef_) > 1e-8)
```

OLS retains all features with nonzero coefficients and serves as the unregularized benchmark.

## Code: Regularization Path

We fit the Lasso over 100 logarithmically spaced $\lambda$ values and track how each
coefficient evolves.

```python
import matplotlib.pyplot as plt

alphas = np.logspace(2, -2, 100)

lasso_coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)
n_features_selected = (np.abs(lasso_coefs) > 1e-8).sum(axis=1)
```

The regularization path reveals the order in which features enter the model as $\lambda$
decreases. Features that appear first are the strongest predictors.

## Code: Cross-Validated Lambda Selection

```python
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

lasso_pred = lasso_cv.predict(X_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_pred))
lasso_r2 = r2_score(y, lasso_pred)
n_nonzero_lasso = np.sum(np.abs(lasso_cv.coef_) > 1e-8)
```

The 5-fold CV procedure evaluates each $\lambda$ and selects the one with the lowest average
MSE.

## Code: Model Comparison

```python
ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=5)
ridge_cv.fit(X_scaled, y)

ridge_pred = ridge_cv.predict(X_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
ridge_r2 = r2_score(y, ridge_pred)
```

## Visualizations

### Regularization Path Plot

The left panel shows coefficient trajectories versus $\log_{10}(\lambda)$, with a vertical
dashed line at the optimal $\lambda$. The right panel shows the number of active (nonzero)
features at each $\lambda$, illustrating the model complexity--regularization tradeoff.

### Cross-Validation Error Plot

The CV RMSE curve (with $\pm 1$ standard deviation bands) is U-shaped. The minimum identifies
the optimal $\lambda$ that balances bias and variance.

### Model Comparison Plot

Bar charts of RMSE and $R^2$ across OLS, Ridge, and Lasso show that:

- All three models achieve similar $R^2$ on this dataset.
- Lasso achieves comparable prediction accuracy with fewer features.

## Interpretation

Key findings from the housing data analysis:

- **Feature selection.** Lasso selects a subset of the original features, setting the
  coefficients of less important predictors (e.g., `YrRenovated`, `NbrLivingUnits`) to zero.
- **Top predictors.** `BldgGrade` (building grade) and `SqFtTotLiving` (total living area)
  typically have the largest absolute coefficients, confirming their importance in predicting
  house prices.
- **Ridge vs. Lasso.** Ridge retains all features with continuous shrinkage, while Lasso
  performs automatic variable selection. On this moderately sized dataset, prediction accuracy
  is similar.
- **Practical benefit.** The Lasso model is more interpretable: a stakeholder can see which
  features drive the price prediction without examining 13+ small coefficients.

## Exercises

**Exercise 1.** Explain why the regularization path for Lasso is piecewise linear (as a
function of $\lambda$), while the Ridge path is smooth. Hint: consider the KKT conditions for
each method.

??? success "Solution to Exercise 1"

    **Lasso:** The KKT (subgradient) conditions for the Lasso are

    $$
    -\frac{1}{n}X_j^\top(y - X\hat{\beta}) + \lambda s_j = 0, \quad s_j \in \partial|\hat{\beta}_j|.
    $$

    For the active set $\mathcal{A} = \{j : \hat{\beta}_j \ne 0\}$, the sign $s_j = \text{sign}(\hat{\beta}_j)$ is fixed. The active coefficients then solve a linear system in
    $\lambda$, so $\hat{\beta}_{\mathcal{A}}(\lambda)$ is linear in $\lambda$ between
    consecutive breakpoints (where a variable enters or leaves the active set). This gives the
    piecewise-linear structure.

    **Ridge:** The closed-form solution $\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$ is a rational function of $\lambda$ (matrix inverse of a linear function of
    $\lambda$), which is smooth (infinitely differentiable) for all $\lambda > 0$. No variables
    are ever set to exactly zero, so there are no breakpoints. $\square$

---

**Exercise 2.** In the housing dataset, `BldgGrade` and `SqFtTotLiving` are likely correlated.
Discuss what would happen if you used Lasso versus Elastic Net for variable selection in the
presence of this correlation.

??? success "Solution to Exercise 2"

    When `BldgGrade` and `SqFtTotLiving` are highly correlated, the Lasso solution is unstable:
    small perturbations in the data can cause the Lasso to select one feature while dropping the
    other. The Lasso might arbitrarily assign the entire effect to one predictor.

    The Elastic Net (with $0 < \alpha < 1$) has a grouping property: if two predictors are
    highly correlated and both truly relevant, the Elastic Net tends to either include both or
    exclude both. The $L_2$ component ensures the solution is unique and stabilizes the
    coefficient paths, leading to more reproducible feature selection.

    In practical terms, if domain knowledge says both features matter, the Elastic Net is
    preferable. If the goal is maximum sparsity and the two features are essentially
    interchangeable, the Lasso's selection of one is acceptable. $\square$

---

**Exercise 3.** The script uses in-sample $R^2$ and RMSE for the model comparison. Explain why
this is potentially misleading and propose a better evaluation strategy.

??? success "Solution to Exercise 3"

    In-sample metrics evaluate the model on the same data used for training, so they are
    optimistically biased. OLS has the most parameters and will always achieve the lowest
    in-sample RSS (and highest in-sample $R^2$) among linear models. This makes OLS look
    comparable to or better than regularized methods, even when it overfits.

    **Better strategy:** Use cross-validated metrics. For a fair comparison:

    1. Use the same $K$-fold splits for all methods.
    2. Inside each fold, standardize using training-fold statistics only.
    3. Report CV RMSE and CV $R^2$ (computed on held-out folds).

    Alternatively, hold out a fixed test set (e.g., 20%) that is never used during model
    selection or training. The test-set RMSE gives an unbiased estimate of generalization
    performance.

    On many datasets, regularized methods outperform OLS in CV metrics even if their in-sample
    metrics are slightly worse. $\square$

---

**Exercise 4.** Suppose you add 50 random noise features to the housing dataset (features
drawn from $N(0,1)$ with no relation to the response). How would you expect the optimal
$\lambda$ and the number of selected features to change for Lasso? Run the experiment and
report your findings.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    # Assume X_scaled and y are already defined from the housing data

    # Add 50 noise features
    noise = np.random.randn(len(y), 50)
    noise_cols = [f'noise_{i}' for i in range(50)]
    X_aug = pd.concat([X_scaled.reset_index(drop=True),
                       pd.DataFrame(noise, columns=noise_cols)], axis=1)

    lasso_aug = LassoCV(n_alphas=100, cv=5, max_iter=10000)
    lasso_aug.fit(X_aug, y)

    n_nz = np.sum(np.abs(lasso_aug.coef_) > 1e-8)
    noise_selected = np.sum(np.abs(lasso_aug.coef_[-50:]) > 1e-8)

    print(f"Optimal lambda: {lasso_aug.alpha_:.4f}")
    print(f"Total nonzero:  {n_nz}")
    print(f"Noise features selected: {noise_selected}/50")
    ```

    **Expected results:** The optimal $\lambda$ increases (more regularization needed to
    counteract the noise dimensions). The Lasso should correctly set most or all of the 50 noise
    features to zero, while retaining the genuinely predictive housing features. A few noise
    features might slip through (false positives), especially if $p$ is large relative to $n$.
    $\square$

---

**Exercise 5.** Derive the relationship between the Lagrangian parameter $\lambda$ and the
constraint bound $t$ in the equivalent constrained formulation
$\min \|y - X\beta\|_2^2$ subject to $\|\beta\|_1 \le t$. Specifically, show that there is a
one-to-one decreasing correspondence between $\lambda > 0$ and $t \in (0, \|\hat{\beta}^{\text{OLS}}\|_1)$.

??? success "Solution to Exercise 5"

    The Lasso penalized problem and the constrained problem are related by Lagrangian duality.
    Let $f(t) = \min_{\|\beta\|_1 \le t} \|y - X\beta\|_2^2$. Since $\|y - X\beta\|_2^2$ is
    strictly convex in $\beta$ (assuming $X$ has full column rank) and the constraint
    $\|\beta\|_1 \le t$ is convex, the KKT conditions guarantee that the penalized solution
    with parameter $\lambda$ coincides with the constrained solution for some $t(\lambda)$.

    By the KKT complementary slackness condition, $\lambda(\|\hat{\beta}\|_1 - t) = 0$. For
    $\lambda > 0$, the constraint is active: $\|\hat{\beta}\|_1 = t$.

    As $\lambda$ increases, the penalty more aggressively shrinks coefficients, so
    $\|\hat{\beta}(\lambda)\|_1$ decreases. Since
    $t(\lambda) = \|\hat{\beta}(\lambda)\|_1$, $t$ is a decreasing function of $\lambda$.

    - At $\lambda = 0$: $t = \|\hat{\beta}^{\text{OLS}}\|_1$.
    - As $\lambda \to \infty$: $t \to 0$.

    The correspondence is one-to-one because the Lasso path is continuous and strictly
    decreasing in $\|\hat{\beta}\|_1$ (each coefficient magnitude is non-increasing in
    $\lambda$, and at least one is strictly decreasing until it reaches zero). $\square$
