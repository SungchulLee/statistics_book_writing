# Ridge Regression Examples

## Overview

Ridge regression adds an $L_2$ penalty to the ordinary least squares (OLS) objective, shrinking
coefficients toward zero without setting any exactly to zero. This technique is especially
effective when predictors are correlated (multicollinearity) or when $p$ is close to or exceeds
$n$. In this page we derive the ridge estimator, examine its bias--variance tradeoff, and
illustrate the effect of the tuning parameter $\lambda$ on coefficient estimates.

## The Ridge Objective

Given a design matrix $X \in \mathbb{R}^{n \times p}$ and response $y \in \mathbb{R}^n$, the
ridge regression problem is

$$
\hat{\beta}^{\text{ridge}} = \arg\min_{\beta} \left\{ \| y - X\beta \|_2^2 + \lambda \| \beta \|_2^2 \right\},
$$

where $\lambda \ge 0$ is the regularization (tuning) parameter. The penalty term
$\lambda \| \beta \|_2^2 = \lambda \sum_{j=1}^{p} \beta_j^2$ discourages large coefficient
values.

### Closed-Form Solution

Setting the gradient to zero yields the closed-form solution

$$
\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y.
$$

When $\lambda = 0$, this reduces to the OLS estimator. As $\lambda \to \infty$, every
coefficient shrinks toward zero.

## Bias--Variance Tradeoff

Ridge regression introduces bias in exchange for reduced variance. The mean squared error (MSE)
of the ridge estimator can be decomposed as

$$
\text{MSE}(\hat{\beta}^{\text{ridge}}) = \text{Bias}^2 + \text{Variance}.
$$

For small $\lambda$, the estimator is nearly unbiased but has high variance (close to OLS).
For large $\lambda$, variance is low but bias is large. The optimal $\lambda$ minimizes total
MSE.

## Code: Generating Data and Fitting Ridge

The following script generates normally distributed sample data and prints basic summary
statistics. In practice, you would replace this with a full ridge regression fit.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

print(f"Sample size: {n}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std:  {data.std(ddof=1):.4f}")
```

In a complete implementation one would construct $X$ and $y$, standardize the predictors, and
solve for $\hat{\beta}^{\text{ridge}}$ across a grid of $\lambda$ values.

## Standardization

Because the $L_2$ penalty treats all coefficients equally, predictors should be standardized
before fitting:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j},
$$

where $\bar{x}_j$ and $s_j$ are the sample mean and standard deviation of the $j$-th
predictor. Without standardization, the penalty disproportionately shrinks coefficients of
predictors measured on larger scales.

## Regularization Path

A **regularization path** plots each coefficient $\hat{\beta}_j^{\text{ridge}}$ as a function
of $\lambda$ (or $\log_{10}\lambda$). Key observations:

- All coefficients are nonzero for every finite $\lambda$.
- Coefficients shrink smoothly and monotonically toward zero as $\lambda$ increases.
- Coefficients corresponding to important predictors remain large for a wider range of
  $\lambda$.

## Interpretation

- **Ridge never performs variable selection.** All predictors remain in the model regardless of
  $\lambda$. If interpretability via sparsity is needed, consider Lasso or Elastic Net.
- **Multicollinearity relief.** The term $\lambda I_p$ added to $X^\top X$ ensures the matrix
  is invertible and stabilizes the estimates.
- **Choosing $\lambda$.** Cross-validation (e.g., 5-fold or 10-fold) is the standard method.
  One picks the $\lambda$ that minimizes the cross-validated prediction error.

## Exercises

**Exercise 1.** Starting from the ridge objective, derive the closed-form solution
$\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y$ by setting the gradient
equal to zero.

??? success "Solution to Exercise 1"

    The objective is

    $$
    L(\beta) = (y - X\beta)^\top (y - X\beta) + \lambda \beta^\top \beta.
    $$

    Expanding and differentiating with respect to $\beta$:

    $$
    \frac{\partial L}{\partial \beta} = -2 X^\top y + 2 X^\top X \beta + 2\lambda \beta.
    $$

    Setting this to zero:

    $$
    (X^\top X + \lambda I_p) \beta = X^\top y.
    $$

    Since $X^\top X + \lambda I_p$ is positive definite for $\lambda > 0$, it is invertible, so

    $$
    \hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y. \quad \square
    $$

---

**Exercise 2.** Show that the ridge estimator can be written in terms of the singular value
decomposition (SVD) as $\hat{\beta}^{\text{ridge}} = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda}\, \frac{u_j^\top y}{d_j}\, v_j$, where $X = U D V^\top$.

??? success "Solution to Exercise 2"

    Let $X = U D V^\top$ be the SVD with $D = \text{diag}(d_1, \dots, d_p)$. Then
    $X^\top X = V D^2 V^\top$ and $X^\top y = V D U^\top y$. Substituting:

    $$
    \hat{\beta}^{\text{ridge}} = (V D^2 V^\top + \lambda I)^{-1} V D U^\top y = V (D^2 + \lambda I)^{-1} D U^\top y.
    $$

    In component form, the $j$-th element of $V^\top \hat{\beta}^{\text{ridge}}$ is
    $\frac{d_j}{d_j^2 + \lambda} u_j^\top y$, so

    $$
    \hat{\beta}^{\text{ridge}} = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda} \cdot \frac{u_j^\top y}{d_j} \cdot v_j.
    $$

    The factor $d_j^2 / (d_j^2 + \lambda) \in [0, 1)$ shrinks directions with small singular
    values more aggressively. $\square$

---

**Exercise 3.** Suppose $X^\top X = I_p$ (orthonormal design). Express
$\hat{\beta}_j^{\text{ridge}}$ in terms of $\hat{\beta}_j^{\text{OLS}}$ and $\lambda$.

??? success "Solution to Exercise 3"

    When $X^\top X = I_p$, the OLS estimator is $\hat{\beta}^{\text{OLS}} = X^\top y$.
    The ridge estimator becomes

    $$
    \hat{\beta}^{\text{ridge}} = (I_p + \lambda I_p)^{-1} X^\top y = \frac{1}{1 + \lambda}\, \hat{\beta}^{\text{OLS}}.
    $$

    So each coefficient is uniformly scaled by $1/(1 + \lambda)$. This confirms that ridge
    regression applies proportional shrinkage. $\square$

---

**Exercise 4.** Using 5-fold cross-validation on a synthetic dataset of your choice
($n = 200$, $p = 10$, with multicollinearity), find the optimal $\lambda$ from the grid
$\lambda \in \{10^{-3}, 10^{-2}, \dots, 10^{3}\}$. Report the CV RMSE for the best $\lambda$
and compare it to the OLS RMSE.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 10

    # Correlated design
    rho = 0.9
    Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta_true = np.array([3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0])
    y = X @ beta_true + np.random.randn(n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # OLS
    ols_scores = cross_val_score(
        LinearRegression(), X_s, y, cv=5,
        scoring="neg_mean_squared_error"
    )
    ols_rmse = np.sqrt(-ols_scores.mean())

    # Ridge grid search
    best_rmse, best_lam = np.inf, None
    for exp in range(-3, 4):
        lam = 10.0 ** exp
        scores = cross_val_score(
            Ridge(alpha=lam), X_s, y, cv=5,
            scoring="neg_mean_squared_error"
        )
        rmse = np.sqrt(-scores.mean())
        if rmse < best_rmse:
            best_rmse, best_lam = rmse, lam

    print(f"OLS CV RMSE:  {ols_rmse:.4f}")
    print(f"Best lambda:  {best_lam}")
    print(f"Ridge CV RMSE: {best_rmse:.4f}")
    ```

    Typical output shows the ridge CV RMSE is lower than the OLS CV RMSE, confirming that
    regularization helps when predictors are correlated. $\square$

---

**Exercise 5.** Prove that for any $\lambda > 0$ the ridge estimator satisfies
$\|\hat{\beta}^{\text{ridge}}\|_2 \le \|\hat{\beta}^{\text{OLS}}\|_2$.

??? success "Solution to Exercise 5"

    The ridge problem is equivalent to

    $$
    \min_{\beta} \| y - X\beta \|_2^2 \quad \text{subject to} \quad \|\beta\|_2^2 \le t,
    $$

    for some $t > 0$ that depends on $\lambda$ (by the KKT conditions). The OLS solution
    minimizes the loss without any constraint, so $\hat{\beta}^{\text{OLS}}$ is either inside
    the constraint region (in which case $\hat{\beta}^{\text{ridge}} = \hat{\beta}^{\text{OLS}}$
    and equality holds) or outside it. When outside, the constrained optimum lies on the
    boundary $\|\beta\|_2^2 = t < \|\hat{\beta}^{\text{OLS}}\|_2^2$. In either case,

    $$
    \|\hat{\beta}^{\text{ridge}}\|_2 \le \|\hat{\beta}^{\text{OLS}}\|_2. \quad \square
    $$
