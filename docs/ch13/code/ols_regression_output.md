# Ordinary Least Squares Regression Output Reproduction

## Overview

This page shows how to reproduce a full OLS regression summary table from scratch using the Normal Equation. Starting from the Advertising dataset (Sales regressed on TV, Radio, and Newspaper), we compute coefficients, standard errors, $t$-statistics, $p$-values, and 95% confidence intervals without relying on a pre-built regression summary function.

## Mathematical Background

The multiple linear regression model in matrix form is

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I}).
$$

The OLS estimator is obtained via the **Normal Equation**:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}.
$$

The residual standard error is

$$
s = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - k}},
$$

where $k$ is the number of parameters (including intercept). The standard error of the $j$-th coefficient is

$$
\mathrm{SE}(\hat{\beta}_j) = s \sqrt{[(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj}}.
$$

The $t$-statistic and $p$-value for testing $H_0\colon \beta_j = 0$ are

$$
t_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)}, \qquad p\text{-value} = 2\,P(T_{n-k} > |t_j|).
$$

A 95% confidence interval for $\beta_j$ is $\hat{\beta}_j \pm t^*_{n-k,\,0.025} \cdot \mathrm{SE}(\hat{\beta}_j)$.

## Code

### Fitting OLS via the Normal Equation

```python
import numpy as np
from scipy import stats

def fit_ols(X, y):
    n, k = X.shape
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    residuals = y - y_hat
    s = np.sqrt(np.sum(residuals ** 2) / (n - k))
    cov_matrix = np.linalg.inv(X.T @ X)
    return beta_hat, s, cov_matrix
```

### Producing the Regression Table

```python
def regression_table(beta_hat, s, cov_matrix, n, k, var_names):
    df = n - k
    t_crit = stats.t(df).ppf(0.975)

    for name, j in zip(var_names, range(k)):
        coef = beta_hat[j, 0]
        v_j = cov_matrix[j, j]
        se = s * np.sqrt(v_j)
        t_stat = coef / se
        p_val = 2 * stats.t(df).sf(np.abs(t_stat))
        ci_lo = coef - t_crit * se
        ci_hi = coef + t_crit * se
        print(f"{name:10}  coef={coef:.4f}  SE={se:.3f}  "
              f"t={t_stat:.3f}  p={p_val:.3f}  "
              f"CI=({ci_lo:.3f}, {ci_hi:.3f})")
```

### Running on the Advertising Dataset

```python
import pandas as pd

url = ('https://raw.githubusercontent.com/justmarkham/'
       'scikit-learn-videos/master/data/Advertising.csv')
data = pd.read_csv(url, usecols=[1, 2, 3, 4])
training_data = data.iloc[:int(len(data) * 0.7)]

y = np.array(training_data.Sales).reshape(-1, 1)
n = y.shape[0]
X = np.concatenate(
    (np.ones((n, 1)), np.array(training_data.iloc[:, :-1])), axis=1
)
k = X.shape[1]

beta_hat, s, cov_matrix = fit_ols(X, y)
var_names = ["Intercept", "TV", "Radio", "Newspaper"]
regression_table(beta_hat, s, cov_matrix, n, k, var_names)
```

## Interpretation

- Each row in the regression table corresponds to one predictor. The coefficient estimates $\hat{\beta}_j$ give the expected change in Sales per unit increase in the predictor, holding other predictors constant.
- The standard error quantifies the precision of each estimate. Smaller SE means more precise estimation.
- The $t$-statistic measures how many standard errors the coefficient is away from zero. Large absolute values indicate statistical significance.
- The $p$-value gives the probability of observing a $t$-statistic at least as extreme under $H_0\colon \beta_j = 0$. A $p$-value below 0.05 is conventionally considered significant.
- The 95% confidence interval provides a range of plausible values for the true coefficient. If it excludes zero, the predictor is significant at the 5% level.

## Exercises

**Exercise 1.** Verify numerically that $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$ satisfies the normal equations $\mathbf{X}^\top \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top \mathbf{y}$.

??? success "Solution to Exercise 1"

    ```python
    lhs = X.T @ X @ beta_hat
    rhs = X.T @ y
    print(np.allclose(lhs, rhs))  # True
    ```

    By construction, multiplying both sides of $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ on the left by $\mathbf{X}^\top\mathbf{X}$ yields $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$. $\square$

---

**Exercise 2.** Compute $R^2$ and Adjusted $R^2$ from the residuals. Show that $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$.

??? success "Solution to Exercise 2"

    ```python
    y_hat = X @ beta_hat
    RSS = np.sum((y - y_hat) ** 2)
    TSS = np.sum((y - y.mean()) ** 2)
    R2 = 1 - RSS / TSS
    adj_R2 = 1 - (1 - R2) * (n - 1) / (n - k)
    print(f"R^2 = {R2:.4f}, Adjusted R^2 = {adj_R2:.4f}")
    ```

    By definition, $\mathrm{TSS} = \sum(y_i - \bar{y})^2$, $\mathrm{RSS} = \sum(y_i - \hat{y}_i)^2$, and $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$ measures the proportion of variance explained by the model. Adjusted $R^2$ penalizes for the number of predictors via $1 - \frac{n-1}{n-k}(1 - R^2)$. $\square$

---

**Exercise 3.** Explain why using $n - k$ instead of $n$ in the denominator of $s^2$ produces an unbiased estimator of $\sigma^2$.

??? success "Solution to Exercise 3"

    The residual vector $\mathbf{e} = \mathbf{M}\mathbf{y}$ where $\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Under the model, $\mathbf{e} = \mathbf{M}\boldsymbol{\varepsilon}$. Then

    $$
    E[\mathbf{e}^\top\mathbf{e}] = E[\boldsymbol{\varepsilon}^\top\mathbf{M}\boldsymbol{\varepsilon}] = \sigma^2 \operatorname{tr}(\mathbf{M}) = \sigma^2(n - k),
    $$

    since $\mathbf{M}$ is idempotent with $\operatorname{tr}(\mathbf{M}) = n - k$. Dividing by $n - k$ gives $E[s^2] = \sigma^2$. $\square$

---

**Exercise 4.** Recompute the regression table using `numpy.linalg.lstsq` instead of explicitly inverting $\mathbf{X}^\top\mathbf{X}$. Discuss why `lstsq` is preferred numerically.

??? success "Solution to Exercise 4"

    ```python
    beta_lstsq, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    ```

    `lstsq` uses the SVD decomposition, which is numerically more stable than explicitly computing $(\mathbf{X}^\top\mathbf{X})^{-1}$. When $\mathbf{X}^\top\mathbf{X}$ is ill-conditioned (nearly singular), direct inversion amplifies floating-point errors, whereas the SVD gracefully handles near-collinearity. The results match to machine precision for well-conditioned problems. $\square$

---

**Exercise 5.** Prove that the $t$-statistic for the $j$-th coefficient can be written as $t_j = \hat{\beta}_j \sqrt{[(\mathbf{X}^\top\mathbf{X})]_{jj}} / s$ only when predictors are orthogonal. What happens in general?

??? success "Solution to Exercise 5"

    When predictors are orthogonal, $\mathbf{X}^\top\mathbf{X}$ is diagonal, so $[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj} = 1/[(\mathbf{X}^\top\mathbf{X})]_{jj}$. In this case:

    $$
    t_j = \frac{\hat{\beta}_j}{s\sqrt{[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}}} = \frac{\hat{\beta}_j\sqrt{[(\mathbf{X}^\top\mathbf{X})]_{jj}}}{s}.
    $$

    In general, $(\mathbf{X}^\top\mathbf{X})^{-1}$ is not simply the reciprocal of the diagonal, because off-diagonal elements of $\mathbf{X}^\top\mathbf{X}$ (correlations between predictors) inflate the diagonal entries of the inverse. This inflation is captured by the Variance Inflation Factor (VIF). $\square$
