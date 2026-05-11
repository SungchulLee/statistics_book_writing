# Weighted Least Squares

## Overview

This page demonstrates Weighted Least Squares (WLS) regression as a remedy for heteroscedasticity. We generate data where the error variance increases linearly with the predictor, fit both OLS and WLS, and compare their coefficient estimates, standard errors, and residual plots to show how WLS corrects for non-constant variance.

## Mathematical Background

When the error variance is not constant, $\mathrm{Var}(\varepsilon_i) = \sigma_i^2$, OLS remains unbiased but is no longer efficient and its standard errors are incorrect. WLS addresses this by minimizing a weighted sum of squared residuals:

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^n w_i(y_i - \mathbf{x}_i^\top\boldsymbol{\beta})^2,
$$

where $w_i = 1/\sigma_i^2$ (observations with higher variance get lower weight). In matrix form:

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{y},
$$

where $\mathbf{W} = \mathrm{diag}(w_1, \ldots, w_n)$.

The covariance matrix of $\hat{\boldsymbol{\beta}}_{\text{WLS}}$ is

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}_{\text{WLS}}) = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}.
$$

WLS is equivalent to applying OLS to the transformed model $\sqrt{w_i}\,y_i = \sqrt{w_i}\,\mathbf{x}_i^\top\boldsymbol{\beta} + \sqrt{w_i}\,\varepsilon_i$, where the transformed errors have constant variance.

## Code

### OLS and WLS Implementations

```python
import numpy as np

def ols_fit(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def wls_fit(X, y, w):
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta
```

### Generating Heteroscedastic Data and Fitting

```python
np.random.seed(42)
n = 120
x = np.random.uniform(1, 10, n)
sigma = 0.5 + 1.5 * x  # variance grows with x
y = 3.0 + 2.0 * x + np.random.normal(0, sigma)

X = np.column_stack([np.ones(n), x])

# OLS
beta_ols = ols_fit(X, y)

# WLS with weights = 1 / sigma^2
w = 1.0 / sigma ** 2
beta_wls = wls_fit(X, y, w)
```

### Standard Errors Comparison

```python
# OLS SE (assumes homoscedasticity)
resid_ols = y - X @ beta_ols
s2_ols = np.sum(resid_ols ** 2) / (n - 2)
se_ols = np.sqrt(np.diag(s2_ols * np.linalg.inv(X.T @ X)))

# WLS SE
W = np.diag(w)
XtWX_inv = np.linalg.inv(X.T @ W @ X)
se_wls = np.sqrt(np.diag(XtWX_inv))

print(f"OLS:  intercept={beta_ols[0]:.3f} (SE={se_ols[0]:.3f}), "
      f"slope={beta_ols[1]:.3f} (SE={se_ols[1]:.3f})")
print(f"WLS:  intercept={beta_wls[0]:.3f} (SE={se_wls[0]:.3f}), "
      f"slope={beta_wls[1]:.3f} (SE={se_wls[1]:.3f})")
```

## Interpretation

- **OLS with heteroscedasticity**: The OLS estimates remain unbiased, but the standard errors computed under the homoscedasticity assumption are incorrect. The residual plot shows a characteristic "fan shape" where spread increases with $x$.
- **WLS correction**: By weighting each observation inversely proportional to its variance, WLS gives more weight to precise observations (small $x$) and less weight to noisy ones (large $x$). The weighted residual plot should show stabilized variance.
- **Standard errors**: WLS standard errors are typically smaller than the (incorrect) OLS standard errors, leading to more powerful tests. The OLS standard errors under heteroscedasticity may be either too large or too small, depending on the pattern.
- **Practical note**: In practice, the true variance function $\sigma_i^2$ is unknown. Common approaches include estimating it from a preliminary regression of squared residuals on the predictors, or using heteroscedasticity-consistent (HC) standard errors as an alternative to WLS.

## Exercises

**Exercise 1.** Generate data with the reverse pattern: variance decreasing with $x$ (e.g., $\sigma_i = 10 - 0.8x_i$). Fit OLS and WLS. Does WLS still outperform OLS in terms of coefficient accuracy?

??? success "Solution to Exercise 1"

    ```python
    sigma_rev = 10 - 0.8 * x
    y_rev = 3.0 + 2.0 * x + np.random.normal(0, sigma_rev)
    w_rev = 1.0 / sigma_rev ** 2
    beta_ols_rev = ols_fit(X, y_rev)
    beta_wls_rev = wls_fit(X, y_rev, w_rev)
    ```

    Yes, WLS still outperforms OLS whenever heteroscedasticity is present, regardless of the direction. WLS uses the correct weights and produces BLUE (Best Linear Unbiased Estimator) under the Gauss-Markov theorem generalized to heteroscedastic errors. $\square$

---

**Exercise 2.** Show that WLS with equal weights $w_i = c$ for all $i$ reduces to OLS. What does this tell us about the relationship between the two methods?

??? success "Solution to Exercise 2"

    If $w_i = c$ for all $i$, then $\mathbf{W} = c\mathbf{I}$, so

    $$
    \hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^\top c\mathbf{I}\,\mathbf{X})^{-1}\mathbf{X}^\top c\mathbf{I}\,\mathbf{y} = (c\mathbf{X}^\top\mathbf{X})^{-1}c\mathbf{X}^\top\mathbf{y} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}.
    $$

    OLS is a special case of WLS where all observations are weighted equally, which is appropriate when the homoscedasticity assumption holds. $\square$

---

**Exercise 3.** In practice, we do not know $\sigma_i^2$. Implement feasible WLS by first fitting OLS, then regressing $\ln(e_i^2)$ on $x_i$ to estimate the variance function, and finally applying WLS with the estimated weights.

??? success "Solution to Exercise 3"

    ```python
    resid_ols = y - X @ beta_ols
    log_resid_sq = np.log(resid_ols ** 2 + 1e-10)
    gamma = np.linalg.lstsq(X, log_resid_sq, rcond=None)[0]
    sigma_hat = np.sqrt(np.exp(X @ gamma))
    w_feas = 1.0 / sigma_hat ** 2
    beta_fwls = wls_fit(X, y, w_feas)
    ```

    Feasible WLS uses estimated weights instead of known ones. The estimates are consistent and asymptotically efficient, though they may be less efficient in small samples compared to WLS with known weights. $\square$

---

**Exercise 4.** Prove that $\hat{\boldsymbol{\beta}}_{\text{WLS}}$ is the BLUE (Best Linear Unbiased Estimator) when the covariance structure $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_n^2)$ is known.

??? success "Solution to Exercise 4"

    By the generalized Gauss-Markov theorem, the BLUE of $\boldsymbol{\beta}$ when $\mathrm{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}$ is

    $$
    \hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{y}.
    $$

    When $\boldsymbol{\Sigma}$ is diagonal, $\boldsymbol{\Sigma}^{-1} = \mathrm{diag}(1/\sigma_1^2, \ldots, 1/\sigma_n^2) = \mathbf{W}$. Therefore $\hat{\boldsymbol{\beta}}_{\text{GLS}} = \hat{\boldsymbol{\beta}}_{\text{WLS}}$. "Best" means it has the smallest variance among all linear unbiased estimators, i.e., $\mathrm{Var}(\mathbf{a}^\top\hat{\boldsymbol{\beta}}_{\text{WLS}}) \leq \mathrm{Var}(\mathbf{a}^\top\tilde{\boldsymbol{\beta}})$ for any linear unbiased $\tilde{\boldsymbol{\beta}}$ and any direction $\mathbf{a}$. $\square$

---

**Exercise 5.** Compare WLS to using heteroscedasticity-consistent (HC) standard errors (White's robust standard errors) with OLS. What are the tradeoffs?

??? success "Solution to Exercise 5"

    HC standard errors correct the standard errors of OLS without changing the coefficient estimates:

    $$
    \widehat{\mathrm{Var}}_{\text{HC}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Omega}}\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1},
    $$

    where $\hat{\boldsymbol{\Omega}} = \mathrm{diag}(e_1^2, \ldots, e_n^2)$. Tradeoffs: (1) HC standard errors are valid under arbitrary heteroscedasticity without specifying the variance function, but OLS coefficients are inefficient; (2) WLS produces efficient estimates but requires specifying the correct weight function; (3) If the variance model is misspecified, WLS can introduce bias in the standard errors, while HC standard errors remain robust. In practice, HC standard errors are preferred when the variance function is unknown. $\square$
