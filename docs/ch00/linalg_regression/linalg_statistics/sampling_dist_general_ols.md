# Sampling Distribution of General OLS Estimators

The multivariate OLS estimator has an exact normal sampling distribution under Gaussian errors, derived through the matrix formulation of least squares.

## Definition

Under the model $\mathbf{Y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 I_n)$ and $X$ an $n \times p$ full-rank design matrix, the OLS estimator is

$$
\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{Y}
$$

Its sampling distribution is

$$
\hat{\boldsymbol{\beta}} \sim N\!\left(\boldsymbol{\beta},\; \sigma^2 (X^T X)^{-1}\right)
$$

## Explanation

Since $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{Y}$ is a linear transformation of the normal vector $\mathbf{Y}$, it is multivariate normal. The mean is $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$ (unbiased), and the covariance is $\sigma^2 (X^T X)^{-1}$.

The residual vector $\hat{\boldsymbol{\varepsilon}} = (I - H)\mathbf{Y}$ where $H = X(X^T X)^{-1}X^T$ is the hat matrix. Because $I - H$ is symmetric idempotent with rank $n - p$:

$$
\frac{\hat{\boldsymbol{\varepsilon}}^T \hat{\boldsymbol{\varepsilon}}}{\sigma^2} = \frac{(n-p)S^2}{\sigma^2} \sim \chi^2(n-p)
$$

The independence of $\hat{\boldsymbol{\beta}}$ and $S^2$ (since $H$ and $I - H$ project onto orthogonal subspaces) gives the pivotal quantities

$$
\frac{\hat{\beta}_j - \beta_j}{S\sqrt{c_{jj}}} \sim t(n-p)
$$

where $c_{jj}$ is the $j$-th diagonal entry of $(X^T X)^{-1}$.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n, p = 100, 4
beta_true = np.array([1.0, -2.0, 0.5, 3.0])
sigma = 2.0

X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
XtX_inv = np.linalg.inv(X.T @ X)
true_cov = sigma**2 * XtX_inv

# Simulate OLS estimates
n_sims = 20_000
betas = np.empty((n_sims, p))
for i in range(n_sims):
    y = X @ beta_true + np.random.normal(0, sigma, n)
    betas[i] = XtX_inv @ X.T @ y

# Compare simulated covariance to theoretical
sim_cov = np.cov(betas, rowvar=False)
print("Theoretical diagonal:", np.diag(true_cov).round(5))
print("Simulated diagonal:  ", np.diag(sim_cov).round(5))

# Check t-distribution for beta_1
j = 1
t_vals = np.empty(n_sims)
for i in range(n_sims):
    y = X @ beta_true + np.random.normal(0, sigma, n)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    s2 = resid @ resid / (n - p)
    t_vals[i] = (b[j] - beta_true[j]) / np.sqrt(s2 * XtX_inv[j, j])

stat, pval = stats.kstest(t_vals, 't', args=(n - p,))
print(f"KS test for t({n-p}): p = {pval:.4f}")
```
