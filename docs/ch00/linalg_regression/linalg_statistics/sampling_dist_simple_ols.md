# Sampling Distribution of Simple OLS Estimators

Under the normal linear model, the OLS slope and intercept in simple regression have exact sampling distributions that follow from projection matrix algebra.

## Definition

Consider the simple linear regression model $Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ with $\varepsilon_i \overset{iid}{\sim} N(0, \sigma^2)$. The OLS estimators are

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}
$$

Their sampling distributions are

$$
\hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{S_{xx}}\right), \qquad \hat{\beta}_0 \sim N\!\left(\beta_0,\; \sigma^2\!\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)\right)
$$

where $S_{xx} = \sum (x_i - \bar{x})^2$.

## Explanation

Each estimator is a linear combination of the $Y_i$, which are independent normals. A linear combination of independent normal random variables is itself normal, so the only task is computing the mean and variance.

For $\hat{\beta}_1$, write $\hat{\beta}_1 = \sum c_i Y_i$ with $c_i = (x_i - \bar{x})/S_{xx}$. Then $E[\hat{\beta}_1] = \beta_1$ (unbiased) and $\text{Var}(\hat{\beta}_1) = \sigma^2 \sum c_i^2 = \sigma^2/S_{xx}$.

The residual sum of squares satisfies $(n-2)S^2/\sigma^2 \sim \chi^2(n-2)$ and is independent of $\hat{\beta}_1$ (a consequence of the projection decomposition). This independence yields the pivotal quantity

$$
\frac{\hat{\beta}_1 - \beta_1}{S/\sqrt{S_{xx}}} \sim t(n-2)
$$

which is the basis for confidence intervals and hypothesis tests on the slope.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(0)
beta0, beta1, sigma = 2.0, 3.0, 1.5
n = 30

# Simulate many OLS fits
n_sims = 50_000
b1_vals = np.empty(n_sims)
x = np.linspace(0, 10, n)
Sxx = np.sum((x - x.mean())**2)

for i in range(n_sims):
    eps = np.random.normal(0, sigma, n)
    y = beta0 + beta1 * x + eps
    b1_vals[i] = np.sum((x - x.mean()) * (y - y.mean())) / Sxx

theoretical_var = sigma**2 / Sxx
print(f"Simulated mean of b1: {b1_vals.mean():.4f}, expected: {beta1}")
print(f"Simulated var of b1:  {b1_vals.var():.6f}, expected: {theoretical_var:.6f}")

# Verify t-distribution of pivotal quantity
t_vals = np.empty(n_sims)
for i in range(n_sims):
    eps = np.random.normal(0, sigma, n)
    y = beta0 + beta1 * x + eps
    b1 = np.sum((x - x.mean()) * (y - y.mean())) / Sxx
    b0 = y.mean() - b1 * x.mean()
    resid = y - b0 - b1 * x
    s2 = np.sum(resid**2) / (n - 2)
    t_vals[i] = (b1 - beta1) / np.sqrt(s2 / Sxx)

stat, pval = stats.kstest(t_vals, 't', args=(n - 2,))
print(f"KS test against t({n-2}): p = {pval:.4f}")
```
