# Testing Pearson's r (t-Test for Correlation)

Computing a sample correlation $r$ tells us the observed strength of linear association, but we also need to determine whether this observed value is statistically significant -- that is, whether it provides evidence against the null hypothesis that the population correlation is zero. The standard test for the Pearson correlation uses a t-statistic and is one of the most commonly performed hypothesis tests in applied statistics.

---

## Hypotheses

The most common test is:

$$
H_0\!: \rho = 0 \quad \text{vs} \quad H_1\!: \rho \neq 0
$$

where $\rho$ is the population Pearson correlation coefficient. One-sided alternatives $H_1\!: \rho > 0$ or $H_1\!: \rho < 0$ are also used when the direction of the association is predicted in advance.

---

## Test Statistic

Under $H_0\!: \rho = 0$ and the assumption that $(X, Y)$ follows a bivariate normal distribution, the statistic

$$
t = \frac{r\sqrt{n-2}}{\sqrt{1 - r^2}}
$$

follows a **Student's t-distribution** with $n - 2$ degrees of freedom, where $r$ is the sample Pearson correlation and $n$ is the sample size.

### Derivation Sketch

The sample correlation $r$ can be written as the slope of the standardized regression of $Y$ on $X$. Under $H_0$, the true slope is zero, and the usual regression t-test applies. The degrees of freedom are $n - 2$ because two parameters (intercept and slope) are estimated.

---

## Decision Rule

For a two-sided test at significance level $\alpha$:

- Reject $H_0$ if $|t| > t_{\alpha/2, \, n-2}$
- Equivalently, reject if the p-value $< \alpha$

The p-value is

$$
p = 2 \cdot P(T_{n-2} > |t|)
$$

where $T_{n-2}$ is a $t$-distributed random variable with $n-2$ degrees of freedom.

For one-sided tests:

- $H_1\!: \rho > 0$: reject if $t > t_{\alpha, \, n-2}$
- $H_1\!: \rho < 0$: reject if $t < -t_{\alpha, \, n-2}$

---

## Example

A researcher collects data from $n = 25$ students and finds a sample correlation of $r = 0.45$ between study hours and exam scores.

$$
t = \frac{0.45\sqrt{25 - 2}}{\sqrt{1 - 0.45^2}} = \frac{0.45 \times 4.796}{\sqrt{0.7975}} = \frac{2.158}{0.8931} = 2.417
$$

With $n - 2 = 23$ degrees of freedom, the critical value for a two-sided test at $\alpha = 0.05$ is $t_{0.025, 23} = 2.069$. Since $|t| = 2.417 > 2.069$, we reject $H_0$ and conclude that there is a statistically significant positive linear correlation between study hours and exam scores.

---

## Confidence Interval for the Population Correlation

To construct a confidence interval for $\rho$, we use **Fisher's z-transformation** because the sampling distribution of $r$ is skewed (especially when $\rho$ is far from zero):

1. Transform: $z = \text{arctanh}(r) = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)$

2. The approximate standard error of $z$ is $\text{SE}_z = \frac{1}{\sqrt{n-3}}$

3. Construct the confidence interval for the transformed parameter:

    $$
    z \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n-3}}
    $$

4. Back-transform each endpoint using $\tanh$ to obtain the confidence interval for $\rho$.

### Example (continued)

With $r = 0.45$ and $n = 25$:

$$
z = \text{arctanh}(0.45) = 0.4847
$$

$$
\text{SE}_z = \frac{1}{\sqrt{22}} = 0.2132
$$

A 95% confidence interval for $\zeta = \text{arctanh}(\rho)$:

$$
0.4847 \pm 1.96 \times 0.2132 = (0.0668, \; 0.9026)
$$

Back-transforming: $(\tanh(0.0668), \; \tanh(0.9026)) = (0.067, \; 0.717)$

The 95% confidence interval for $\rho$ is approximately $(0.07, 0.72)$.

---

## Testing Against a Nonzero Value

To test $H_0\!: \rho = \rho_0$ for some $\rho_0 \neq 0$, the t-test above does not apply (its null distribution depends on $\rho = 0$). Instead, use the Fisher z-transformation:

$$
Z = \frac{\text{arctanh}(r) - \text{arctanh}(\rho_0)}{1/\sqrt{n-3}}
$$

Under $H_0$, $Z$ follows approximately a standard normal distribution.

---

## Assumptions

The t-test for correlation requires:

1. **Bivariate normality**: both $X$ and $Y$ are normally distributed (or at least the conditional distribution of $Y \mid X$ is normal with constant variance).
2. **Random sampling**: observations are independent.
3. **Linearity**: the relationship between $X$ and $Y$ is linear.

!!! warning "Large samples do not fix non-linearity"
    With large $n$, even a tiny $r$ will be statistically significant. A significant p-value tells you $\rho \neq 0$ but says nothing about the practical importance of the correlation. Always report and interpret $r$ alongside the p-value.

When normality is violated, permutation tests or bootstrap methods provide nonparametric alternatives.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Sample data
np.random.seed(42)
n = 25
x = np.random.normal(5, 2, n)
y = 0.5 * x + np.random.normal(0, 1, n)

# Test H0: rho = 0
r, p_value = stats.pearsonr(x, y)
print(f"r = {r:.4f}")
print(f"p-value = {p_value:.4f}")

# Manual t-statistic
t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
p_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
print(f"t-statistic = {t_stat:.4f}")
print(f"Manual p-value = {p_manual:.4f}")

# Fisher z confidence interval
z = np.arctanh(r)
se_z = 1 / np.sqrt(n - 3)
ci_z = (z - 1.96 * se_z, z + 1.96 * se_z)
ci_r = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
print(f"95% CI for rho: ({ci_r[0]:.3f}, {ci_r[1]:.3f})")
```

---

## Summary

The t-test for Pearson's $r$ determines whether the observed sample correlation provides statistically significant evidence against $H_0\!: \rho = 0$. The test statistic $t = r\sqrt{n-2}/\sqrt{1-r^2}$ follows a $t$-distribution with $n-2$ degrees of freedom under bivariate normality. Confidence intervals for $\rho$ use Fisher's z-transformation to handle the skewed sampling distribution of $r$. The test requires bivariate normality, independence, and a linear relationship; when these assumptions are violated, rank-based tests such as [Spearman's test](test_spearman.md) or [Kendall's test](test_kendall.md) are preferred.
