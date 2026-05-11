# One-Sample Variance Test

## Overview

The one-sample variance test (chi-squared test for variance) assesses whether the population variance $\sigma^2$ equals a hypothesized value $\sigma_0^2$. This test assumes the underlying population is normally distributed. It is commonly used in quality control to verify that a manufacturing process maintains acceptable variability.

## Test Formulation

**Hypotheses:**

- Two-sided: $H_0\colon \sigma^2 = \sigma_0^2$ vs $H_1\colon \sigma^2 \neq \sigma_0^2$
- One-sided: $H_0\colon \sigma^2 = \sigma_0^2$ vs $H_1\colon \sigma^2 > \sigma_0^2$ (or $< \sigma_0^2$)

**Test statistic:** Given the sample variance $S^2$ (with $\text{ddof}=1$) from a sample of size $n$,

$$
\chi^2 = \frac{(n-1)\,S^2}{\sigma_0^2} \sim \chi^2_{n-1}
$$

under $H_0$ and the assumption of normality.

For a **two-sided** test at level $\alpha$, reject $H_0$ if

$$
\chi^2 < \chi^2_{\alpha/2,\,n-1} \quad \text{or} \quad \chi^2 > \chi^2_{1-\alpha/2,\,n-1}.
$$

## Code

```python
from scipy.stats import chi2

def test_variance_one_sample(n, s2, sigma0, alt="two-sided", alpha=0.05):
    """
    H0: sigma^2 = sigma0^2 (Normal population assumed).
    Provide s2 = sample variance (ddof=1).
    Returns (chi2_stat, pvalue, reject_bool).
    """
    df = n - 1
    chi2_stat = df * s2 / (sigma0 ** 2)
    if alt == "two-sided":
        p = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
    elif alt == "less":
        p = chi2.cdf(chi2_stat, df)
    else:
        p = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p, (p < alpha)
```

### Example

```python
stat, p, reject = test_variance_one_sample(
    n=12, s2=2.1**2, sigma0=2.0, alt="greater"
)
print("chi2:", stat, "p:", p, "reject:", reject)
```

### Interpretation

We test $H_0\colon \sigma^2 = 4.0$ vs $H_1\colon \sigma^2 > 4.0$ with $n=12$ and $s^2 = 4.41$. The test statistic is

$$
\chi^2 = \frac{11 \times 4.41}{4.0} = 12.1275.
$$

Under $H_0$, $\chi^2 \sim \chi^2_{11}$. The one-sided p-value $P(\chi^2_{11} \geq 12.1275) \approx 0.353$, so we fail to reject $H_0$.

## Exercises

**Exercise 1.** A machine fills bottles with a target variance of $\sigma_0^2 = 0.01$ mL$^2$. A sample of $n = 25$ bottles has $s^2 = 0.015$. Test $H_0\colon \sigma^2 = 0.01$ vs $H_1\colon \sigma^2 > 0.01$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    The test statistic is

    $$
    \chi^2 = \frac{24 \times 0.015}{0.01} = 36.0.
    $$

    Under $H_0$, $\chi^2 \sim \chi^2_{24}$. The critical value is $\chi^2_{0.05,\,24} = 36.415$. Since $36.0 < 36.415$, we fail to reject $H_0$ (barely). The p-value is $P(\chi^2_{24} \geq 36.0) \approx 0.055$. $\square$

---

**Exercise 2.** Explain why the chi-squared test for variance is sensitive to the normality assumption. What happens if the population is heavy-tailed?

??? success "Solution to Exercise 2"

    The test statistic $(n-1)S^2/\sigma_0^2 \sim \chi^2_{n-1}$ holds exactly only when the data come from a normal distribution. The chi-squared distribution of the sample variance depends on the fourth moment (kurtosis) of the population. For heavy-tailed distributions (e.g., $t$-distributions with small degrees of freedom), the sample variance $S^2$ has greater variability than predicted by the chi-squared distribution. This means the actual Type I error rate can be much larger than the nominal $\alpha$, and the test becomes unreliable. In such cases, robust alternatives (e.g., Levene's test or bootstrap methods) are preferred. $\square$

---

**Exercise 3.** Derive the distribution of $(n-1)S^2/\sigma^2$ under normality.

??? success "Solution to Exercise 3"

    Let $X_1, \dots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$. Define $Z_i = (X_i - \mu)/\sigma \overset{\text{iid}}{\sim} N(0,1)$. Then $\sum Z_i^2 \sim \chi^2_n$. The sample variance is

    $$
    S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2.
    $$

    By Cochran's theorem, $\sum (X_i - \bar{X})^2 / \sigma^2 \sim \chi^2_{n-1}$ because projecting onto the orthogonal complement of the constant vector reduces the dimension by 1. Therefore

    $$
    \frac{(n-1)S^2}{\sigma^2} = \frac{\sum(X_i - \bar{X})^2}{\sigma^2} \sim \chi^2_{n-1}.
    $$

    Under $H_0\colon \sigma^2 = \sigma_0^2$, substituting $\sigma_0^2$ for $\sigma^2$ gives the test statistic. $\square$

---

**Exercise 4.** For a two-sided test with $n = 20$ and $\alpha = 0.05$, find the critical values $\chi^2_{L}$ and $\chi^2_{U}$ and the acceptance region for $\chi^2$.

??? success "Solution to Exercise 4"

    With $\text{df} = 19$ and $\alpha/2 = 0.025$:

    $$
    \chi^2_L = \chi^2_{0.025,\,19} = 8.907, \qquad \chi^2_U = \chi^2_{0.975,\,19} = 32.852.
    $$

    The acceptance region (fail to reject $H_0$) is $8.907 \leq \chi^2 \leq 32.852$. $\square$

---

**Exercise 5.** A quality engineer collects $n = 15$ measurements with $s = 3.2$. Construct a 95% confidence interval for $\sigma^2$ and use it to test $H_0\colon \sigma^2 = 9$.

??? success "Solution to Exercise 5"

    A $100(1-\alpha)\%$ CI for $\sigma^2$ is

    $$
    \left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}},\; \frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}\right).
    $$

    With $n=15$, $s^2 = 10.24$, $\text{df}=14$, $\chi^2_{0.975,14} = 26.119$, and $\chi^2_{0.025,14} = 5.629$:

    $$
    \left(\frac{14 \times 10.24}{26.119},\; \frac{14 \times 10.24}{5.629}\right) = \left(\frac{143.36}{26.119},\; \frac{143.36}{5.629}\right) = (5.49,\; 25.47).
    $$

    Since $\sigma_0^2 = 9$ lies inside the interval $(5.49, 25.47)$, we fail to reject $H_0\colon \sigma^2 = 9$ at $\alpha = 0.05$. $\square$
