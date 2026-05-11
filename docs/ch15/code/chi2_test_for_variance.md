# Chi-Squared Test for Variance

## Overview

The chi-squared test for variance is a one-sample hypothesis test that determines whether the variance of a normally distributed population equals a hypothesized value. It is built directly on the chi-squared distribution and serves as the variance analog of the one-sample $z$- or $t$-test for the mean. Because the test statistic depends on the ratio of the sample variance to the hypothesized variance, it is especially useful in quality-control settings where a process must meet a specified variability target.

## Test Setup

Let $X_1, X_2, \ldots, X_n$ be an independent random sample from $N(\mu, \sigma^2)$. We wish to test

$$
H_0 : \sigma^2 = \sigma_0^2 \quad \text{versus} \quad H_1 : \sigma^2 \neq \sigma_0^2
$$

where $\sigma_0^2$ is the hypothesized population variance.

## Test Statistic

Define the sample variance

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2.
$$

Under $H_0$ the test statistic

$$
T = \frac{(n-1) S^2}{\sigma_0^2}
$$

follows a chi-squared distribution with $n - 1$ degrees of freedom:

$$
T \sim \chi^2(n-1).
$$

## Decision Rule

For a two-sided test at significance level $\alpha$, reject $H_0$ when

$$
T < \chi^2_{\alpha/2,\, n-1} \quad \text{or} \quad T > \chi^2_{1-\alpha/2,\, n-1},
$$

where $\chi^2_{q,\, n-1}$ denotes the $q$-th quantile of $\chi^2(n-1)$. Equivalently, the two-sided $p$-value is

$$
p = 2 \min\!\bigl(F_{\chi^2}(T),\; 1 - F_{\chi^2}(T)\bigr),
$$

where $F_{\chi^2}$ is the CDF of $\chi^2(n-1)$.

## Code

The function below implements the one-sample chi-squared test for variance.

```python
import numpy as np
import scipy.stats as stats


def chi2_test_for_variance(data, sigma2_0=1.0):
    """
    One-sample chi-squared test for variance.

    H0: sigma^2 = sigma2_0
    H1: sigma^2 != sigma2_0
    """
    n = len(data)
    s2 = np.var(data, ddof=1)
    statistic = (n - 1) * s2 / sigma2_0
    p_value = 2 * min(
        stats.chi2(df=n - 1).cdf(statistic),
        stats.chi2(df=n - 1).sf(statistic),
    )
    return statistic, p_value
```

A typical usage generates data under varying true standard deviations and checks whether the test detects a departure from $\sigma_0^2 = 1$:

```python
import matplotlib.pyplot as plt

size, seed = 100, 0
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = chi2_test_for_variance(y, sigma2_0=1.0)
    print(f"sigma={scale:.2f}  T={stat:.2f}  p={pval:.3f}")
```

## Interpretation

- When the true variance equals the hypothesized value ($\sigma = 1.00$), the $p$-value is typically large, so we fail to reject $H_0$.
- As the true standard deviation increases (e.g., $\sigma = 1.10, 1.15, 1.20$), the test statistic grows and the $p$-value decreases, providing stronger evidence against $H_0$.
- The test assumes normality. For heavy-tailed or skewed data, the actual Type I error rate can deviate substantially from the nominal level $\alpha$.

## Exercises

**Exercise 1.** A manufacturing process is designed so that the diameter of a part has variance $\sigma_0^2 = 0.04\;\text{mm}^2$. A sample of $n = 25$ parts yields $S^2 = 0.06$. Compute the chi-squared test statistic and, using Python, obtain the two-sided $p$-value. State your conclusion at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    The test statistic is

    $$
    T = \frac{(25 - 1)(0.06)}{0.04} = \frac{1.44}{0.04} = 36.
    $$

    ```python
    import scipy.stats as stats

    T = 24 * 0.06 / 0.04  # 36.0
    p = 2 * min(stats.chi2(df=24).cdf(T), stats.chi2(df=24).sf(T))
    print(f"T = {T:.1f}, p = {p:.4f}")
    ```

    With $\chi^2(24)$, the right-tail probability $P(\chi^2 > 36) \approx 0.054$, giving $p \approx 0.108$. At $\alpha = 0.05$ we fail to reject $H_0$; the evidence is insufficient to conclude the variance differs from $0.04$.

---

**Exercise 2.** Derive the test statistic $T = (n-1)S^2 / \sigma_0^2$ starting from the definition of the sample variance and the fact that under $H_0$ each $(X_i - \mu)/\sigma_0 \sim N(0,1)$. Explain why the degrees of freedom are $n - 1$ rather than $n$.

??? success "Solution to Exercise 2"

    Write $Z_i = (X_i - \mu)/\sigma_0$. Then $\sum_{i=1}^n Z_i^2 \sim \chi^2(n)$. However, $\mu$ is unknown and replaced by $\bar{X}$. The constraint $\sum (X_i - \bar{X}) = 0$ removes one degree of freedom, so

    $$
    \sum_{i=1}^{n} \frac{(X_i - \bar{X})^2}{\sigma_0^2} = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1).
    $$

    This is the projection of the $n$-dimensional standard normal vector onto the $(n-1)$-dimensional subspace orthogonal to the all-ones vector, which by Cochran's theorem has a $\chi^2(n-1)$ distribution. $\square$

---

**Exercise 3.** Write a simulation (at least 5000 replications) that draws samples of size $n = 30$ from $N(0,1)$ and applies the chi-squared test with $\sigma_0^2 = 1$ at $\alpha = 0.05$. Estimate the empirical Type I error rate and verify it is close to 0.05.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    n, sigma2_0, alpha, n_sims = 30, 1.0, 0.05, 10000
    rejections = 0

    for _ in range(n_sims):
        x = rng.normal(0, 1, size=n)
        s2 = np.var(x, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        p = 2 * min(stats.chi2(df=n - 1).cdf(T), stats.chi2(df=n - 1).sf(T))
        if p < alpha:
            rejections += 1

    print(f"Empirical Type I error: {rejections / n_sims:.4f}")
    ```

    The result should be approximately 0.05, confirming the test maintains its nominal size under normality.

---

**Exercise 4.** Repeat the simulation from Exercise 3 but draw from a $t(5)$ distribution (heavier tails) instead of a normal. Compare the empirical rejection rate to $\alpha = 0.05$ and explain why the chi-squared test is problematic for non-normal data.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    n, sigma2_0, alpha, n_sims = 30, 5 / 3, 0.05, 10000
    # Var(t(5)) = 5/(5-2) = 5/3
    rejections = 0

    for _ in range(n_sims):
        x = stats.t(df=5).rvs(size=n, random_state=rng)
        s2 = np.var(x, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        p = 2 * min(stats.chi2(df=n - 1).cdf(T), stats.chi2(df=n - 1).sf(T))
        if p < alpha:
            rejections += 1

    print(f"Empirical rejection rate (t(5)): {rejections / n_sims:.4f}")
    ```

    The rejection rate will be well above 0.05 (often around 0.10--0.15). The chi-squared test for variance is sensitive to kurtosis: heavier tails inflate $S^2$, causing the test to reject $H_0$ too often even when the true variance matches $\sigma_0^2$.

---

**Exercise 5.** Construct the one-sided test $H_0 : \sigma^2 \le \sigma_0^2$ versus $H_1 : \sigma^2 > \sigma_0^2$. Write the rejection rule in terms of $\chi^2_{1-\alpha,\, n-1}$ and modify the Python function `chi2_test_for_variance` to return the one-sided $p$-value.

??? success "Solution to Exercise 5"

    For the right-sided alternative $H_1: \sigma^2 > \sigma_0^2$, reject $H_0$ when

    $$
    T = \frac{(n-1)S^2}{\sigma_0^2} > \chi^2_{1-\alpha,\, n-1}.
    $$

    The one-sided $p$-value is $p = P(\chi^2(n-1) \ge T) = 1 - F_{\chi^2}(T)$.

    ```python
    import numpy as np
    import scipy.stats as stats

    def chi2_test_one_sided(data, sigma2_0=1.0):
        n = len(data)
        s2 = np.var(data, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        p_value = stats.chi2(df=n - 1).sf(T)
        return T, p_value
    ```

    This returns the survival-function value, which is the probability of observing a test statistic at least as extreme in the right tail. $\square$
