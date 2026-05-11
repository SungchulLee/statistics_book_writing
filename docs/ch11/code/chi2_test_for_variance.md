# Chi2 Test For Variance

## Overview

The chi-squared test for variance is a one-sample test that assesses whether the variance of a normally distributed population equals a pre-specified value $\sigma_0^2$. It is the variance analogue of the one-sample $t$-test for the mean. This page derives the test statistic, shows its connection to the chi-squared distribution, and demonstrates it across several variance-ratio scenarios using Python.

## Hypotheses and Test Statistic

Given a random sample $X_1, \dots, X_n$ from $N(\mu, \sigma^2)$, the two-sided test is

$$
H_0: \sigma^2 = \sigma_0^2, \qquad H_1: \sigma^2 \neq \sigma_0^2
$$

The test statistic is

$$
T = \frac{(n - 1)\, S^2}{\sigma_0^2}
$$

where $S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2$ is the sample variance. Under $H_0$, this statistic follows a chi-squared distribution:

$$
T \sim \chi^2(n - 1)
$$

For a two-sided test at significance level $\alpha$, we reject $H_0$ when

$$
T < \chi^2_{\alpha/2}(n-1) \quad \text{or} \quad T > \chi^2_{1-\alpha/2}(n-1)
$$

The two-sided p-value is

$$
p = 2 \min\!\bigl(P(\chi^2_{n-1} \le T),\; P(\chi^2_{n-1} \ge T)\bigr)
$$

## Implementation

The following function implements the test from scratch:

```python
import numpy as np
import scipy.stats as stats

def chi2_test_for_variance(data, sigma2_0=1.0):
    n = len(data)
    s2 = np.var(data, ddof=1)
    statistic = (n - 1) * s2 / sigma2_0
    p_value = 2 * min(
        stats.chi2(df=n - 1).cdf(statistic),
        stats.chi2(df=n - 1).sf(statistic)
    )
    return statistic, p_value
```

The demonstration generates samples from $N(1, \sigma_Y^2)$ for $\sigma_Y \in \{1.00, 1.05, 1.10, 1.15, 1.20\}$ and tests $H_0: \sigma^2 = 1$:

```python
x = stats.norm(loc=0, scale=1).rvs(100, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(100, random_state=seed)
    stat, pval = chi2_test_for_variance(y, sigma2_0=1.0)
    print(f"sigma={scale:.2f}: p={pval:.3f}")
```

## Interpretation

- When $\sigma_Y = 1.00$, the true variance matches $\sigma_0^2 = 1$ and the p-value is large, correctly retaining $H_0$.
- As $\sigma_Y$ increases beyond 1.00, the sample variance $S^2$ tends to exceed 1, pushing $T$ into the upper tail of $\chi^2(n-1)$ and producing smaller p-values.
- With $n = 100$, the test has reasonable power to detect moderate departures from $\sigma_0^2 = 1$ (e.g., $\sigma_Y = 1.15$).

An important limitation is that this test is valid only when the underlying population is normal. For non-normal data, the sampling distribution of $S^2$ deviates from the chi-squared form, and the test's Type I error rate is no longer controlled at the nominal level.

## Exercises

**Exercise 1.**
A sample of $n = 25$ measurements yields $S^2 = 12.5$. Test $H_0: \sigma^2 = 10$ at $\alpha = 0.05$. Compute the test statistic and state your conclusion.

??? success "Solution to Exercise 1"
    The test statistic is

    $$
    T = \frac{(n-1) S^2}{\sigma_0^2} = \frac{24 \times 12.5}{10} = 30.0
    $$

    Under $H_0$, $T \sim \chi^2(24)$. The critical values at $\alpha = 0.05$ (two-sided) are $\chi^2_{0.025}(24) = 12.40$ and $\chi^2_{0.975}(24) = 39.36$. Since $12.40 < 30.0 < 39.36$, the test statistic falls inside the acceptance region. We fail to reject $H_0$ at the 5% level.

---

**Exercise 2.**
Derive a $(1 - \alpha)$ confidence interval for $\sigma^2$ using the pivotal quantity $T = (n-1)S^2 / \sigma^2$.

??? success "Solution to Exercise 2"
    Since $(n-1)S^2 / \sigma^2 \sim \chi^2(n-1)$ under normality, we have

    $$
    P\!\left(\chi^2_{\alpha/2}(n-1) \le \frac{(n-1)S^2}{\sigma^2} \le \chi^2_{1-\alpha/2}(n-1)\right) = 1 - \alpha
    $$

    Inverting the inequalities:

    $$
    \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)} \le \sigma^2 \le \frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}
    $$

    This is the $(1-\alpha)$ confidence interval for $\sigma^2$. Note that the larger chi-squared quantile appears in the denominator of the lower bound, producing the correct orientation. $\square$

---

**Exercise 3.**
Explain why the two-sided p-value uses $2\min(\cdot, \cdot)$ rather than simply $2 \cdot P(\chi^2_{n-1} \ge T)$.

??? success "Solution to Exercise 3"
    The chi-squared distribution is not symmetric, so the two tails have different shapes and areas. The test can reject in either direction: $T$ may be unusually small (suggesting $\sigma^2 < \sigma_0^2$) or unusually large (suggesting $\sigma^2 > \sigma_0^2$). The p-value must measure the probability of a result at least as extreme as observed in whichever tail $T$ falls into, then double it to account for the two-sided nature.

    Using $2 \cdot P(\chi^2 \ge T)$ would only be correct if $T$ falls in the upper tail. When $T$ is in the lower tail, $P(\chi^2 \ge T)$ is close to 1 and doubling it gives a nonsensical value exceeding 1. The $\min$ formulation correctly identifies the relevant tail and doubles the smaller tail probability.

---

**Exercise 4.**
A factory produces bolts with a target variance of $\sigma_0^2 = 0.04 \text{ mm}^2$. A sample of $n = 50$ bolts yields $S^2 = 0.06 \text{ mm}^2$. Should the factory recalibrate its machine? Use $\alpha = 0.01$.

??? success "Solution to Exercise 4"
    The test statistic is

    $$
    T = \frac{(50 - 1)(0.06)}{0.04} = \frac{2.94}{0.04} = 73.5
    $$

    Under $H_0$, $T \sim \chi^2(49)$. The upper critical value at $\alpha/2 = 0.005$ is $\chi^2_{0.995}(49) \approx 79.49$. Since $73.5 < 79.49$, the test statistic does not exceed the upper critical value. The lower critical value is $\chi^2_{0.005}(49) \approx 27.99$, and $73.5 > 27.99$. Therefore $T$ falls inside the acceptance region, and we fail to reject $H_0$ at the 1% level. The data do not provide sufficient evidence that the variance differs from the target, so immediate recalibration is not warranted. However, the sample variance is 50% larger than the target, so continued monitoring is prudent.

---

**Exercise 5.**
This test assumes normality. Describe what happens to the Type I error rate when the underlying distribution has excess kurtosis $\kappa > 0$, and suggest an alternative approach.

??? success "Solution to Exercise 5"
    When the population has excess kurtosis $\kappa > 0$ (heavier tails than normal), the sampling distribution of $S^2$ has greater variability than the chi-squared theory predicts. Specifically, $\text{Var}(S^2) = \sigma^4 \bigl(\frac{2}{n-1} + \frac{\kappa}{n}\bigr)$, which exceeds the normal-theory value of $\frac{2\sigma^4}{n-1}$. This means the test statistic $T$ has a more dispersed distribution than $\chi^2(n-1)$, causing it to land in the rejection region more often than expected. The result is an inflated Type I error rate.

    A robust alternative is the bootstrap test: resample the data with replacement many times, compute $S^2$ for each resample, and use the empirical distribution of $(n-1)S^2/\sigma_0^2$ to obtain the p-value. This approach makes no distributional assumptions.
