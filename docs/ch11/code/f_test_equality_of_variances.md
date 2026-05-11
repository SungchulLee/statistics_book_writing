# F Test Equality Of Variances

## Overview

The F-test for equality of variances compares the variances of two normally distributed populations by forming the ratio of sample variances. It is the classical two-sample test for homoscedasticity and forms the basis of many other procedures, including the ANOVA F-test. However, it is extremely sensitive to violations of normality, and Levene's test is generally preferred in practice. This page derives the test, implements it from scratch, and examines its behavior across several variance-ratio scenarios.

## Hypotheses and Test Statistic

Given two independent samples of sizes $n_1$ and $n_2$ from normal populations with variances $\sigma_1^2$ and $\sigma_2^2$, the hypotheses are

$$
H_0: \sigma_1^2 = \sigma_2^2, \qquad H_1: \sigma_1^2 \neq \sigma_2^2
$$

The test statistic is the ratio of sample variances:

$$
F = \frac{S_1^2}{S_2^2}
$$

Under $H_0$, this ratio follows an F-distribution:

$$
F \sim F(n_1 - 1,\; n_2 - 1)
$$

For a two-sided test at significance level $\alpha$, we reject $H_0$ when

$$
F < F_{\alpha/2}(n_1 - 1,\, n_2 - 1) \quad \text{or} \quad F > F_{1-\alpha/2}(n_1 - 1,\, n_2 - 1)
$$

The two-sided p-value is

$$
p = 2\min\!\bigl(P(F_{n_1-1,\, n_2-1} \le F_{\text{obs}}),\; P(F_{n_1-1,\, n_2-1} \ge F_{\text{obs}})\bigr)
$$

## Connection to the Chi-Squared Distribution

The F-distribution arises as a ratio of two independent chi-squared random variables, each divided by its degrees of freedom. Since $(n_i - 1)S_i^2 / \sigma_i^2 \sim \chi^2(n_i - 1)$ for normal data, under $H_0: \sigma_1^2 = \sigma_2^2$:

$$
F = \frac{S_1^2}{S_2^2} = \frac{\chi^2(n_1 - 1) / (n_1 - 1)}{\chi^2(n_2 - 1) / (n_2 - 1)}
$$

This is the defining form of the $F(n_1 - 1, n_2 - 1)$ distribution.

## Implementation

The following function implements the two-sided F-test:

```python
import numpy as np
import scipy.stats as stats

def f_test(data_0, data_1):
    statistic = data_0.var(ddof=1) / data_1.var(ddof=1)
    df1 = data_0.shape[0] - 1
    df2 = data_1.shape[0] - 1
    p_value = 2 * min(
        stats.f(df1, df2).cdf(statistic),
        stats.f(df1, df2).sf(statistic)
    )
    return statistic, p_value
```

The demonstration generates $X \sim N(0, 1)$ and $Y \sim N(1, \sigma_Y)$ for several values of $\sigma_Y$:

```python
seed, size = 1, 100
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = f_test(x, y)
    print(f"sigma_y={scale:.2f}: F={stat:.2f}, p={pval:.3f}")
```

## Interpretation

- When $\sigma_Y = 1.00$, the true variance ratio is 1 and the F-statistic is near 1, producing a large p-value. The test correctly retains $H_0$.
- As $\sigma_Y$ increases, $S_Y^2$ grows relative to $S_X^2$, pushing the F-statistic away from 1 and reducing the p-value.
- The test's power depends on the sample size: with $n = 100$ per group, moderate departures (e.g., $\sigma_Y = 1.15$) may or may not be detected, whereas larger samples would detect them reliably.

The critical limitation of this test is its extreme sensitivity to non-normality. Even mild departures from normality (e.g., moderate skewness or a few outliers) can inflate the Type I error rate substantially. For real-world data, Levene's test is recommended.

## Exercises

**Exercise 1.**
Two samples of sizes $n_1 = 20$ and $n_2 = 25$ yield sample variances $S_1^2 = 15.3$ and $S_2^2 = 8.7$. Compute the F-statistic and state the degrees of freedom.

??? success "Solution to Exercise 1"
    The F-statistic is

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{15.3}{8.7} \approx 1.759
    $$

    The degrees of freedom are $df_1 = n_1 - 1 = 19$ and $df_2 = n_2 - 1 = 24$. Under $H_0$, $F \sim F(19, 24)$.

---

**Exercise 2.**
Show that if $F \sim F(d_1, d_2)$, then $1/F \sim F(d_2, d_1)$. Why does this property matter for the two-sided test?

??? success "Solution to Exercise 2"
    By definition, if $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent, then

    $$
    F = \frac{U/d_1}{V/d_2} \sim F(d_1, d_2)
    $$

    Taking the reciprocal:

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1} \sim F(d_2, d_1)
    $$

    This property matters because it means we can always arrange the F-test so that $F \ge 1$ by placing the larger sample variance in the numerator. The two-sided p-value can then be computed as $2 \cdot P(F_{d_1, d_2} \ge F_{\text{obs}})$. Alternatively, the $\min$ formulation in the implementation handles both tails directly without requiring this convention.

---

**Exercise 3.**
Explain why the F-test for equality of variances is more sensitive to non-normality than the two-sample $t$-test for equality of means.

??? success "Solution to Exercise 3"
    The $t$-test is based on sample means, which converge to normality by the Central Limit Theorem regardless of the underlying distribution (provided moments exist). The F-test for variances, however, is based on sample variances, which involve fourth moments of the data. The sampling distribution of $S^2$ is strongly affected by the kurtosis of the underlying distribution. Heavy-tailed distributions produce occasional extreme values that dramatically inflate $S^2$, causing the ratio $S_1^2/S_2^2$ to deviate from the F-distribution far more than the $t$-statistic deviates from the $t$-distribution. This is why the F-test for variances is considered one of the least robust classical tests.

---

**Exercise 4.**
For $n_1 = n_2 = 50$ and $\alpha = 0.05$, find the approximate critical values $F_{0.025}(49, 49)$ and $F_{0.975}(49, 49)$. Use the reciprocal relationship to express one in terms of the other.

??? success "Solution to Exercise 4"
    From F-distribution tables or software, $F_{0.975}(49, 49) \approx 1.607$. By the reciprocal property:

    $$
    F_{0.025}(49, 49) = \frac{1}{F_{0.975}(49, 49)} \approx \frac{1}{1.607} \approx 0.622
    $$

    We reject $H_0$ when $F < 0.622$ or $F > 1.607$. Note that for equal degrees of freedom, the critical region is symmetric about 1 on the log scale: $\ln(0.622) \approx -0.476$ and $\ln(1.607) \approx 0.476$.

---

**Exercise 5.**
A quality engineer measures the variance of a process at two factories and obtains $S_1^2 = 2.1$ from $n_1 = 30$ and $S_2^2 = 3.8$ from $n_2 = 30$. The data show moderate right skewness. Should the engineer use the F-test? Propose a better alternative and explain why.

??? success "Solution to Exercise 5"
    The engineer should not use the F-test because the data are right-skewed, violating the normality assumption that the F-test requires. Even moderate skewness can produce misleading p-values.

    A better alternative is Levene's test, which computes absolute deviations from each group's median and then runs a standard ANOVA on these deviations. Because it operates on absolute deviations rather than squared deviations, it is far less affected by skewness and outliers. In Python: `scipy.stats.levene(data_1, data_2, center='median')`. If the engineer specifically needs a test for the ratio of variances, a bootstrap approach (resampling the variance ratio) provides a distribution-free alternative.
