# F-Test for Two Variances

## Overview

The F-test for two variances compares the variances of two independent normal populations. It is used to assess whether two groups have equal variability, which is a prerequisite for the pooled two-sample t-test. The test statistic follows an F-distribution under the null hypothesis of equal variances. Like the chi-squared variance test, it is sensitive to departures from normality.

## Test Formulation

**Hypotheses:** Given two independent normal populations with variances $\sigma_1^2$ and $\sigma_2^2$,

- Two-sided: $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$ vs $H_1\colon \sigma_1^2/\sigma_2^2 \neq \theta_0$
- One-sided: $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$ vs $H_1\colon \sigma_1^2/\sigma_2^2 > \theta_0$

The most common case is $\theta_0 = 1$ (testing equality of variances).

**Test statistic:** Given sample standard deviations $S_1$ and $S_2$ from samples of sizes $n_1$ and $n_2$,

$$
F = \frac{S_1^2 / S_2^2}{\theta_0} \sim F_{n_1-1,\,n_2-1}
$$

under $H_0$ and the normality assumption.

## Code

```python
from scipy.stats import f

def test_ratio_two_variances(n1, s1, n2, s2, theta0=1.0,
                              alt="two-sided", alpha=0.05):
    """
    H0: sigma1^2 / sigma2^2 = theta0 (Normal populations).
    Provide s1, s2 = sample std (ddof=1). Returns (F, p, reject).
    """
    df1, df2 = n1 - 1, n2 - 1
    F_stat = (s1 ** 2 / s2 ** 2) / theta0
    if alt == "two-sided":
        p = 2 * min(f.cdf(F_stat, df1, df2),
                     1 - f.cdf(F_stat, df1, df2))
    elif alt == "less":
        p = f.cdf(F_stat, df1, df2)
    else:
        p = 1 - f.cdf(F_stat, df1, df2)
    return F_stat, p, (p < alpha)
```

### Example

```python
F_stat, p, reject = test_ratio_two_variances(
    n1=15, s1=1.3, n2=12, s2=0.9, theta0=1.0, alt="greater"
)
print("F:", F_stat, "p:", p, "reject:", reject)
```

### Interpretation

We test $H_0\colon \sigma_1^2 = \sigma_2^2$ vs $H_1\colon \sigma_1^2 > \sigma_2^2$ with $s_1 = 1.3$, $n_1 = 15$ and $s_2 = 0.9$, $n_2 = 12$. The F-statistic is

$$
F = \frac{1.3^2}{0.9^2} = \frac{1.69}{0.81} \approx 2.086.
$$

With $(14, 11)$ degrees of freedom, the one-sided p-value $P(F_{14,11} \geq 2.086)$ determines whether we reject $H_0$.

## Exercises

**Exercise 1.** Two production lines yield samples with $s_1 = 4.2$ ($n_1 = 20$) and $s_2 = 3.1$ ($n_2 = 25$). Test $H_0\colon \sigma_1^2 = \sigma_2^2$ vs $H_1\colon \sigma_1^2 \neq \sigma_2^2$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    The F-statistic is

    $$
    F = \frac{4.2^2}{3.1^2} = \frac{17.64}{9.61} \approx 1.836.
    $$

    Degrees of freedom: $(19, 24)$. For a two-sided test, the p-value is $2 \times \min(P(F \leq 1.836),\, P(F \geq 1.836))$. Using tables or software, $P(F_{19,24} \geq 1.836) \approx 0.082$, so the two-sided p-value is approximately $0.164$. Since $0.164 > 0.05$, we fail to reject $H_0$. $\square$

---

**Exercise 2.** Derive the F-test statistic from the ratio of two independent chi-squared random variables.

??? success "Solution to Exercise 2"

    Under normality, $(n_i - 1)S_i^2/\sigma_i^2 \sim \chi^2_{n_i - 1}$ for $i = 1, 2$, and the two are independent. The F-distribution is defined as the ratio of two independent chi-squared variables divided by their degrees of freedom:

    $$
    F = \frac{\chi^2_{n_1-1}/(n_1-1)}{\chi^2_{n_2-1}/(n_2-1)} = \frac{(n_1-1)S_1^2/\sigma_1^2/(n_1-1)}{(n_2-1)S_2^2/\sigma_2^2/(n_2-1)} = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}.
    $$

    Under $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$, this simplifies to $F = (S_1^2/S_2^2)/\theta_0 \sim F_{n_1-1,\,n_2-1}$. $\square$

---

**Exercise 3.** Explain why the F-test for variances is more sensitive to non-normality than the t-test for means. What alternatives exist?

??? success "Solution to Exercise 3"

    The t-test is robust to moderate non-normality because the CLT ensures the sampling distribution of $\bar{X}$ is approximately normal for moderate $n$. However, the distribution of $S^2$ depends on the kurtosis of the population. For heavy-tailed distributions, $S^2$ has much larger variability than predicted by the chi-squared distribution, which distorts the F-ratio. The F-test for variances therefore has inflated Type I error rates under non-normality.

    Alternatives include:

    - **Levene's test**: uses the mean of absolute deviations, robust to non-normality.
    - **Brown--Forsythe test**: uses the median instead of the mean in Levene's test.
    - **Bartlett's test**: a likelihood ratio test that is still sensitive to non-normality but often used in ANOVA contexts.
    - **Bootstrap methods**: nonparametric resampling-based variance comparison. $\square$

---

**Exercise 4.** Show that if $F \sim F_{\nu_1, \nu_2}$, then $1/F \sim F_{\nu_2, \nu_1}$.

??? success "Solution to Exercise 4"

    By definition, $F = (U/\nu_1)/(V/\nu_2)$ where $U \sim \chi^2_{\nu_1}$ and $V \sim \chi^2_{\nu_2}$ are independent. Then

    $$
    \frac{1}{F} = \frac{V/\nu_2}{U/\nu_1},
    $$

    which is the ratio of $\chi^2_{\nu_2}/\nu_2$ to $\chi^2_{\nu_1}/\nu_1$. By definition this is $F_{\nu_2,\nu_1}$. $\square$

---

**Exercise 5.** In a two-sided F-test with $\alpha = 0.10$, $n_1 = 10$, and $n_2 = 8$, find the critical values $F_L$ and $F_U$ such that $P(F_L < F < F_U) = 0.90$.

??? success "Solution to Exercise 5"

    The degrees of freedom are $(\nu_1, \nu_2) = (9, 7)$. The upper critical value is

    $$
    F_U = F_{0.95,\,9,\,7} \approx 3.677.
    $$

    For the lower critical value, we use the reciprocal property:

    $$
    F_L = \frac{1}{F_{0.95,\,7,\,9}} \approx \frac{1}{3.293} \approx 0.304.
    $$

    The acceptance region is $0.304 < F < 3.677$. We reject $H_0$ if the observed F falls outside this interval. $\square$
