# F-Distribution and Degrees of Freedom

The F-distribution arises naturally when comparing the variances of two independent normal populations. It is constructed as the ratio of two independent chi-square random variables, each divided by its degrees of freedom. Understanding the F-distribution is essential because it provides the null distribution for the F-test and appears throughout ANOVA and regression analysis.

## Definition

Let $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ be independent chi-square random variables with $d_1$ and $d_2$ degrees of freedom, respectively. The random variable

$$
F = \frac{U / d_1}{V / d_2}
$$

follows an F-distribution with $d_1$ numerator degrees of freedom and $d_2$ denominator degrees of freedom, written $F \sim F_{d_1, d_2}$.

The order of the degrees of freedom matters: $F_{d_1, d_2}$ and $F_{d_2, d_1}$ are different distributions unless $d_1 = d_2$.

## Connection to Sample Variances

Suppose two independent random samples are drawn from normal populations:

$$
X_1, \ldots, X_{n_1} \sim N(\mu_1, \sigma_1^2), \qquad Y_1, \ldots, Y_{n_2} \sim N(\mu_2, \sigma_2^2)
$$

From the chi-square result for sample variances,

$$
\frac{(n_1 - 1)S_1^2}{\sigma_1^2} \sim \chi^2_{n_1 - 1}, \qquad \frac{(n_2 - 1)S_2^2}{\sigma_2^2} \sim \chi^2_{n_2 - 1}
$$

where $S_1^2$ and $S_2^2$ are the sample variances. Because the two samples are independent, these chi-square variables are independent. Forming the ratio gives

$$
F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} = \frac{S_1^2 \sigma_2^2}{S_2^2 \sigma_1^2} \sim F_{n_1-1,\, n_2-1}
$$

Under the null hypothesis $H_0\colon \sigma_1^2 = \sigma_2^2$, the ratio simplifies to

$$
F = \frac{S_1^2}{S_2^2} \sim F_{n_1-1,\, n_2-1}
$$

This is the test statistic for the F-test of equal variances.

## Properties of the F-Distribution

The F-distribution with $d_1$ and $d_2$ degrees of freedom has the following properties:

- **Support:** $F > 0$ (the ratio of positive quantities is always positive)
- **Mean:** $E[F] = \dfrac{d_2}{d_2 - 2}$ for $d_2 > 2$
- **Variance:** $\operatorname{Var}(F) = \dfrac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2-2)^2(d_2-4)}$ for $d_2 > 4$
- **Mode:** $\dfrac{d_1 - 2}{d_1} \cdot \dfrac{d_2}{d_2 + 2}$ for $d_1 > 2$
- **Right-skewed:** The distribution is always positively skewed, though the skewness decreases as both degrees of freedom increase

!!! note "Mean Near 1 Under the Null"
    When $d_2$ is moderately large, $E[F] \approx 1$. This makes intuitive sense: under $H_0\colon \sigma_1^2 = \sigma_2^2$, the ratio $S_1^2/S_2^2$ should be close to 1 on average. Values of $F$ substantially larger or smaller than 1 provide evidence against equal variances.

## Role of the Degrees of Freedom

The two degrees of freedom parameters $d_1$ and $d_2$ control the shape of the F-distribution:

- **Numerator degrees of freedom** $d_1 = n_1 - 1$: reflects the sample size of the group whose variance appears in the numerator. Increasing $d_1$ makes the distribution more concentrated.
- **Denominator degrees of freedom** $d_2 = n_2 - 1$: reflects the sample size of the group in the denominator. Increasing $d_2$ reduces the variance and pulls the mean closer to 1.

When both $d_1$ and $d_2$ are large, the F-distribution approaches a normal distribution centered near 1.

## Reciprocal Property

If $F \sim F_{d_1, d_2}$, then

$$
\frac{1}{F} \sim F_{d_2, d_1}
$$

This property is useful for computing lower-tail critical values. Instead of looking up $F_{\alpha, d_1, d_2}$ (which is not always tabulated), one can use

$$
F_{\alpha,\, d_1,\, d_2} = \frac{1}{F_{1-\alpha,\, d_2,\, d_1}}
$$

## Relationship to Other Distributions

The F-distribution is connected to several other distributions:

- **Chi-square:** If $F \sim F_{d_1, d_2}$, then $d_1 F / (d_1 F + d_2) \sim \text{Beta}(d_1/2, d_2/2)$.
- **$t$-distribution:** If $T \sim t_\nu$, then $T^2 \sim F_{1, \nu}$. The square of a $t$-test statistic is an F-statistic with 1 numerator degree of freedom.
- **Large-sample limit:** As $d_2 \to \infty$, $d_1 F \to \chi^2_{d_1}$. This connects the F-distribution back to the chi-square.

## Example

Two independent samples of sizes $n_1 = 16$ and $n_2 = 21$ are drawn from normal populations. The sample variances are $S_1^2 = 45$ and $S_2^2 = 28$. Under $H_0\colon \sigma_1^2 = \sigma_2^2$:

$$
F = \frac{45}{28} = 1.607
$$

This statistic follows $F_{15, 20}$ under the null. The mean of $F_{15,20}$ is $20/18 \approx 1.111$. An observed value of 1.607 is above the mean but must be compared against the critical value $F_{0.025,\, 15,\, 20} = 2.573$ (two-tailed at $\alpha = 0.05$). Since $1.607 < 2.573$, we fail to reject $H_0$.

## Python Implementation

```python
from scipy import stats

# Degrees of freedom
d1, d2 = 15, 20

# F-distribution properties
print(f"Mean: {stats.f.mean(d1, d2):.4f}")
print(f"Variance: {stats.f.var(d1, d2):.4f}")

# Critical values for two-tailed test at alpha = 0.05
alpha = 0.05
f_lower = stats.f.ppf(alpha / 2, d1, d2)
f_upper = stats.f.ppf(1 - alpha / 2, d1, d2)
print(f"Critical values: [{f_lower:.3f}, {f_upper:.3f}]")

# Test statistic and p-value
f_stat = 45 / 28
p_value = 2 * min(stats.f.cdf(f_stat, d1, d2), stats.f.sf(f_stat, d1, d2))
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.4f}")
```
