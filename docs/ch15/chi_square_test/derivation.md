# Derivation and Distribution Theory

The chi-square test for a population variance rests on a single distributional result: when the data are normally distributed, the scaled sample variance follows a chi-square distribution. This section derives that result from first principles, starting with the connection between standard normal variables and the chi-square family.

## Sum of Squared Standard Normals

Recall from Chapter 5 that if $Z_1, Z_2, \ldots, Z_n$ are independent standard normal random variables, then their sum of squares follows a chi-square distribution with $n$ degrees of freedom:

$$
\sum_{i=1}^{n} Z_i^2 \sim \chi^2_n
$$

This is the defining property of the chi-square distribution. Each $Z_i^2$ contributes one degree of freedom, and independence ensures the degrees of freedom add.

## Standardizing the Observations

Suppose $X_1, X_2, \ldots, X_n$ are independent and identically distributed with $X_i \sim N(\mu, \sigma^2)$. Define the standardized observations

$$
Z_i = \frac{X_i - \mu}{\sigma}
$$

so that each $Z_i \sim N(0, 1)$ independently. Then

$$
\sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 = \sum_{i=1}^{n} Z_i^2 \sim \chi^2_n
$$

This sum has $n$ degrees of freedom because it involves $n$ independent squared standard normals and no estimated parameters.

## Replacing the Population Mean with the Sample Mean

In practice, $\mu$ is unknown and is replaced by the sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$. Consider the decomposition

$$
\sum_{i=1}^{n} (X_i - \mu)^2 = \sum_{i=1}^{n} (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2
$$

Dividing both sides by $\sigma^2$ gives

$$
\sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 = \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 + \left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}\right)^2
$$

The left side is $\chi^2_n$. The last term on the right is the square of a standard normal variable (since $\bar{X} \sim N(\mu, \sigma^2/n)$), so it is $\chi^2_1$. By Cochran's theorem, the two terms on the right are independent, and therefore

$$
\sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

The loss of one degree of freedom reflects the constraint $\sum_{i=1}^{n}(X_i - \bar{X}) = 0$, which removes one free dimension from the $n$ squared terms.

## The Central Result

The sample variance is defined as

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2
$$

Multiplying both sides by $(n-1)/\sigma^2$ yields

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

This is the pivotal quantity for inference about $\sigma^2$. Under $H_0\colon \sigma^2 = \sigma_0^2$, the test statistic becomes

$$
\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2_{n-1}
$$

!!! note "Role of Cochran's Theorem"
    Cochran's theorem guarantees that $\sum(X_i - \bar{X})^2/\sigma^2$ and $(\bar{X} - \mu)^2/(\sigma^2/n)$ are independent chi-square variables whose degrees of freedom sum to $n$. Without this independence, the decomposition would not yield an exact chi-square distribution for the sample variance.

## Properties of the Chi-Square Distribution

The chi-square distribution with $\nu$ degrees of freedom has several properties relevant to variance testing:

- **Mean:** $E[\chi^2_\nu] = \nu$
- **Variance:** $\operatorname{Var}(\chi^2_\nu) = 2\nu$
- **Skewness:** $\sqrt{8/\nu}$, which decreases as $\nu$ increases
- **Mode:** $\nu - 2$ for $\nu \ge 2$

Since the chi-square distribution is right-skewed, confidence intervals for $\sigma^2$ are asymmetric. For large degrees of freedom, the distribution becomes approximately normal by the central limit theorem.

## Example

Suppose a random sample of $n = 21$ observations is drawn from a normal population, and the sample variance is $S^2 = 18.5$. To test $H_0\colon \sigma^2 = 15$ against $H_1\colon \sigma^2 \neq 15$ at the $\alpha = 0.05$ significance level:

**Step 1.** Compute the test statistic:

$$
\chi^2 = \frac{(21 - 1)(18.5)}{15} = \frac{20 \times 18.5}{15} = 24.667
$$

**Step 2.** Determine the critical values for $\chi^2_{20}$ at $\alpha = 0.05$ (two-tailed):

- Lower critical value: $\chi^2_{0.975, 20} = 9.591$
- Upper critical value: $\chi^2_{0.025, 20} = 34.170$

**Step 3.** Since $9.591 < 24.667 < 34.170$, the test statistic falls within the acceptance region. We fail to reject $H_0$.

The data do not provide sufficient evidence to conclude that the population variance differs from 15 at the 5% significance level.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Sample data
n = 21
s_squared = 18.5
sigma_0_squared = 15
alpha = 0.05

# Test statistic
chi2_stat = (n - 1) * s_squared / sigma_0_squared

# Critical values (two-tailed)
df = n - 1
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

# p-value (two-tailed)
p_value = 2 * min(stats.chi2.cdf(chi2_stat, df), stats.chi2.sf(chi2_stat, df))

print(f"Test statistic: {chi2_stat:.3f}")
print(f"Critical values: [{chi2_lower:.3f}, {chi2_upper:.3f}]")
print(f"P-value: {p_value:.4f}")

if p_value < alpha:
    print("Reject H0: variance differs from the hypothesized value.")
else:
    print("Fail to reject H0: insufficient evidence of a difference.")
```
