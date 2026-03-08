# F-Test for Comparing Two Variances

The F-test for comparing two variances is a statistical test used to determine whether the variances of two independent samples are significantly different. It is based on the ratio of the sample variances and assumes that both samples come from normally distributed populations. The F-test is particularly useful for assessing the homogeneity of variances, which is often a critical assumption in methods such as ANOVA.

## Hypotheses

**Null Hypothesis ($H_0$):** The two population variances are equal:

$$
H_0: \sigma_1^2 = \sigma_2^2
$$

**Alternative Hypothesis ($H_1$):** The two population variances are not equal:

- **Two-tailed test** (variances are simply different):

$$
H_1: \sigma_1^2 \neq \sigma_2^2
$$

- **One-tailed test** (one variance is greater or smaller than the other):

$$
H_1: \sigma_1^2 > \sigma_2^2 \quad \text{or} \quad H_1: \sigma_1^2 < \sigma_2^2
$$

The choice between one-tailed and two-tailed depends on the research question. In many applications, the two-tailed version is used when there is no prior expectation about which variance is larger.

## Assumptions

The F-test relies on several key assumptions:

1. **Normality:** The populations from which the samples are drawn must follow a normal distribution. The F-test is highly sensitive to deviations from normality.
2. **Independence:** The two samples must be independent of each other.
3. **Random Sampling:** Both samples should be random and representative of their respective populations.

If these assumptions are violated, particularly normality, the F-test may lead to incorrect conclusions. In such cases, alternative methods like Levene's test or the Brown–Forsythe test are more appropriate.

## Test Statistic

The F-test statistic is based on the ratio of the two sample variances. Given two independent samples with sample variances $s_1^2$ and $s_2^2$ from populations with variances $\sigma_1^2$ and $\sigma_2^2$, under $H_0$ with $\sigma := \sigma_1 = \sigma_2$:

$$
\frac{s_1^2}{s_2^2} = \frac{\dfrac{(n_1-1)s_1^2/\sigma^2}{n_1-1}}{\dfrac{(n_2-1)s_2^2/\sigma^2}{n_2-1}} = \frac{\dfrac{\chi^2_{n_1-1}}{n_1-1}}{\dfrac{\chi^2_{n_2-1}}{n_2-1}} \sim F_{n_1-1, \, n_2-1}
$$

The sample variances $s_1^2$ and $s_2^2$ are calculated as:

$$
s_1^2 = \frac{1}{n_1 - 1} \sum_{i=1}^{n_1} (X_{1i} - \bar{X}_1)^2, \qquad
s_2^2 = \frac{1}{n_2 - 1} \sum_{i=1}^{n_2} (X_{2i} - \bar{X}_2)^2
$$

where $X_{1i}$ and $X_{2i}$ are individual observations in samples 1 and 2, $\bar{X}_1$ and $\bar{X}_2$ are the sample means, and $n_1$ and $n_2$ are the sample sizes.

The F-distribution is used to evaluate the significance of the ratio, with degrees of freedom $n_1 - 1$ and $n_2 - 1$. The F-statistic always compares the larger sample variance to the smaller one, ensuring that $F \geq 1$.

## Critical Region and Decision Rule

To determine whether the difference between the two sample variances is statistically significant, the calculated F-statistic is compared to critical values from the F-distribution:

- Degrees of freedom for the numerator: $df_1 = n_1 - 1$
- Degrees of freedom for the denominator: $df_2 = n_2 - 1$

**Two-tailed test:**

- Reject $H_0$ if $F < F_{\text{lower}}$ or $F > F_{\text{upper}}$.
- Fail to reject $H_0$ if $F_{\text{lower}} < F < F_{\text{upper}}$.

**One-tailed test:**

- If testing $H_1: \sigma_1^2 > \sigma_2^2$, reject $H_0$ if $F > F_{\text{upper}}$.
- If testing $H_1: \sigma_1^2 < \sigma_2^2$, reject $H_0$ if $F < F_{\text{lower}}$.

## Example Problem and Solution

**Example:** A researcher wants to compare the variability in the test scores of two different student groups. Group 1 has a sample size of 15 and sample variance $s_1^2 = 25$. Group 2 has a sample size of 20 and sample variance $s_2^2 = 16$. Test whether the variances of the two groups are significantly different at the 5% significance level.

### Step-by-Step Solution

**Step 1 — Formulate Hypotheses:**

- $H_0: \sigma_1^2 = \sigma_2^2$
- $H_1: \sigma_1^2 \neq \sigma_2^2$ (two-tailed test)

**Step 2 — Compute the Test Statistic:**

$$
F = \frac{s_1^2}{s_2^2} = \frac{25}{16} = 1.5625
$$

**Step 3 — Determine Critical Values:**

For $df_1 = 14$ and $df_2 = 19$ degrees of freedom at $\alpha = 0.05$ (two-tailed):

- Lower critical value: $F_{\text{lower}} = 0.390$
- Upper critical value: $F_{\text{upper}} = 2.657$

**Step 4 — Decision Rule:**

Since $F = 1.5625$ falls between the critical values ($0.390 < 1.5625 < 2.657$), we fail to reject the null hypothesis.

**Step 5 — Conclusion:**

There is insufficient evidence to conclude that the variances of the two groups are significantly different at the 5% significance level.

## Python Implementation

```python
import numpy as np
from scipy.stats import f

# Example data: Two sample datasets
sample1 = [12, 15, 14, 10, 13, 14, 12, 11]
sample2 = [22, 25, 20, 18, 24, 23, 19, 21]

# Calculate sample variances
var1 = np.var(sample1, ddof=1)
var2 = np.var(sample2, ddof=1)

# Calculate the F-statistic
f_statistic = var1 / var2

# Degrees of freedom
df1 = len(sample1) - 1
df2 = len(sample2) - 1

# One-sided test: H1: sigma1 < sigma2
p_value = f(df1, df2).cdf(f_statistic)

# Display results
print(f"F-statistic: {f_statistic}")
print(f"Degrees of freedom: {df1}, {df2}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in variances.")
```
