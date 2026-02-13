# Chi-Square Test for Variance

The Chi-Square test for variance determines if the variance of a population differs from a specified value. It is a one-sample test, typically applied when the population variance is known or hypothesized. This test is highly sensitive to the assumption that the data follow a normal distribution, and any significant deviation from normality can lead to unreliable results.

## Hypotheses

The hypotheses for the Chi-Square test for variance are formulated as follows:

**Null Hypothesis ($H_0$):** The population variance $\sigma^2$ is equal to some specified value $\sigma_0^2$:

$$
H_0: \sigma^2 = \sigma_0^2
$$

**Alternative Hypothesis ($H_1$):** The population variance differs from the specified value $\sigma_0^2$. This can be expressed depending on whether we are conducting a two-tailed or one-tailed test:

- **Two-tailed test** (variance is simply different):

$$
H_1: \sigma^2 \neq \sigma_0^2
$$

- **One-tailed test** (variance is greater or less than the specified value):

$$
H_1: \sigma^2 > \sigma_0^2 \quad \text{or} \quad H_1: \sigma^2 < \sigma_0^2
$$

## Assumptions

For the Chi-Square test for variance to be valid, the following assumptions must be met:

1. The data must be drawn from a **normally distributed** population. This assumption is crucial, as the test is not robust to deviations from normality.
2. The sample must consist of **independent observations**.
3. The variance of the population $\sigma^2$ is hypothesized to be equal to a specified value $\sigma_0^2$.

If these assumptions are violated, especially the assumption of normality, the test results can be misleading.

## Test Statistic

The test statistic for the Chi-Square test for variance is based on the sample variance $s^2$ and the hypothesized population variance $\sigma_0^2$:

$$
\frac{(n - 1) s^2}{\sigma_0^2} = \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{\sigma_0}\right)^2 \sim \chi^2_{n-1}
$$

where:

- $n$ is the sample size,
- $s^2$ is the sample variance,
- $\sigma_0^2$ is the hypothesized population variance.

The test statistic $\chi^2$ follows a chi-square distribution with $n-1$ degrees of freedom under the null hypothesis. The degrees of freedom reflect the sample size, with larger samples providing more precise estimates of variance.

## Critical Region and Decision Rule

To determine whether to reject the null hypothesis, we compare the test statistic to critical values from the chi-square distribution table, which depends on the significance level $\alpha$ and the degrees of freedom ($n - 1$).

**For a two-tailed test**, we check both the lower and upper tails of the chi-square distribution:

$$
\chi^2_{\text{lower}} < \chi^2 < \chi^2_{\text{upper}}
$$

If the calculated test statistic falls within this range, we fail to reject the null hypothesis. If it falls outside, we reject $H_0$.

**For a one-tailed test**, we only check one end of the distribution:

- If testing $H_1: \sigma^2 > \sigma_0^2$, compare the test statistic to the upper critical value.
- If testing $H_1: \sigma^2 < \sigma_0^2$, compare the test statistic to the lower critical value.

The critical values are derived from chi-square distribution tables and depend on the desired significance level (commonly $\alpha = 0.05$).

## Example Problem and Solution

**Example:** A factory claims that the variance in the weight of a product is $0.04$ grams$^2$. A sample of 25 products is taken, and the sample variance is found to be $0.05$ grams$^2$. At the 5% significance level, test whether the population variance is different from $0.04$ grams$^2$.

### Step-by-Step Solution

**Step 1 — Formulate Hypotheses:**

- $H_0: \sigma^2 = 0.04$
- $H_1: \sigma^2 \neq 0.04$ (two-tailed test)

**Step 2 — Compute the Test Statistic:**

- Sample size $n = 25$
- Sample variance $s^2 = 0.05$
- Hypothesized variance $\sigma_0^2 = 0.04$

$$
\chi^2 = \frac{(25 - 1) \times 0.05}{0.04} = \frac{24 \times 0.05}{0.04} = 30
$$

**Step 3 — Determine Critical Values:**

For $n - 1 = 24$ degrees of freedom and $\alpha = 0.05$ (two-tailed):

- Lower critical value: $\chi^2_{\text{lower}} = 13.848$
- Upper critical value: $\chi^2_{\text{upper}} = 36.415$

**Step 4 — Decision Rule:**

Since the test statistic ($\chi^2 = 30$) lies between the lower and upper critical values ($13.848 < 30 < 36.415$), we fail to reject the null hypothesis.

**Step 5 — Conclusion:**

There is insufficient evidence to suggest that the population variance differs from $0.04$ grams$^2$ at the 5% significance level.

## Confidence Interval for Variance

In addition to hypothesis testing, we can construct a confidence interval for the population variance using the chi-square distribution. The $100(1 - \alpha)\%$ confidence interval for the population variance $\sigma^2$ is:

$$
\left( \frac{(n - 1) s^2}{\chi^2_{\text{upper}}}, \quad \frac{(n - 1) s^2}{\chi^2_{\text{lower}}} \right)
$$

where $\chi^2_{\text{upper}}$ and $\chi^2_{\text{lower}}$ are the critical values of the chi-square distribution for $n - 1$ degrees of freedom, and $s^2$ is the sample variance.

Using the previous scenario, the 95% confidence interval for the population variance is:

$$
\left( \frac{24 \times 0.05}{36.415}, \quad \frac{24 \times 0.05}{13.848} \right) = (0.0329, \; 0.0867)
$$

Thus, the 95% confidence interval for the population variance is between $0.0329$ and $0.0867$ grams$^2$.
