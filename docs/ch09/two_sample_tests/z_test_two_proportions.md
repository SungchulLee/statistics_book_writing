# Two-Sample Z-Test for the Difference of Proportions

## Comparing Success Rates Across Groups

Many practical questions involve comparing the proportion of "successes" in two populations. Does a new website design lead to a higher conversion rate than the current design? Is the infection rate lower in the vaccinated group than in the placebo group? Does the approval rating differ between two demographic groups? The two-sample z-test for proportions provides a formal framework for answering these questions by testing whether two population proportions $p_1$ and $p_2$ are equal.

## Setup and Notation

We observe two independent random samples of binary (success/failure) outcomes:

- Sample 1: $n_1$ independent trials with $X_1$ successes, giving sample proportion $\hat{p}_1 = X_1 / n_1$
- Sample 2: $n_2$ independent trials with $X_2$ successes, giving sample proportion $\hat{p}_2 = X_2 / n_2$

Here $X_1 \sim \text{Binomial}(n_1, p_1)$ and $X_2 \sim \text{Binomial}(n_2, p_2)$, and the two samples are independent of each other.

## Hypotheses

The null hypothesis states that the two population proportions are equal:

$$
H_0 : p_1 = p_2
$$

The alternative hypothesis takes one of three forms:

| Name | Alternative $H_a$ | Rejects when |
|---|---|---|
| Two-sided | $p_1 \neq p_2$ | $\lvert Z \rvert$ is large |
| Left-tailed | $p_1 < p_2$ | $Z$ is very negative |
| Right-tailed | $p_1 > p_2$ | $Z$ is very positive |

??? note "Testing a nonzero difference"
    The test as presented here applies specifically to $H_0: p_1 - p_2 = 0$. Testing $H_0: p_1 - p_2 = \delta_0$ for $\delta_0 \neq 0$ requires a different standard error formula (without pooling), because the pooled proportion is only meaningful when $p_1 = p_2$ is assumed under the null.

## The Pooled Proportion

Under $H_0: p_1 = p_2$, both samples come from populations with the same success probability. The best estimate of this common proportion pools the data from both samples:

$$
\hat{p} = \frac{X_1 + X_2}{n_1 + n_2}
$$

This pooled proportion $\hat{p}$ combines all the success counts and all the trials into a single estimate. It is used in the denominator of the test statistic to estimate the common standard error under the null hypothesis.

## Derivation of the Test Statistic

Under $H_0$, the difference $\hat{p}_1 - \hat{p}_2$ has expected value 0. Because the two samples are independent, the variance of the difference is:

$$
\text{Var}(\hat{p}_1 - \hat{p}_2) = p(1 - p)\left(\frac{1}{n_1} + \frac{1}{n_2}\right)
$$

where $p = p_1 = p_2$ is the common population proportion under $H_0$. Replacing the unknown $p$ with its pooled estimate $\hat{p}$ and standardizing gives the test statistic:

$$
Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\dfrac{1}{n_1} + \dfrac{1}{n_2}\right)}}
$$

Under $H_0$ and for sufficiently large samples, $Z$ is approximately standard normal: $Z \dot{\sim} N(0, 1)$.

## Rejection Regions and P-values

Let $z_{\alpha}$ denote the upper $\alpha$-quantile of the standard normal distribution and $z_{\text{obs}}$ the observed value of the test statistic.

### Two-sided test ($H_a: p_1 \neq p_2$)

**Rejection region**: reject $H_0$ if $|z_{\text{obs}}| > z_{\alpha/2}$.

**P-value**:

$$
p\text{-value} = 2\bigl[1 - \Phi(|z_{\text{obs}}|)\bigr]
$$

### Left-tailed test ($H_a: p_1 < p_2$)

**Rejection region**: reject $H_0$ if $z_{\text{obs}} < -z_{\alpha}$.

**P-value**:

$$
p\text{-value} = \Phi(z_{\text{obs}})
$$

### Right-tailed test ($H_a: p_1 > p_2$)

**Rejection region**: reject $H_0$ if $z_{\text{obs}} > z_{\alpha}$.

**P-value**:

$$
p\text{-value} = 1 - \Phi(z_{\text{obs}})
$$

## Assumptions

The two-sample z-test for proportions requires the following conditions:

1. **Independent samples**: The two samples are drawn independently from their respective populations.
2. **Independent observations**: Within each sample, the binary outcomes are independent.
3. **Large sample sizes**: The normal approximation to the binomial is adequate. A standard rule of thumb requires all four of the following:

$$
n_1 \hat{p} \geq 5, \quad n_1(1 - \hat{p}) \geq 5, \quad n_2 \hat{p} \geq 5, \quad n_2(1 - \hat{p}) \geq 5
$$

Some textbooks state this condition using the individual sample proportions ($n_1\hat{p}_1 \geq 5$, etc.), but checking with the pooled proportion $\hat{p}$ is more appropriate since $\hat{p}$ is the estimate used under $H_0$.

!!! warning "Small samples or extreme proportions"
    When sample sizes are small or proportions are close to 0 or 1, the normal approximation is poor. In such cases, Fisher's exact test or a permutation test provides a more reliable alternative.

## Example: A/B Test for Conversion Rates

An e-commerce company runs an A/B test to compare conversion rates between the current checkout page (A) and a redesigned version (B). Over one week:

- Page A: $n_1 = 500$ visitors, $X_1 = 45$ conversions, $\hat{p}_1 = 45/500 = 0.090$
- Page B: $n_2 = 480$ visitors, $X_2 = 58$ conversions, $\hat{p}_2 = 58/480 = 0.121$

Test at $\alpha = 0.05$ whether the conversion rates differ.

**Step 1: State the hypotheses.**

$$
H_0: p_1 = p_2 \quad \text{vs} \quad H_a: p_1 \neq p_2
$$

**Step 2: Compute the pooled proportion.**

$$
\hat{p} = \frac{45 + 58}{500 + 480} = \frac{103}{980} \approx 0.1051
$$

**Step 3: Check the sample size condition.** The smallest expected count is $n_2(1 - \hat{p}) = 480 \times 0.8949 \approx 430 \geq 5$. All four conditions are satisfied.

**Step 4: Compute the standard error and test statistic.**

$$
\text{SE} = \sqrt{0.1051 \times 0.8949 \times \left(\frac{1}{500} + \frac{1}{480}\right)} = \sqrt{0.09406 \times 0.004083} \approx \sqrt{0.000384} \approx 0.01960
$$

$$
z_{\text{obs}} = \frac{0.090 - 0.121}{0.01960} = \frac{-0.031}{0.01960} \approx -1.582
$$

**Step 5: Compute the p-value.**

$$
p\text{-value} = 2\bigl[1 - \Phi(1.582)\bigr] = 2(1 - 0.9431) = 2(0.0569) \approx 0.114
$$

**Step 6: Make the decision.** Since $p \approx 0.114 > 0.05 = \alpha$, we fail to reject $H_0$. At the 5% significance level, there is insufficient evidence to conclude that the conversion rates differ between the two page designs.

## Python Implementation

```python
import numpy as np
from scipy import stats

def two_proportion_z_test(x1, n1, x2, n2, alternative="two-sided"):
    """Two-sample z-test for the difference of proportions.

    Parameters
    ----------
    x1, x2 : int
        Number of successes in each sample.
    n1, n2 : int
        Sample sizes.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    z_stat : float
        Test statistic.
    p_value : float
        P-value.
    """
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    p_hat = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
    z_stat = (p1_hat - p2_hat) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return z_stat, p_value


# Example: A/B test
z, p = two_proportion_z_test(45, 500, 58, 480, alternative="two-sided")
print(f"z = {z:.3f}, p-value = {p:.3f}")
```

## Connection to Confidence Intervals

By the [duality between hypothesis tests and confidence intervals](../errors_and_power/duality.md), failing to reject $H_0: p_1 = p_2$ at level $\alpha$ is equivalent to 0 being contained in the $(1 - \alpha)$ confidence interval for $p_1 - p_2$. However, the confidence interval typically uses an **unpooled** standard error:

$$
\text{SE}_{\text{CI}} = \sqrt{\frac{\hat{p}_1(1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2(1 - \hat{p}_2)}{n_2}}
$$

because the confidence interval does not assume $p_1 = p_2$. The pooled standard error is specific to the hypothesis test under $H_0$.

## Related Topics

- [Two-Sample Z-Test for Means](z_test_two_means.md): comparing means when variances are known
- [Two-Sample t-Test](t_test_two_means.md): comparing means when variances are unknown
- [F-Test for Two Variances](f_test_two_variances.md): testing equality of population variances
