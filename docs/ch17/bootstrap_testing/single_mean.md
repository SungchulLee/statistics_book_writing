# Bootstrap Test for a Single Mean

## Motivation

The one-sample $t$-test assumes that the sampling distribution of $\bar{X}$ is approximately normal. When the population is heavily skewed, has heavy tails, or the sample size is small, this assumption may not hold. The **bootstrap test for a single mean** provides an alternative that does not rely on normality: it estimates the null distribution of the test statistic by resampling, allowing valid inference even when the $t$-distribution is a poor approximation.

## The Hypothesis

We test:

$$
H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0
$$

(two-sided; one-sided variants follow by adjusting the $p$-value calculation).

The observed test statistic is:

$$
t_{\text{obs}} = \bar{x} - \mu_0
$$

or, in studentized form:

$$
t_{\text{obs}} = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

## Centering the Bootstrap Distribution

A critical step in bootstrap hypothesis testing is generating bootstrap samples that satisfy the null hypothesis. Under $H_0: \mu = \mu_0$, we need bootstrap samples centered at $\mu_0$, but the original data are centered at $\bar{x}$.

The solution is to **shift the data** before resampling:

$$
\tilde{x}_i = x_i - \bar{x} + \mu_0, \quad i = 1, \ldots, n
$$

The shifted sample $\{\tilde{x}_1, \ldots, \tilde{x}_n\}$ has mean $\mu_0$ while preserving the original spread and shape. Bootstrap samples drawn from $\{\tilde{x}_1, \ldots, \tilde{x}_n\}$ represent "what the data would look like if $H_0$ were true."

!!! note "Why Centering Is Necessary"
    Without centering, the bootstrap resamples from data centered at $\bar{x}$, which approximates the sampling distribution under the true parameter value. For hypothesis testing, we need the null distribution — the distribution of the test statistic assuming $H_0$ is true. Centering at $\mu_0$ ensures the bootstrap generates this null distribution.

## Algorithm: Unstudentized Version

1. Compute $t_{\text{obs}} = \bar{x} - \mu_0$
2. Create the centered data: $\tilde{x}_i = x_i - \bar{x} + \mu_0$ for $i = 1, \ldots, n$
3. **For** $b = 1, \ldots, B$:
    - Draw $\tilde{x}_1^*, \ldots, \tilde{x}_n^*$ with replacement from $\{\tilde{x}_1, \ldots, \tilde{x}_n\}$
    - Compute $t^{*(b)} = \bar{\tilde{x}}^{*(b)} - \mu_0$
4. The two-sided $p$-value is:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(|t^{*(b)}| \ge |t_{\text{obs}}|\right)
$$

## Algorithm: Studentized Version

The studentized version uses the $t$-statistic and generally has better power:

1. Compute $t_{\text{obs}} = (\bar{x} - \mu_0) / (s / \sqrt{n})$
2. Create the centered data: $\tilde{x}_i = x_i - \bar{x} + \mu_0$
3. **For** $b = 1, \ldots, B$:
    - Draw $\tilde{x}_1^*, \ldots, \tilde{x}_n^*$ with replacement from $\{\tilde{x}_1, \ldots, \tilde{x}_n\}$
    - Compute $t^{*(b)} = (\bar{\tilde{x}}^{*(b)} - \mu_0) / (s^{*(b)} / \sqrt{n})$
4. The two-sided $p$-value is:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(|t^{*(b)}| \ge |t_{\text{obs}}|\right)
$$

!!! tip "Studentized vs Unstudentized"
    The studentized version is preferred because it accounts for variability in the standard error across bootstrap samples. This makes the test more robust to heterogeneity and generally produces more accurate $p$-values.

## One-Sided Tests

For $H_1: \mu > \mu_0$, the $p$-value counts bootstrap replicates at least as extreme in the positive direction:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(t^{*(b)} \ge t_{\text{obs}}\right)
$$

For $H_1: \mu < \mu_0$:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(t^{*(b)} \le t_{\text{obs}}\right)
$$

## Example

A sample of $n = 12$ delivery times (in minutes) from a restaurant has $\bar{x} = 34.2$ and $s = 8.7$. The restaurant claims the average delivery time is 30 minutes. We test $H_0: \mu = 30$ vs $H_1: \mu \neq 30$.

**Step 1.** $t_{\text{obs}} = (34.2 - 30) / (8.7/\sqrt{12}) = 1.672$

**Step 2.** Shift the data: $\tilde{x}_i = x_i - 34.2 + 30 = x_i - 4.2$

**Step 3.** Generate $B = 10{,}000$ bootstrap replicates of the studentized statistic from the shifted data.

**Step 4.** Suppose 832 out of $10{,}000$ replicates satisfy $|t^{*(b)}| \ge 1.672$.

The bootstrap $p$-value is $832/10{,}000 = 0.0832$.

For comparison, the classical $t$-test gives $p = 0.123$ (using $t_{11}$ distribution). The difference arises because the bootstrap does not assume normality; if the delivery times are right-skewed, the bootstrap $p$-value may be more reliable.

## Connection to Confidence Intervals

There is a direct duality between the bootstrap test and bootstrap confidence intervals. The bootstrap test rejects $H_0: \mu = \mu_0$ at level $\alpha$ if and only if $\mu_0$ falls outside the corresponding $100(1-\alpha)\%$ bootstrap confidence interval.

This means that instead of running the bootstrap test separately, one can construct a bootstrap confidence interval and check whether $\mu_0$ is contained in it.

!!! warning "Monte Carlo Variability in p-Values"
    Because the bootstrap $p$-value is estimated by simulation, it has Monte Carlo error. With $B = 10{,}000$, a true $p$-value of 0.05 has a Monte Carlo standard error of approximately $\sqrt{0.05 \times 0.95 / 10{,}000} \approx 0.002$. When the $p$-value is close to the significance level, increase $B$ to reduce the chance of a wrong decision.

## Summary

The bootstrap test for a single mean generates the null distribution by resampling from data that have been centered at the hypothesized value $\mu_0$. The $p$-value is the proportion of bootstrap test statistics at least as extreme as the observed statistic. The studentized version is preferred for its better power and accuracy. This approach avoids the normality assumption of the classical $t$-test and is particularly valuable for skewed or heavy-tailed data with moderate sample sizes.
