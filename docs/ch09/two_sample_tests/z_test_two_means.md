# Two-Sample Z-Test for the Difference of Means

## When Population Variances Are Known

Most two-sample comparisons of means use the [t-test](t_test_two_means.md) because population variances $\sigma_1^2$ and $\sigma_2^2$ are rarely known. However, when these variances **are** known -- for instance, from extensive historical data or from the physical characteristics of a measurement instrument -- the two-sample z-test provides an exact test based on the standard normal distribution without the need for degrees-of-freedom adjustments. Even when variances are unknown, the z-test serves as the conceptual foundation for the two-sample t-test and is a useful pedagogical starting point.

## Setup and Notation

We observe two independent random samples:

- Sample 1: $X_1, X_2, \ldots, X_{n_1}$ drawn from a population with mean $\mu_1$ and **known** variance $\sigma_1^2$
- Sample 2: $Y_1, Y_2, \ldots, Y_{n_2}$ drawn from a population with mean $\mu_2$ and **known** variance $\sigma_2^2$

The sample means are $\bar{X} = \frac{1}{n_1}\sum_{i=1}^{n_1} X_i$ and $\bar{Y} = \frac{1}{n_2}\sum_{j=1}^{n_2} Y_j$. We wish to test whether the difference $\mu_1 - \mu_2$ equals a hypothesized value $\delta_0$ (often $\delta_0 = 0$).

## Hypotheses

The null hypothesis is:

$$
H_0 : \mu_1 - \mu_2 = \delta_0
$$

The alternative hypothesis takes one of three forms depending on the research question:

| Name | Alternative $H_a$ | Rejects when |
|---|---|---|
| Two-sided | $\mu_1 - \mu_2 \neq \delta_0$ | $\lvert Z \rvert$ is large |
| Left-tailed | $\mu_1 - \mu_2 < \delta_0$ | $Z$ is very negative |
| Right-tailed | $\mu_1 - \mu_2 > \delta_0$ | $Z$ is very positive |

In most applications, $\delta_0 = 0$, so the test asks whether the two population means are equal.

## Derivation of the Test Statistic

Under the null hypothesis, the difference in sample means $\bar{X} - \bar{Y}$ estimates $\delta_0$. Because the two samples are independent, the variance of $\bar{X} - \bar{Y}$ is the sum of the individual variances:

$$
\text{Var}(\bar{X} - \bar{Y}) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
$$

If both populations are normal, or if $n_1$ and $n_2$ are large enough for the central limit theorem to apply, then $\bar{X} - \bar{Y}$ is approximately normal. Standardizing under $H_0$ gives the test statistic:

$$
Z = \frac{(\bar{X} - \bar{Y}) - \delta_0}{\sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}}
$$

Under $H_0$, this statistic follows the **standard normal distribution**: $Z \sim N(0, 1)$.

??? note "Exact vs approximate normality"
    When both populations are exactly normal, $Z$ follows $N(0, 1)$ exactly for any sample sizes. When the populations are not normal, the central limit theorem guarantees that $Z$ is approximately $N(0, 1)$ for sufficiently large $n_1$ and $n_2$. A common guideline is $n_1 \geq 30$ and $n_2 \geq 30$, although this threshold depends on how far the population distributions deviate from normality.

## Rejection Regions and P-values

Let $z_{\alpha}$ denote the upper $\alpha$-quantile of the standard normal distribution, and let $z_{\text{obs}}$ denote the observed value of the test statistic.

### Two-sided test (Hₐ: μ₁ - μ₂ ≠ δ₀)

**Rejection region**: reject $H_0$ if $|z_{\text{obs}}| > z_{\alpha/2}$.

**P-value**:

$$
p = 2\,P(Z > |z_{\text{obs}}|) = 2\bigl[1 - \mathcal{N}(|z_{\text{obs}}|)\bigr]
$$

where $\mathcal{N}$ is the standard normal CDF.

### Left-tailed test (Hₐ: μ₁ - μ₂ < δ₀)

**Rejection region**: reject $H_0$ if $z_{\text{obs}} < -z_{\alpha}$.

**P-value**:

$$
p = P(Z < z_{\text{obs}}) = \mathcal{N}(z_{\text{obs}})
$$

### Right-tailed test (Hₐ: μ₁ - μ₂ > δ₀)

**Rejection region**: reject $H_0$ if $z_{\text{obs}} > z_{\alpha}$.

**P-value**:

$$
p = P(Z > z_{\text{obs}}) = 1 - \mathcal{N}(z_{\text{obs}})
$$

## Assumptions

The two-sample z-test requires four conditions:

1. **Known variances**: The population variances $\sigma_1^2$ and $\sigma_2^2$ are known, not estimated from the data. If they must be estimated, use the [two-sample t-test](t_test_two_means.md) instead.
2. **Independence between samples**: The two samples are drawn independently. No observation in one sample influences or is paired with an observation in the other.
3. **Independence within samples**: Observations within each sample are independent (e.g., obtained via simple random sampling).
4. **Normality or large samples**: Each population is normally distributed, or the sample sizes are large enough ($n_1 \geq 30$, $n_2 \geq 30$) for the central limit theorem to provide an adequate normal approximation.

!!! warning "Known variance is rarely realistic"
    In practice, population variances are almost never known with certainty. The two-sample z-test is therefore more of a theoretical building block than a tool used in day-to-day analysis. The [two-sample t-test](t_test_two_means.md) is the standard procedure for comparing means when variances are unknown.

## Example: Comparing Standardized Test Scores

Two school districts adopt different mathematics curricula. From historical testing data, the population standard deviations of scores are known to be $\sigma_1 = 12$ (District A) and $\sigma_2 = 15$ (District B). A random sample of $n_1 = 50$ students from District A has a mean score of $\bar{x} = 78$, and a random sample of $n_2 = 60$ students from District B has a mean score of $\bar{y} = 74$. Test at $\alpha = 0.05$ whether the two districts differ in mean score.

**Step 1: State the hypotheses.**

$$
H_0: \mu_1 - \mu_2 = 0 \quad \text{vs} \quad H_a: \mu_1 - \mu_2 \neq 0
$$

**Step 2: Compute the standard error.**

$$
\text{SE} = \sqrt{\frac{12^2}{50} + \frac{15^2}{60}} = \sqrt{\frac{144}{50} + \frac{225}{60}} = \sqrt{2.88 + 3.75} = \sqrt{6.63} \approx 2.575
$$

**Step 3: Compute the test statistic.**

$$
z_{\text{obs}} = \frac{(78 - 74) - 0}{2.575} = \frac{4}{2.575} \approx 1.553
$$

**Step 4: Compute the p-value.**

$$
p = 2\bigl[1 - \mathcal{N}(1.553)\bigr] = 2(1 - 0.9398) = 2(0.0602) \approx 0.120
$$

**Step 5: Make the decision.** Since $p \approx 0.120 > 0.05 = \alpha$, we fail to reject $H_0$. At the 5% significance level, there is insufficient evidence to conclude that the two districts differ in mean mathematics score.

## Python Implementation

```python
import numpy as np
from scipy import stats

def two_sample_z_test(x_bar, y_bar, sigma1, sigma2, n1, n2,
                      delta0=0, alternative="two-sided"):
    """Two-sample z-test for the difference of means.

    Parameters
    ----------
    x_bar, y_bar : float
        Sample means.
    sigma1, sigma2 : float
        Known population standard deviations.
    n1, n2 : int
        Sample sizes.
    delta0 : float
        Hypothesized difference (default 0).
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    z_stat : float
        Test statistic.
    p_value : float
        P-value.
    """
    se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    z_stat = ((x_bar - y_bar) - delta0) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return z_stat, p_value


# Example: District A vs District B
z, p = two_sample_z_test(78, 74, 12, 15, 50, 60, alternative="two-sided")
print(f"z = {z:.3f}, p-value = {p:.3f}")
```

## Connection to the Two-Sample t-Test

The two-sample z-test and the [two-sample t-test](t_test_two_means.md) have the same structure. The only difference is in the denominator: the z-test uses the known variances $\sigma_1^2$ and $\sigma_2^2$, whereas the t-test replaces them with the sample variances $S_1^2$ and $S_2^2$. This substitution introduces additional uncertainty, which is why the t-test uses the $t$-distribution (with heavier tails) rather than the standard normal. As sample sizes grow, the $t$-distribution approaches the standard normal, and the two tests become equivalent.

## Related Topics

- [Two-Sample t-Test](t_test_two_means.md): the standard test when population variances are unknown
- [Two-Sample Z-Test for Proportions](z_test_two_proportions.md): comparing two population proportions
- [F-Test for Two Variances](f_test_two_variances.md): testing equality of population variances

## Exercises

**Exercise 1.**
Two independent samples: $\bar{x}_1 = 82$, $\sigma_1 = 10$, $n_1 = 50$ and $\bar{x}_2 = 78$, $\sigma_2 = 12$, $n_2 = 60$. Test $H_0: \mu_1 - \mu_2 = 0$ at $\alpha = 0.05$ using a two-sided z-test.

??? success "Solution to Exercise 1"
    The test statistic is:

    $$
    Z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} = \frac{82 - 78}{\sqrt{100/50 + 144/60}} = \frac{4}{\sqrt{2 + 2.4}} = \frac{4}{\sqrt{4.4}} = \frac{4}{2.098} \approx 1.907
    $$

    The critical value for a two-sided test is $z_{0.025} = 1.96$. Since $|Z| = 1.907 < 1.96$, we fail to reject $H_0$.

    The p-value is $2P(Z > 1.907) = 2(0.0283) = 0.0566$. At $\alpha = 0.05$, the evidence is suggestive but not statistically significant.

---

**Exercise 2.**
Construct a 95% confidence interval for $\mu_1 - \mu_2$ using the data from Exercise 1. Does the interval contain 0?

??? success "Solution to Exercise 2"
    The 95% CI for $\mu_1 - \mu_2$ is:

    $$
    (\bar{x}_1 - \bar{x}_2) \pm z_{0.025}\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}} = 4 \pm 1.96 \times 2.098 = 4 \pm 4.112
    $$

    $$
    = (-0.112, 8.112)
    $$

    The interval contains 0, consistent with the failure to reject $H_0$ in Exercise 1. Note: the confidence interval provides more information than the hypothesis test, showing that the true difference could range from a small negative value to about 8 units.

---

**Exercise 3.**
Explain when the two-sample z-test is appropriate versus the two-sample t-test. What assumption about the population variances is crucial?

??? success "Solution to Exercise 3"
    The **two-sample z-test** is appropriate when the population variances $\sigma_1^2$ and $\sigma_2^2$ are **known**. In this case, the test statistic $Z = (\bar{X}_1 - \bar{X}_2)/\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}$ has an exact standard normal distribution under $H_0$ (assuming normality or large samples).

    The **two-sample t-test** is used when the population variances are **unknown** and must be estimated from the data. The variances are replaced by sample variances $s_1^2, s_2^2$, and the test statistic follows an approximate t-distribution (Welch's approximation for unequal variances) or an exact t-distribution (pooled variance, when $\sigma_1^2 = \sigma_2^2$).

    In practice, population variances are almost never known, so the t-test is far more common. The z-test primarily serves as a pedagogical stepping stone and is applicable when sample sizes are very large (making the distinction between $\sigma$ and $s$ negligible).

---

**Exercise 4.**
Show that the two-sample z-test statistic can be derived from the general principle: "estimate minus hypothesized value, divided by standard error." What is the standard error of $\bar{X}_1 - \bar{X}_2$?

??? success "Solution to Exercise 4"
    The general test statistic structure is:

    $$
    Z = \frac{\text{estimator} - \text{hypothesized value}}{\text{SE(estimator)}} = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{\text{SE}(\bar{X}_1 - \bar{X}_2)}
    $$

    Since $\bar{X}_1$ and $\bar{X}_2$ are independent:

    $$
    \text{Var}(\bar{X}_1 - \bar{X}_2) = \text{Var}(\bar{X}_1) + \text{Var}(\bar{X}_2) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
    $$

    $$
    \text{SE}(\bar{X}_1 - \bar{X}_2) = \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
    $$

    Under $H_0: \mu_1 - \mu_2 = \delta_0$ (typically $\delta_0 = 0$), $Z \sim N(0, 1)$ by the CLT or exact normality of the data. This derivation unifies the z-test for a single mean, the z-test for a proportion, and the two-sample z-test under a single framework.
