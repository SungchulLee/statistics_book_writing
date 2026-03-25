# Bootstrap Test for Two Means

## Motivation

The two-sample $t$-test compares the means of two independent groups under the assumption that the data are normally distributed (or that the sample sizes are large enough for the CLT to apply). When these conditions fail — for example, with small samples from skewed or heavy-tailed distributions — the bootstrap provides an alternative approach.

The key challenge in the two-sample setting is generating bootstrap samples that reflect the null hypothesis $H_0: \mu_X = \mu_Y$. Two main strategies exist: **pooling** and **centering**.

## The Hypothesis

We observe two independent samples:

- Group X: $x_1, \ldots, x_m$ with sample mean $\bar{x}$ and sample standard deviation $s_x$
- Group Y: $y_1, \ldots, y_n$ with sample mean $\bar{y}$ and sample standard deviation $s_y$

The test is:

$$
H_0: \mu_X = \mu_Y \quad \text{vs} \quad H_1: \mu_X \neq \mu_Y
$$

The observed test statistic (unstudentized) is:

$$
d_{\text{obs}} = \bar{x} - \bar{y}
$$

or in studentized form:

$$
t_{\text{obs}} = \frac{\bar{x} - \bar{y}}{\sqrt{s_x^2/m + s_y^2/n}}
$$

## Method 1: Pooled Bootstrap

Under $H_0$, both groups come from the same distribution. The pooled bootstrap enforces this by combining all observations and resampling from the pooled sample.

**Algorithm:**

1. Compute $t_{\text{obs}}$
2. Pool all observations: $z_1, \ldots, z_{m+n} = x_1, \ldots, x_m, y_1, \ldots, y_n$
3. **For** $b = 1, \ldots, B$:
    - Draw $m$ observations with replacement from the pooled sample → "Group X" bootstrap sample
    - Draw $n$ observations with replacement from the pooled sample → "Group Y" bootstrap sample
    - Compute $t^{*(b)}$ using the same test statistic formula
4. The two-sided $p$-value is:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\left(|t^{*(b)}| \ge |t_{\text{obs}}|\right)
$$

!!! note "When Pooling Is Appropriate"
    The pooled bootstrap assumes that under $H_0$, the two groups have the **same distribution** — not just the same mean. If the groups have equal means but different variances or shapes, pooling can be misleading. In that case, the centering approach (Method 2) is preferred.

## Method 2: Centered Bootstrap

The centered bootstrap enforces only $\mu_X = \mu_Y$ without assuming equal distributions. It shifts each group to have mean zero (or any common value):

$$
\tilde{x}_i = x_i - \bar{x}, \quad \tilde{y}_j = y_j - \bar{y}
$$

**Algorithm:**

1. Compute $t_{\text{obs}}$
2. Center each group: $\tilde{x}_i = x_i - \bar{x}$, $\tilde{y}_j = y_j - \bar{y}$
3. **For** $b = 1, \ldots, B$:
    - Draw $m$ observations with replacement from $\{\tilde{x}_1, \ldots, \tilde{x}_m\}$
    - Draw $n$ observations with replacement from $\{\tilde{y}_1, \ldots, \tilde{y}_n\}$
    - Compute $t^{*(b)}$ using the same test statistic formula
4. Compute the $p$-value as above

The centered approach allows the two groups to have different variances and different shapes, imposing only the null hypothesis that their means are equal (both zero after centering).

!!! tip "Choosing Between Pooled and Centered"
    Use the **pooled** approach when you believe both groups have similar distributions under $H_0$ (analogous to the pooled $t$-test). Use the **centered** approach when the groups may have different variances or shapes (analogous to the Welch $t$-test). When in doubt, use the centered approach as it is more robust.

## Studentized vs Unstudentized

As in the one-sample case, the studentized version generally performs better:

| Version | Test statistic | Properties |
|---|---|---|
| Unstudentized | $d^{*(b)} = \bar{x}^{*(b)} - \bar{y}^{*(b)}$ | Simpler, but sensitive to variance differences |
| Studentized | $t^{*(b)} = \frac{\bar{x}^{*(b)} - \bar{y}^{*(b)}}{\sqrt{s_x^{2*(b)}/m + s_y^{2*(b)}/n}}$ | More robust, better Type I error control |

The studentized version adapts to the variability in each bootstrap sample, providing more accurate $p$-values especially when the two groups have different variances.

## Connection to Permutation Tests

The pooled bootstrap is closely related to the **permutation test** for two means. Both pool the data under $H_0$, but they differ in how they generate null samples:

- **Permutation test**: shuffles the group labels without replacement (preserves the original observations exactly)
- **Pooled bootstrap**: resamples with replacement (creates new combinations)

The permutation test conditions on the observed data and provides exact $p$-values (up to the number of permutations). The bootstrap allows for repeated observations and approximates the sampling distribution more broadly. For testing $H_0: \mu_X = \mu_Y$, both approaches are valid; the permutation test is often preferred when the distributions are assumed identical under $H_0$.

## Example

A clinical trial compares a treatment group ($m = 18$, $\bar{x} = 5.8$, $s_x = 3.2$) with a control group ($n = 22$, $\bar{y} = 4.1$, $s_y = 2.5$).

**Observed test statistic** (Welch-type):

$$
t_{\text{obs}} = \frac{5.8 - 4.1}{\sqrt{3.2^2/18 + 2.5^2/22}} = \frac{1.7}{\sqrt{0.569 + 0.284}} = \frac{1.7}{0.924} = 1.84
$$

**Centered bootstrap** (since variances differ):

1. Center: $\tilde{x}_i = x_i - 5.8$, $\tilde{y}_j = y_j - 4.1$
2. Generate $B = 10{,}000$ bootstrap replicates of $t^*$
3. Suppose 712 out of $10{,}000$ satisfy $|t^{*(b)}| \ge 1.84$

The bootstrap $p$-value is $712/10{,}000 = 0.071$.

For comparison, the Welch $t$-test gives $p = 0.074$ ($df \approx 30$). The close agreement suggests the normal approximation is adequate here, but the bootstrap provides reassurance without relying on it.

!!! warning "Equal Variance Assumption"
    If you use the pooled bootstrap but the true variances differ, the Type I error rate can be inflated. The centered bootstrap with a studentized test statistic is more robust to heteroscedasticity and should be the default choice when variance equality is uncertain.

## Summary

The bootstrap test for two means generates the null distribution by either pooling the data (assuming identical distributions under $H_0$) or centering each group separately (allowing different variances). The studentized version is preferred for its robustness to heteroscedasticity. The centered bootstrap parallels the Welch $t$-test in spirit, while the pooled bootstrap parallels the pooled $t$-test. For settings where the classical $t$-test assumptions are questionable, the bootstrap provides a reliable nonparametric alternative.
