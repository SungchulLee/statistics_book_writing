# Dunnett's Test (vs Control)

## Overview

Many experiments include a **control group** alongside several treatment groups -- for example, a placebo group and three drug dosages, or a baseline process and four proposed improvements. In these designs, the relevant comparisons are not all pairwise differences but specifically the $k - 1$ comparisons of each treatment group against the control. Testing only these $k - 1$ comparisons rather than all $\binom{k}{2}$ pairwise differences reduces the multiple-testing burden, and Dunnett's test exploits this structure to achieve higher statistical power than methods like Tukey's HSD or Bonferroni.

Dunnett (1955) developed a procedure that controls the family-wise error rate (FWER) at level $\alpha$ for exactly these many-to-one comparisons. The key insight is that the $k - 1$ test statistics are not independent -- they share the control group mean in their denominators -- and Dunnett's critical values account for this correlation.

## Hypotheses

Let group $0$ denote the control and groups $1, 2, \ldots, k-1$ denote the treatments. Dunnett's test can be formulated in two ways:

**Two-sided (non-directional):** For each treatment $i = 1, \ldots, k-1$:

$$
H_0: \mu_i = \mu_0 \quad \text{vs} \quad H_a: \mu_i \neq \mu_0
$$

**One-sided (directional):** If we expect treatments to increase the response:

$$
H_0: \mu_i \leq \mu_0 \quad \text{vs} \quad H_a: \mu_i > \mu_0
$$

The one-sided version is more powerful when the direction of the expected effect is known in advance.

## Test Statistic

For each treatment group $i$ compared to the control group $0$, the test statistic is:

$$
t_i = \frac{\bar{Y}_{i\cdot} - \bar{Y}_{0\cdot}}{\sqrt{\text{MS}_W \left(\dfrac{1}{n_i} + \dfrac{1}{n_0}\right)}}
$$

where:

- $\bar{Y}_{i\cdot}$ is the sample mean of treatment group $i$
- $\bar{Y}_{0\cdot}$ is the sample mean of the control group
- $\text{MS}_W$ is the within-group mean square from the one-way ANOVA with $N - k$ degrees of freedom
- $n_i$ and $n_0$ are the sample sizes of group $i$ and the control group, respectively

## Correlation Structure and Critical Values

The $k - 1$ test statistics $t_1, t_2, \ldots, t_{k-1}$ are not independent because they all involve $\bar{Y}_{0\cdot}$. Under $H_0$, these statistics follow a joint multivariate $t$-distribution. When all treatment groups have the same sample size $n_i = n$, the pairwise correlation between any two test statistics is:

$$
\rho = \frac{n_0}{n_0 + n}
$$

When the control group and all treatment groups have equal sample sizes ($n_0 = n$), this simplifies to $\rho = 1/2$.

Dunnett's critical values $d_{\alpha, k-1, \nu}$ (where $\nu = N - k$) are tabulated from this multivariate $t$-distribution. They account for the simultaneous testing of $k - 1$ correlated comparisons and are smaller than the Bonferroni-adjusted critical values, which is why Dunnett's test has higher power.

## Decision Rule

**Two-sided test:** Reject $H_0: \mu_i = \mu_0$ if $|t_i| > d_{\alpha/2, k-1, N-k}$.

**One-sided test (upper):** Reject $H_0: \mu_i \leq \mu_0$ if $t_i > d_{\alpha, k-1, N-k}$.

## Worked Example

A pharmaceutical company tests three drug formulations against a placebo. The response variable is symptom reduction score (higher is better). Each group has $n = 6$ subjects ($N = 24$, $k = 4$):

| Group | Sample Mean | Sample Size |
|-------|-------------|-------------|
| Placebo (control) | $\bar{Y}_0 = 4.2$ | $n_0 = 6$ |
| Drug A | $\bar{Y}_1 = 7.8$ | $n_1 = 6$ |
| Drug B | $\bar{Y}_2 = 5.9$ | $n_2 = 6$ |
| Drug C | $\bar{Y}_3 = 8.5$ | $n_3 = 6$ |

The one-way ANOVA yields $\text{MS}_W = 3.2$ with $N - k = 20$ degrees of freedom. The overall F-test is significant, so we proceed with Dunnett's test to determine which drugs outperform the placebo.

The standard error for each comparison is the same (equal sample sizes):

$$
\text{SE} = \sqrt{\text{MS}_W \left(\frac{1}{n_i} + \frac{1}{n_0}\right)} = \sqrt{3.2 \times \frac{2}{6}} = \sqrt{1.067} = 1.033
$$

The test statistics are:

$$
t_1 = \frac{7.8 - 4.2}{1.033} = \frac{3.6}{1.033} = 3.49
$$

$$
t_2 = \frac{5.9 - 4.2}{1.033} = \frac{1.7}{1.033} = 1.65
$$

$$
t_3 = \frac{8.5 - 4.2}{1.033} = \frac{4.3}{1.033} = 4.16
$$

For a two-sided test at $\alpha = 0.05$ with $k - 1 = 3$ comparisons and $\nu = 20$ degrees of freedom, the Dunnett critical value is $d_{0.025, 3, 20} \approx 2.54$.

| Comparison | $t_i$ | $\|t_i\| > 2.54$? | Conclusion |
|------------|-------|-------------------|------------|
| Drug A vs Placebo | 3.49 | Yes | Significant |
| Drug B vs Placebo | 1.65 | No | Not significant |
| Drug C vs Placebo | 4.16 | Yes | Significant |

Drugs A and C produce significantly higher symptom reduction scores than the placebo. Drug B does not differ significantly from the placebo at the $\alpha = 0.05$ level.

## When to Use Dunnett's Test

Dunnett's test is the optimal choice when the experimental design involves comparisons to a single reference group. The following comparison highlights when it is preferred:

| Scenario | Recommended Method |
|----------|-------------------|
| All pairwise comparisons | Tukey's HSD |
| Small number of planned comparisons | Bonferroni |
| All contrasts (exploratory) | Scheffe |
| Each treatment vs one control | **Dunnett's test** |
| Unequal variances | Games-Howell |

Dunnett's test is more powerful than Bonferroni for the same set of $k - 1$ control comparisons because it accounts for the positive correlation among the test statistics rather than treating them as independent. For example, with $k = 5$ groups, Dunnett tests 4 comparisons using a critical value derived from their joint distribution, while Bonferroni would use $\alpha/4$ for each, ignoring the correlation and yielding a slightly larger critical value.

!!! warning "Do not use Dunnett's test for all pairwise comparisons"

    Dunnett's test is designed exclusively for many-to-one comparisons against a control. If you need to compare treatments to each other (e.g., Drug A vs Drug B), use Tukey's HSD or another all-pairwise method. Applying Dunnett's test to pairwise comparisons that do not involve the control group is incorrect because the critical values do not account for those additional comparisons.
