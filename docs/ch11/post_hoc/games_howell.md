# Games-Howell Test (Unequal Variances)

## Overview

Post-hoc methods such as Tukey's HSD and Bonferroni assume that the population variances are equal across all groups -- the same homoscedasticity assumption that underlies the standard ANOVA F-test. When this assumption is violated, these methods can produce misleading results: inflated Type I error rates when smaller groups have larger variances, or reduced power in the reverse case. The Games-Howell procedure (Games and Howell, 1976) addresses this problem by using separate variance estimates for each pairwise comparison and adjusting the degrees of freedom via the Welch-Satterthwaite approximation. It does not assume equal variances or equal sample sizes, making it the default post-hoc choice after Welch's ANOVA.

## Hypotheses

For each pair of groups $i$ and $j$ (where $1 \leq i < j \leq k$), the Games-Howell test evaluates:

$$
H_0: \mu_i = \mu_j \quad \text{vs} \quad H_a: \mu_i \neq \mu_j
$$

Unlike Tukey's HSD, the test does not use a pooled variance estimate. Instead, it constructs a separate standard error and degrees-of-freedom estimate for each pair.

## Test Statistic

For comparing groups $i$ and $j$, the test statistic is:

$$
t_{ij} = \frac{\bar{Y}_{i\cdot} - \bar{Y}_{j\cdot}}{\sqrt{\dfrac{s_i^2}{n_i} + \dfrac{s_j^2}{n_j}}}
$$

where:

- $\bar{Y}_{i\cdot}$ and $\bar{Y}_{j\cdot}$ are the sample means of groups $i$ and $j$
- $s_i^2$ and $s_j^2$ are the sample variances of groups $i$ and $j$
- $n_i$ and $n_j$ are the sample sizes

This is identical to the two-sample Welch $t$-statistic. The Games-Howell test applies this statistic to all $\binom{k}{2}$ pairs while controlling the family-wise error rate.

## Welch-Satterthwaite Degrees of Freedom

Because the variance estimates differ across pairs, each comparison has its own degrees of freedom, computed via the Welch-Satterthwaite approximation:

$$
\nu_{ij} = \frac{\left(\dfrac{s_i^2}{n_i} + \dfrac{s_j^2}{n_j}\right)^2}{\dfrac{\left(s_i^2 / n_i\right)^2}{n_i - 1} + \dfrac{\left(s_j^2 / n_j\right)^2}{n_j - 1}}
$$

The result is typically not an integer and is used directly (or rounded down) when looking up critical values.

## Decision Rule

The Games-Howell test compares $|t_{ij}|$ against the critical value from the Studentized range distribution:

Reject $H_0: \mu_i = \mu_j$ if

$$
|t_{ij}| > \frac{q_{\alpha,\, k,\, \nu_{ij}}}{\sqrt{2}}
$$

where $q_{\alpha, k, \nu_{ij}}$ is the upper $\alpha$ critical value of the Studentized range distribution with $k$ groups and $\nu_{ij}$ degrees of freedom. The division by $\sqrt{2}$ converts from the range-based $q$-statistic scale to the $t$-statistic scale.

!!! note "Why the Studentized range distribution?"

    The Studentized range distribution controls the FWER for all pairwise comparisons simultaneously, just as in Tukey's HSD. The difference is that Tukey uses a common degrees-of-freedom parameter (from the pooled $\text{MS}_W$), while Games-Howell uses pair-specific degrees of freedom $\nu_{ij}$, allowing each comparison to reflect its own variance structure.

## Worked Example

A marketing team tests four advertising strategies. Due to budget constraints, the sample sizes and variability differ across groups:

| Strategy | $n_i$ | $\bar{Y}_{i\cdot}$ | $s_i^2$ |
|----------|-------|---------------------|---------|
| A (baseline) | 15 | 12.0 | 4.0 |
| B (social media) | 10 | 16.5 | 12.0 |
| C (email) | 20 | 13.2 | 3.5 |
| D (influencer) | 8 | 18.0 | 15.0 |

Levene's test rejects homoscedasticity ($p = 0.003$), so the standard ANOVA F-test is inappropriate. Welch's ANOVA is significant ($F_W = 6.84$, $p = 0.002$), confirming that at least one strategy differs. We apply the Games-Howell test to identify which pairs differ.

**Comparison: Strategy B vs Strategy A**

Test statistic:

$$
t_{BA} = \frac{16.5 - 12.0}{\sqrt{12.0/10 + 4.0/15}} = \frac{4.5}{\sqrt{1.200 + 0.267}} = \frac{4.5}{\sqrt{1.467}} = \frac{4.5}{1.211} = 3.72
$$

Degrees of freedom:

$$
\nu_{BA} = \frac{(1.200 + 0.267)^2}{\dfrac{(1.200)^2}{9} + \dfrac{(0.267)^2}{14}} = \frac{(1.467)^2}{\dfrac{1.440}{9} + \dfrac{0.071}{14}} = \frac{2.152}{0.160 + 0.005} = \frac{2.152}{0.165} = 13.04
$$

With $k = 4$ groups and $\nu_{BA} \approx 13$ degrees of freedom, the critical value is $q_{0.05, 4, 13} / \sqrt{2} \approx 3.97 / 1.414 \approx 2.81$.

Since $|t_{BA}| = 3.72 > 2.81$, the difference between strategies B and A is **significant**.

**Comparison: Strategy C vs Strategy A**

$$
t_{CA} = \frac{13.2 - 12.0}{\sqrt{3.5/20 + 4.0/15}} = \frac{1.2}{\sqrt{0.175 + 0.267}} = \frac{1.2}{\sqrt{0.442}} = \frac{1.2}{0.665} = 1.80
$$

$$
\nu_{CA} = \frac{(0.442)^2}{\dfrac{(0.175)^2}{19} + \dfrac{(0.267)^2}{14}} = \frac{0.195}{0.00161 + 0.00509} = \frac{0.195}{0.00670} = 29.1
$$

With $\nu_{CA} \approx 29$, the critical value is $q_{0.05, 4, 29} / \sqrt{2} \approx 3.85 / 1.414 \approx 2.72$.

Since $|t_{CA}| = 1.80 < 2.72$, the difference between strategies C and A is **not significant**.

The remaining four comparisons follow the same procedure. Each pair uses its own standard error and degrees of freedom, allowing the test to handle the heteroscedastic data appropriately.

## When to Use Games-Howell

The Games-Howell test is the appropriate post-hoc procedure when:

- **Levene's test** rejects the null hypothesis of equal variances
- **Sample sizes** differ substantially across groups
- The analysis follows **Welch's ANOVA** (which also does not assume equal variances)

| Condition | Recommended Post-Hoc Method |
|-----------|---------------------------|
| Equal variances, all pairwise comparisons | Tukey's HSD |
| Equal variances, planned comparisons | Bonferroni |
| Equal variances, comparisons to control | Dunnett's test |
| **Unequal variances, any pairwise comparisons** | **Games-Howell** |

!!! warning "Games-Howell with small sample sizes"

    The Games-Howell test relies on the Welch-Satterthwaite approximation, which performs well when group sample sizes are moderate ($n_i \geq 6$). With very small groups ($n_i < 6$), the variance estimates are unstable and the $\nu_{ij}$ approximation becomes unreliable. In such cases, consider non-parametric alternatives like the Kruskal-Wallis test followed by Dunn's test.
