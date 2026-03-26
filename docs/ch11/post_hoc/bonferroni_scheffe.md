# Bonferroni and Scheffe Methods


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

When a one-way ANOVA rejects the null hypothesis, it tells us that at least one group mean differs from the rest -- but not which group or groups are responsible. A natural next step is to test individual comparisons: pairwise differences such as $\mu_i - \mu_j$, or more complex linear combinations of the group means. The difficulty is that performing many tests simultaneously inflates the probability of at least one false positive. If we test $m$ independent comparisons each at level $\alpha$, the family-wise error rate (FWER) can be as high as $1 - (1 - \alpha)^m$, which grows quickly with $m$.

The Bonferroni and Scheffe methods both control the FWER at level $\alpha$, but they do so in different ways and are suited to different situations. Bonferroni adjusts the per-comparison significance level and works best for a small number of planned comparisons. Scheffe controls the FWER simultaneously over all possible linear contrasts and is the method of choice for data-driven, exploratory comparisons.

## Bonferroni Method

### Intuition

The Bonferroni correction is the simplest multiple-comparison adjustment: divide the significance budget $\alpha$ equally among all comparisons. If we plan $m$ comparisons and test each one at level $\alpha/m$, the Boole--Bonferroni inequality guarantees that the probability of making at least one Type I error is at most $\alpha$, regardless of the correlation structure among the test statistics.

### Formal Procedure

Suppose we plan $m$ comparisons after a one-way ANOVA with $k$ groups, $N$ total observations, and within-group mean square $\text{MS}_W$ with $N - k$ degrees of freedom. For each comparison, the procedure is:

**Step 1.** Define the comparison. For a pairwise comparison $\mu_i - \mu_j$, or more generally for any contrast $L = \sum_{i=1}^{k} c_i \mu_i$ with $\sum c_i = 0$.

**Step 2.** Compute the test statistic. For a pairwise comparison:

$$
t = \frac{\bar{Y}_{i\cdot} - \bar{Y}_{j\cdot}}{\sqrt{\text{MS}_W \left(\dfrac{1}{n_i} + \dfrac{1}{n_j}\right)}}
$$

For a general contrast $L = \sum c_i \mu_i$:

$$
t = \frac{\sum_{i=1}^{k} c_i \bar{Y}_{i\cdot}}{\sqrt{\text{MS}_W \sum_{i=1}^{k} \dfrac{c_i^2}{n_i}}}
$$

**Step 3.** Compare $|t|$ against the critical value $t_{\alpha/(2m),\, N-k}$. Reject $H_0: L = 0$ if $|t| > t_{\alpha/(2m),\, N-k}$.

Equivalently, compute the p-value for each test and reject if $p < \alpha/m$.

!!! tip "When to use Bonferroni"

    Bonferroni is most powerful when the number of planned comparisons $m$ is small. For all pairwise comparisons among $k$ groups, $m = \binom{k}{2}$, and Tukey's HSD is typically more powerful. The Bonferroni correction shines when you have a small set of pre-specified hypotheses -- for instance, testing 3 specific contrasts among 5 groups rather than all 10 pairwise comparisons.

### Worked Example

A one-way ANOVA with $k = 4$ groups (each with $n = 8$ observations, so $N = 32$) yields $\text{MS}_W = 6.0$ with $N - k = 28$ degrees of freedom. The group means are $\bar{Y}_1 = 14.0$, $\bar{Y}_2 = 11.5$, $\bar{Y}_3 = 15.2$, $\bar{Y}_4 = 12.0$. The researcher planned $m = 3$ comparisons before collecting data:

1. $\mu_1 - \mu_2$
2. $\mu_3 - \mu_4$
3. $\mu_1 - \mu_4$

The Bonferroni-adjusted significance level is $\alpha^* = 0.05 / 3 = 0.0167$, giving a two-sided critical value of $t_{0.0083, 28} \approx 2.57$.

For comparison 1:

$$
t = \frac{14.0 - 11.5}{\sqrt{6.0 \times (1/8 + 1/8)}} = \frac{2.5}{\sqrt{1.5}} = \frac{2.5}{1.225} = 2.04
$$

Since $|2.04| < 2.57$, this comparison is not significant after the Bonferroni correction.

For comparison 2:

$$
t = \frac{15.2 - 12.0}{\sqrt{1.5}} = \frac{3.2}{1.225} = 2.61
$$

Since $|2.61| > 2.57$, this comparison is significant. Groups 3 and 4 differ at the Bonferroni-corrected level.

For comparison 3:

$$
t = \frac{14.0 - 12.0}{\sqrt{1.5}} = \frac{2.0}{1.225} = 1.63
$$

Since $|1.63| < 2.57$, this comparison is not significant.

## Scheffe's Method

### Intuition

While Bonferroni controls the FWER for a fixed, pre-specified set of comparisons, Scheffe's method provides a stronger guarantee: it controls the FWER simultaneously over all possible linear contrasts of the group means. This makes it the appropriate choice when comparisons are suggested by the data rather than planned in advance. The trade-off is that Scheffe's critical value is larger (more conservative), so the method has lower power for any individual comparison.

### Linear Contrasts

A **linear contrast** is a linear combination of the population means:

$$
L = \sum_{i=1}^{k} c_i \mu_i \quad \text{where} \quad \sum_{i=1}^{k} c_i = 0
$$

The constraint $\sum c_i = 0$ ensures that $L$ measures a difference rather than a level. Pairwise comparisons are a special case: $\mu_i - \mu_j$ corresponds to $c_i = 1$, $c_j = -1$, and all other coefficients zero. More complex contrasts are also possible, such as comparing one group to the average of two others: $\mu_1 - \frac{1}{2}(\mu_2 + \mu_3)$ uses coefficients $c_1 = 1$, $c_2 = -1/2$, $c_3 = -1/2$.

### Formal Procedure

For a contrast $L = \sum c_i \mu_i$ with estimate $\hat{L} = \sum c_i \bar{Y}_{i\cdot}$:

**Step 1.** Compute the F-statistic for the contrast:

$$
F_L = \frac{\hat{L}^2}{\text{MS}_W \displaystyle\sum_{i=1}^{k} \dfrac{c_i^2}{n_i}}
$$

**Step 2.** Compare $F_L$ against the Scheffe critical value:

$$
F_{\text{crit}}^{S} = (k - 1) \cdot F_{\alpha,\, k-1,\, N-k}
$$

**Step 3.** Reject $H_0: L = 0$ if $F_L > F_{\text{crit}}^{S}$.

Equivalently, one can use a $t$-form of the test: reject if $|t_L| > \sqrt{(k-1) F_{\alpha, k-1, N-k}}$, where $t_L = \hat{L} / \text{SE}(\hat{L})$.

!!! note "Why Scheffe uses (k-1) F rather than F"

    The factor $(k - 1)$ accounts for the fact that the FWER is controlled over all possible contrasts simultaneously. Roy's union-intersection principle shows that the maximum of $F_L$ over all contrasts $L$ equals the overall ANOVA F-statistic, which has critical value $F_{\alpha, k-1, N-k}$. Multiplying by $(k-1)$ converts the per-contrast F-statistic to the same scale, ensuring the family-wise guarantee.

### Worked Example

Using the same data as the Bonferroni example ($k = 4$, $n = 8$, $\text{MS}_W = 6.0$, $N - k = 28$), suppose a researcher notices the data and decides to test the contrast $L = \mu_3 - \frac{1}{3}(\mu_1 + \mu_2 + \mu_4)$, comparing group 3 to the average of the other three groups. The contrast coefficients are $c_1 = -1/3$, $c_2 = -1/3$, $c_3 = 1$, $c_4 = -1/3$.

The contrast estimate is:

$$
\hat{L} = -\frac{1}{3}(14.0) - \frac{1}{3}(11.5) + 1(15.2) - \frac{1}{3}(12.0) = -4.667 - 3.833 + 15.2 - 4.0 = 2.7
$$

The standard error denominator:

$$
\text{MS}_W \sum \frac{c_i^2}{n_i} = 6.0 \times \frac{(1/9 + 1/9 + 1 + 1/9)}{8} = 6.0 \times \frac{1.333}{8} = 1.0
$$

The F-statistic for the contrast:

$$
F_L = \frac{(2.7)^2}{1.0} = 7.29
$$

The Scheffe critical value with $F_{0.05, 3, 28} \approx 2.95$:

$$
F_{\text{crit}}^{S} = (4 - 1) \times 2.95 = 8.85
$$

Since $F_L = 7.29 < 8.85$, the contrast is not significant by Scheffe's method. This illustrates the conservatism of Scheffe's approach: the contrast has a sizable point estimate but does not reach the threshold required for simultaneous control over all possible contrasts.

## Comparison of Bonferroni and Scheffe

The two methods address the same problem -- controlling the FWER -- but are optimal in different situations:

| Feature | Bonferroni | Scheffe |
|---------|-----------|---------|
| **Controls FWER for** | A fixed set of $m$ pre-specified comparisons | All possible linear contrasts simultaneously |
| **Critical value depends on** | Number of comparisons $m$ | Number of groups $k$ (not $m$) |
| **Power** | Higher when $m$ is small | Lower for any fixed set, but applies to unlimited contrasts |
| **Best used when** | Comparisons are planned before data collection | Comparisons are exploratory or data-driven |
| **Conservatism** | Increases with $m$ | Constant regardless of how many contrasts are tested |

!!! warning "Common pitfall"

    Using Bonferroni for data-driven comparisons violates the method's assumptions. If the choice of which comparisons to test was influenced by the data, the actual number of "implicit" comparisons exceeds $m$, and the FWER is no longer controlled at $\alpha$. In such cases, Scheffe's method is the correct choice because its guarantee holds regardless of how contrasts are selected.

As a rule of thumb: if the number of planned comparisons satisfies $m < k - 1$, Bonferroni is typically more powerful than Scheffe. When $m$ approaches or exceeds $\binom{k}{2}$, or when the comparisons are not pre-specified, Scheffe becomes the preferred method.
