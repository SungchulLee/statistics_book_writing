# Permutation Test for Paired Data

## Motivation

The paired $t$-test compares two related measurements (before vs after, treatment vs control on the same subject) by analyzing the differences $d_i = x_i - y_i$ and testing whether their mean is zero. This test assumes the differences are normally distributed, which may not hold for small samples from skewed or heavy-tailed populations.

The **permutation test for paired data** provides an exact, distribution-free alternative. The key insight is that under $H_0$, the sign of each difference is equally likely to be positive or negative. The test exploits this symmetry by randomly flipping the signs of the observed differences.

## The Hypothesis

Given $n$ paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, compute the differences:

$$
d_i = x_i - y_i, \quad i = 1, \ldots, n
$$

The test is:

$$
H_0: \text{The distribution of } d_i \text{ is symmetric about } 0 \quad \text{vs} \quad H_1: \text{The center of } d_i \text{ is not } 0
$$

Under $H_0$, each $d_i$ is equally likely to be positive or negative. The observed test statistic is typically the mean of the differences:

$$
\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i
$$

or the sum $T = \sum_{i=1}^n d_i$ (equivalent for testing purposes).

!!! note "Symmetry Assumption"
    The paired permutation test assumes that the distribution of each $d_i$ is symmetric about zero under $H_0$. This is weaker than the normality assumption of the paired $t$-test but stronger than simply assuming $E[d_i] = 0$. If the differences are skewed even under $H_0$ (which is unusual in practice), the sign-flip approach may not be valid.

## Algorithm: Sign-Flip Permutation

1. Compute the observed differences $d_1, \ldots, d_n$ and the test statistic $\bar{d}_{\text{obs}} = \frac{1}{n}\sum_{i=1}^n d_i$
2. **For** $b = 1, 2, \ldots, B$:
    - For each $i = 1, \ldots, n$: independently set $s_i = +1$ or $s_i = -1$ with equal probability
    - Compute $d_i^{*(b)} = s_i \cdot d_i$
    - Compute $\bar{d}^{*(b)} = \frac{1}{n}\sum_{i=1}^n d_i^{*(b)}$
3. The two-sided $p$-value is:

$$
p = \frac{\#\{b : |\bar{d}^{*(b)}| \ge |\bar{d}_{\text{obs}}|\} + 1}{B + 1}
$$

Each sign-flip assignment represents one equally likely arrangement of the data under $H_0$. There are $2^n$ possible sign assignments in total.

## The Exact Distribution

For small $n$, all $2^n$ sign-flip configurations can be enumerated:

| $n$ | Number of configurations ($2^n$) | Feasibility |
|---|---|---|
| 5 | 32 | Trivial |
| 10 | 1,024 | Easy |
| 15 | 32,768 | Feasible |
| 20 | 1,048,576 | Feasible with fast computation |
| 25 | 33,554,432 | Borderline |
| 30 | $> 10^9$ | Random sampling required |

For $n \le 20$, exact enumeration is practical. For larger $n$, random sampling of $B = 10{,}000$ or more sign-flip configurations provides an excellent approximation.

## Why Sign-Flipping Works

Under $H_0$, the distribution of $d_i$ is symmetric about zero. This means $d_i$ and $-d_i$ have the same distribution. Therefore, replacing $d_i$ with $-d_i$ (a "sign flip") produces a dataset that is equally likely under $H_0$.

The collection of all $2^n$ sign-flip configurations generates the **exact conditional distribution** of the test statistic under $H_0$, given the absolute values $|d_1|, \ldots, |d_n|$. The $p$-value from this distribution is exact — it controls Type I error at exactly level $\alpha$ for any sample size.

## Alternative Test Statistics

The sign-flip framework allows any test statistic computed from the differences. Common choices include:

**Mean of differences** ($\bar{d}$): most powerful when differences are symmetric and light-tailed.

**Sum of positive ranks** (Wilcoxon signed-rank statistic):

$$
W^+ = \sum_{i=1}^n \text{rank}(|d_i|) \cdot \mathbf{1}(d_i > 0)
$$

This is more robust to outliers because it uses ranks rather than raw values.

**Trimmed mean of differences**: excludes the most extreme differences before averaging, providing a compromise between the mean and the median.

!!! tip "Which Test Statistic to Use"
    Use $\bar{d}$ when the differences are expected to be roughly symmetric with no extreme outliers. Use the Wilcoxon signed-rank statistic when outliers are a concern. The permutation mechanism (sign flipping) is the same regardless of the test statistic chosen.

## Example

A study measures blood pressure before and after a new meditation program for $n = 12$ participants. The observed differences (after $-$ before) are:

$$
d = (-8, -3, 2, -12, -5, -1, 4, -7, -6, -2, -9, -4)
$$

The observed mean difference is $\bar{d}_{\text{obs}} = -4.25$.

**Permutation test:**

1. For each of $B = 10{,}000$ random sign-flip configurations, compute $\bar{d}^{*(b)}$
2. Count how many satisfy $|\bar{d}^{*(b)}| \ge 4.25$
3. Suppose 198 out of $10{,}000$ configurations satisfy this condition

The $p$-value is $(198 + 1)/(10{,}000 + 1) = 0.020$.

Since $n = 12$, we can also enumerate all $2^{12} = 4{,}096$ configurations for an exact $p$-value. Suppose 78 out of $4{,}096$ satisfy $|\bar{d}^{*(b)}| \ge 4.25$:

$$
p_{\text{exact}} = \frac{78}{4096} = 0.019
$$

Both the Monte Carlo and exact $p$-values agree closely.

For comparison, the paired $t$-test gives $t = \bar{d}/(s_d/\sqrt{n}) = -4.25/(3.89/\sqrt{12}) = -3.79$ with $p = 0.003$ ($t_{11}$ distribution). The discrepancy between $p = 0.019$ (permutation) and $p = 0.003$ ($t$-test) suggests the $t$-distribution may not be the best approximation here, possibly due to skewness in the differences.

## Comparison with Related Tests

| Test | Assumption | Statistic | Exactness |
|---|---|---|---|
| Paired $t$-test | Normal differences | $t = \bar{d}/(s_d/\sqrt{n})$ | Approximate (exact if normal) |
| Sign test | Continuous, symmetric | Number of positive $d_i$ | Exact (binomial) |
| Wilcoxon signed-rank | Symmetric differences | Rank sum of positives | Exact (permutation of ranks) |
| Permutation (sign-flip) | Symmetric differences | Any statistic of $d_i$ | Exact (conditional on $|d_i|$) |

The sign-flip permutation test generalizes the Wilcoxon signed-rank test: using the Wilcoxon statistic within the permutation framework yields the same result as the standard Wilcoxon test, but the permutation framework also allows using $\bar{d}$ or any other statistic.

!!! warning "Zero Differences"
    If any $d_i = 0$, the sign flip is irrelevant for that observation (flipping the sign of zero gives zero). Common approaches are to exclude zero differences before testing (reducing $n$) or to assign them randomly to positive or negative with equal probability.

## Summary

The permutation test for paired data exploits the symmetry of the differences under $H_0$ by randomly flipping the signs of the observed differences. Each sign-flip configuration is equally likely under the null, producing an exact conditional distribution of the test statistic. The test is distribution-free (requiring only symmetry of differences under $H_0$), works with any test statistic (mean, Wilcoxon rank sum, trimmed mean), and provides exact $p$-values for any sample size. It is the nonparametric counterpart to the paired $t$-test.
