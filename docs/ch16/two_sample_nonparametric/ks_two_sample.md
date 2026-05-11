# Kolmogorov-Smirnov Two-Sample Test

The [Wilcoxon rank-sum](rank_sum.md) and [Mann-Whitney U](mann_whitney.md) tests are primarily sensitive to **location shifts** -- differences in the central tendency of two distributions. The **Kolmogorov-Smirnov (KS) two-sample test** takes a broader view: it compares the *entire* empirical cumulative distribution functions (ECDFs) of two samples and can detect differences in location, spread, shape, or any other distributional feature.

This generality makes the KS test a versatile diagnostic tool, though it comes at the cost of lower power against specific alternatives (such as pure location shifts) compared to rank-based tests.

## Intuition

Given two independent samples, each sample generates an empirical CDF -- a step function that jumps by $1/n$ at each observed value. If the two populations are identical, their ECDFs should track each other closely. The KS test measures the largest vertical gap between the two ECDFs and rejects $H_0$ when this gap is too large to be explained by sampling variability alone.

## Hypotheses

$$
H_0 \colon F_X = F_Y \quad \text{(the two populations have the same continuous distribution)}
$$

$$
H_a \colon F_X \ne F_Y \quad \text{(the distributions differ in some way)}
$$

The test is inherently two-sided: it detects *any* difference between the distributions.

## Test Statistic

Let $\hat{F}_1(x)$ and $\hat{F}_2(x)$ be the ECDFs of the two samples of sizes $n_1$ and $n_2$. The KS statistic is the supremum of the absolute difference:

$$
D_{n_1, n_2} = \sup_{x \in \mathbb{R}} \left|\hat{F}_1(x) - \hat{F}_2(x)\right|
$$

In practice, $D$ is computed by combining and sorting all $N = n_1 + n_2$ observations and evaluating $|\hat{F}_1 - \hat{F}_2|$ at each observed value (and just before each jump).

## Null Distribution

Under $H_0$ with continuous distributions, the null distribution of $D_{n_1, n_2}$ does not depend on the common distribution $F$ -- the test is distribution-free.

For large samples, the scaled statistic

$$
\sqrt{\frac{n_1 \, n_2}{n_1 + n_2}} \, D_{n_1, n_2}
$$

converges in distribution to the **Kolmogorov distribution**, whose CDF is

$$
K(t) = 1 - 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2 t^2}
$$

The asymptotic $p$-value is $p = 1 - K(c)$ where $c = \sqrt{n_1 n_2 / (n_1 + n_2)} \cdot D$.

For small samples, exact $p$-values are computed by enumeration or dynamic programming.

## Worked Example

Two manufacturing processes produce ball bearings. We measure diameter (mm) for samples from each process.

**Process A** ($n_1 = 5$): 10.1, 10.3, 10.2, 10.5, 10.4

**Process B** ($n_2 = 5$): 10.0, 10.2, 10.6, 10.8, 10.4

**Step 1.** Sort the combined sample and compute ECDFs:

| Value | $\hat{F}_1(x)$ | $\hat{F}_2(x)$ | $|\hat{F}_1 - \hat{F}_2|$ |
|:-----:|:-----:|:-----:|:-----:|
| 10.0 | 0/5 = 0.0 | 1/5 = 0.2 | 0.2 |
| 10.1 | 1/5 = 0.2 | 1/5 = 0.2 | 0.0 |
| 10.2 | 2/5 = 0.4 | 2/5 = 0.4 | 0.0 |
| 10.3 | 3/5 = 0.6 | 2/5 = 0.4 | 0.2 |
| 10.4 | 4/5 = 0.8 | 3/5 = 0.6 | 0.2 |
| 10.5 | 5/5 = 1.0 | 3/5 = 0.6 | 0.4 |
| 10.6 | 5/5 = 1.0 | 4/5 = 0.8 | 0.2 |
| 10.8 | 5/5 = 1.0 | 5/5 = 1.0 | 0.0 |

**Step 2.** The KS statistic is $D = 0.4$, occurring at $x = 10.5$.

**Step 3.** Scaled statistic: $\sqrt{5 \times 5 / 10} \times 0.4 = \sqrt{2.5} \times 0.4 \approx 0.632$.

Using the Kolmogorov distribution (or exact tables for $n_1 = n_2 = 5$), the $p$-value is approximately $0.73$.

At $\alpha = 0.05$, we fail to reject $H_0$. The two processes do not show a significant difference in diameter distributions.

## What the KS Test Detects

Unlike rank-based tests that focus on location, the KS test is sensitive to differences in:

- **Location** -- one distribution shifted relative to the other
- **Scale** -- one distribution more spread out
- **Shape** -- different skewness, kurtosis, or modality
- **Any combination** of the above

!!! warning "Lower power for specific alternatives"
    The KS test's generality comes at a cost. For a pure location shift, the Wilcoxon rank-sum test will typically have higher power. The KS test is most useful when the nature of the difference is unknown or when differences in shape or spread are of interest.

## Comparison with Rank-Based Tests

| Feature | Wilcoxon Rank-Sum | KS Two-Sample |
|:--------|:-----------------|:--------------|
| Detects location shift | High power | Moderate power |
| Detects spread difference | Low power | Moderate power |
| Detects shape difference | Low power | Moderate power |
| Test statistic | Sum of ranks ($W$) | Maximum ECDF gap ($D$) |
| Handles ties | Via midranks | Requires continuity |
| Effect size interpretation | $P(X > Y)$ via $U$ | Maximum distributional gap |

## Summary

The Kolmogorov-Smirnov two-sample test compares the entire empirical distribution functions of two independent samples by computing the maximum absolute difference $D = \sup|\hat{F}_1(x) - \hat{F}_2(x)|$. It is distribution-free under the null hypothesis of identical continuous distributions and can detect differences in location, spread, and shape. This versatility makes it a useful complement to rank-based tests, especially when the alternative hypothesis is not restricted to a location shift. For pure location alternatives, the Wilcoxon rank-sum test generally provides higher power.

## Exercises

**Exercise 1.**
Two groups of students take different preparation courses, and their exam scores are:

- **Course 1**: 72, 78, 85, 90, 65
- **Course 2**: 80, 88, 92, 95, 85, 76

**(a)** Compute the empirical CDF $\hat{F}_1(x)$ and $\hat{F}_2(x)$ for each group.

**(b)** Find the Kolmogorov-Smirnov test statistic $D = \max_x |\hat{F}_1(x) - \hat{F}_2(x)|$.

**(c)** Explain what a large value of $D$ indicates about the two distributions.

??? success "Solution to Exercise 1"

    **(a)** Sort each sample and compute the ECDF (each jump has size $1/n_i$):

    **Course 1** ($n_1 = 5$): 65, 72, 78, 85, 90. ECDF jumps by $1/5 = 0.2$ at each value.

    **Course 2** ($n_2 = 6$): 76, 80, 85, 88, 92, 95. ECDF jumps by $1/6 \approx 0.167$ at each value.

    **(b)** To find $D$, evaluate $|\hat{F}_1(x) - \hat{F}_2(x)|$ at every observed value:

    | $x$ | $\hat{F}_1(x)$ | $\hat{F}_2(x)$ | $|\hat{F}_1 - \hat{F}_2|$ |
    |:---:|:---:|:---:|:---:|
    | 65 | 0.2 | 0 | 0.200 |
    | 72 | 0.4 | 0 | 0.400 |
    | 76 | 0.4 | 1/6 | 0.233 |
    | 78 | 0.6 | 1/6 | 0.433 |
    | 80 | 0.6 | 2/6 | 0.267 |
    | 85 | 0.8 | 3/6 | 0.300 |
    | 88 | 0.8 | 4/6 | 0.133 |
    | 90 | 1.0 | 4/6 | 0.333 |
    | 92 | 1.0 | 5/6 | 0.167 |
    | 95 | 1.0 | 1.0 | 0.000 |

    $$
    D = \max_x |\hat{F}_1(x) - \hat{F}_2(x)| = 0.433 \text{ (at } x = 78\text{)}
    $$

    **(c)** A large value of $D$ indicates that the two empirical CDFs differ substantially, suggesting the two samples come from different underlying distributions. The KS test is sensitive to differences in both location (shift) and shape. For these data, $D = 0.433$ suggests Course 2 students tend to score higher, but with $n_1 = 5$ and $n_2 = 6$, the critical value at $\alpha = 0.05$ is approximately $c(\alpha)\sqrt{(n_1 + n_2)/(n_1 n_2)} = 1.36\sqrt{11/30} \approx 0.823$, so $D = 0.433 < 0.823$ and we would not reject $H_0$ at this sample size.
