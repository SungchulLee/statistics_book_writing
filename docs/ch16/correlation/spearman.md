# Spearman's Rank Correlation (Revisited)

Spearman's rank correlation coefficient $r_s$ was introduced in [Chapter 12](../../ch12/correlation/spearman.md) as a measure of monotonic association between two variables. This section revisits $r_s$ from the non-parametric testing perspective, focusing on hypothesis testing, the handling of ties, and the connection to the rank-based framework developed throughout this chapter.

## Definition

Given paired observations $(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$, let $R_i$ be the rank of $X_i$ among $X_1, \ldots, X_n$ and let $S_i$ be the rank of $Y_i$ among $Y_1, \ldots, Y_n$. Spearman's correlation is the Pearson correlation computed on the ranks:

$$
r_s = \frac{\sum_{i=1}^{n}(R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^{n}(R_i - \bar{R})^2 \sum_{i=1}^{n}(S_i - \bar{S})^2}}
$$

When there are no ties, this simplifies to

$$
r_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
$$

where $d_i = R_i - S_i$ is the difference between the paired ranks.

The coefficient satisfies $-1 \le r_s \le 1$, with $r_s = 1$ indicating a perfect monotonically increasing relationship and $r_s = -1$ a perfect monotonically decreasing one.

## Hypothesis Test

### Hypotheses

$$
H_0 \colon \rho_s = 0 \quad \text{(no monotonic association)}
$$

$$
H_a \colon \rho_s \ne 0 \quad \text{(two-sided)}
$$

where $\rho_s$ is the population Spearman correlation.

### Test Statistic

For large $n$ (typically $n \ge 10$), the statistic

$$
t = r_s \sqrt{\frac{n - 2}{1 - r_s^2}}
$$

follows approximately a $t$-distribution with $n - 2$ degrees of freedom under $H_0$.

For small $n$, exact critical values based on the permutation distribution of $r_s$ are available in tables.

### Exact Null Distribution

Under $H_0$ (independence), every permutation of the ranks is equally likely. There are $n!$ possible rank permutations, and the exact null distribution of $r_s$ can be obtained by computing $r_s$ for each. For small $n$, this is feasible; for larger $n$, the $t$-approximation is used.

## Handling Ties

When tied observations occur, the simplified formula $1 - 6\sum d_i^2 / [n(n^2-1)]$ is no longer exact. Instead, the full Pearson formula applied to midranks should be used:

$$
r_s = \frac{\sum_{i=1}^{n}(R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^{n}(R_i - \bar{R})^2 \sum_{i=1}^{n}(S_i - \bar{S})^2}}
$$

where $R_i$ and $S_i$ are the midranks. This formula naturally handles ties and reduces to the simplified version when no ties are present.

!!! warning "Do not use the shortcut formula with ties"
    The formula $r_s = 1 - 6\sum d_i^2 / [n(n^2-1)]$ assumes no tied ranks. When ties are present, this formula can produce values outside $[-1, 1]$ or give incorrect results. Always use the full Pearson-on-ranks formula when ties exist.

## Worked Example

An analyst examines the relationship between employee experience (years) and customer satisfaction rating (1--10 scale) for 8 employees.

| Employee | Experience ($X$) | Satisfaction ($Y$) | $R_i$ | $S_i$ | $d_i$ | $d_i^2$ |
|:--------:|:------:|:------:|:--:|:--:|:---:|:---:|
| 1 | 2 | 7 | 1 | 4 | $-3$ | 9 |
| 2 | 5 | 8 | 3 | 5.5 | $-2.5$ | 6.25 |
| 3 | 3 | 6 | 2 | 2.5 | $-0.5$ | 0.25 |
| 4 | 8 | 9 | 5.5 | 7.5 | $-2$ | 4 |
| 5 | 10 | 10 | 7.5 | 8 | $-0.5$ | 0.25 (not used) |
| 6 | 8 | 8 | 5.5 | 5.5 | 0 | 0 |
| 7 | 10 | 6 | 7.5 | 2.5 | 5 | 25 (not used) |
| 8 | 6 | 9 | 4 | 7.5 | $-3.5$ | 12.25 (not used) |

Because of ties (Experience: 8 appears twice, 10 appears twice; Satisfaction: 8 appears twice, 6 appears twice, 9 appears twice), we use the full Pearson formula on the midranks.

Computing directly from the midranks: $r_s \approx 0.619$.

**Test:** $t = 0.619\sqrt{6/(1 - 0.619^2)} = 0.619\sqrt{6/0.617} \approx 0.619 \times 3.118 \approx 1.930$.

With $n - 2 = 6$ degrees of freedom, $p \approx 0.10$ (two-sided). At $\alpha = 0.05$, we fail to reject $H_0$.

## Comparison with Pearson's r

| Feature | Pearson's $r$ | Spearman's $r_s$ |
|:--------|:-------------|:-----------------|
| Measures | Linear association | Monotonic association |
| Sensitive to | Outliers, non-linearity | Robust to both |
| Assumes | Bivariate normality (for testing) | None (rank-based) |
| ARE vs Pearson (bivariate normal) | 1.0 | $3/\pi \approx 0.955$ |
| Detects non-linear monotonic trends | Poorly | Well |

## When to Prefer Spearman over Pearson

- The relationship is monotonic but not linear (e.g., diminishing returns).
- The data contain outliers that could inflate or deflate Pearson's $r$.
- The data are ordinal (e.g., survey ratings, rankings).
- Bivariate normality is not a reasonable assumption.

## Summary

Spearman's $r_s$ is the Pearson correlation coefficient applied to the ranks of the observations, providing a non-parametric measure of monotonic association. The hypothesis test uses a $t$-approximation for large samples and the exact permutation distribution for small samples. Ties are handled by using midranks in the full Pearson formula. Spearman's $r_s$ achieves an ARE of $3/\pi \approx 0.955$ relative to Pearson's $r$ under bivariate normality and can be substantially more informative under non-normality or non-linearity.
