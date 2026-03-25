# Kruskal-Wallis Test

When comparing three or more independent groups, one-way ANOVA is the standard parametric approach, but it requires normality and equal variances. The **Kruskal-Wallis test** is the non-parametric counterpart: it tests whether the groups share the same distribution by comparing their mean ranks. Like the [Wilcoxon rank-sum test](../two_sample_nonparametric/rank_sum.md), it operates on ranks of the combined sample, inheriting robustness to outliers and non-normality.

When there are only two groups ($k = 2$), the Kruskal-Wallis test reduces to the Wilcoxon rank-sum (Mann-Whitney) test.

## Assumptions

1. The $k$ samples are **independent**.
2. The observations are drawn from **continuous** distributions.
3. The distributions have the **same shape** (the test is primarily sensitive to location differences).

## Hypotheses

$$
H_0 \colon F_1 = F_2 = \cdots = F_k \quad \text{(all groups share the same distribution)}
$$

$$
H_a \colon \text{At least one group differs from the others}
$$

Under the location shift model, this is equivalent to testing whether all group medians are equal.

## Test Statistic

**Step 1.** Combine all $N = \sum_{i=1}^{k} n_i$ observations and rank them from 1 to $N$, using midranks for ties.

**Step 2.** Compute the rank sum $R_i$ and mean rank $\bar{R}_i = R_i / n_i$ for each group $i$.

**Step 3.** The Kruskal-Wallis $H$ statistic is

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
$$

This can equivalently be written as

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} n_i \left(\bar{R}_i - \frac{N+1}{2}\right)^2
$$

which shows that $H$ measures the weighted between-group variance of the mean ranks.

## Tie Correction

When ties are present, divide $H$ by the correction factor:

$$
H_{\text{corrected}} = \frac{H}{1 - \frac{\sum_{j=1}^{g}(t_j^3 - t_j)}{N^3 - N}}
$$

where $g$ is the number of tied groups and $t_j$ is the number of tied observations in the $j$-th group. Without ties, the correction factor equals 1.

## Null Distribution

Under $H_0$, for large sample sizes, $H$ is approximately distributed as

$$
H \sim \chi^2_{k-1}
$$

with $k - 1$ degrees of freedom. The approximation is considered reliable when each $n_i \ge 5$.

For small samples, exact $p$-values can be obtained by enumerating all possible rank assignments.

## Worked Example

Three fertilizers are tested on plant growth (height in cm). Five plants are randomly assigned to each fertilizer.

**Fertilizer A** ($n_1 = 5$): 12, 15, 14, 10, 13

**Fertilizer B** ($n_2 = 5$): 20, 18, 22, 17, 19

**Fertilizer C** ($n_3 = 5$): 8, 11, 9, 7, 10

**Step 1.** Combine and rank ($N = 15$):

| Value | Group | Rank |
|:-----:|:-----:|:----:|
| 7 | C | 1 |
| 8 | C | 2 |
| 9 | C | 3 |
| 10 | A, C | 4.5 |
| 10 | A, C | 4.5 |
| 11 | C | 6 |
| 12 | A | 7 |
| 13 | A | 8 |
| 14 | A | 9 |
| 15 | A | 10 |
| 17 | B | 11 |
| 18 | B | 12 |
| 19 | B | 13 |
| 20 | B | 14 |
| 22 | B | 15 |

**Step 2.** Rank sums:

- $R_A = 4.5 + 7 + 8 + 9 + 10 = 38.5$
- $R_B = 11 + 12 + 13 + 14 + 15 = 65$
- $R_C = 1 + 2 + 3 + 4.5 + 6 = 16.5$

**Check:** $38.5 + 65 + 16.5 = 120 = 15 \times 16/2$. $\checkmark$

**Step 3.** Compute $H$:

$$
H = \frac{12}{15 \times 16}\left(\frac{38.5^2}{5} + \frac{65^2}{5} + \frac{16.5^2}{5}\right) - 3(16)
$$

$$
= \frac{12}{240}\left(\frac{1482.25 + 4225 + 272.25}{5}\right) - 48
$$

$$
= 0.05 \times 1195.9 - 48 = 59.795 - 48 = 11.795
$$

**Step 4.** Compare to $\chi^2_2$: $P(\chi^2_2 > 11.795) \approx 0.003$.

At $\alpha = 0.05$, we reject $H_0$. At least one fertilizer produces significantly different plant growth. The mean ranks ($\bar{R}_A = 7.7$, $\bar{R}_B = 13.0$, $\bar{R}_C = 3.3$) indicate that Fertilizer B produces the tallest plants and Fertilizer C the shortest.

## Post-Hoc Comparisons

A significant Kruskal-Wallis result tells us that at least one group differs, but not which pairs differ. Post-hoc analysis options include:

- **[Dunn's test](dunn.md)** -- pairwise comparisons based on mean rank differences, with $p$-value adjustment for multiple testing (Bonferroni, Holm, or Benjamini-Hochberg).
- **Pairwise Mann-Whitney tests** -- conduct $\binom{k}{2}$ two-sample tests with a Bonferroni correction.

!!! warning "Do not skip the omnibus test"
    Post-hoc pairwise comparisons should only be conducted after the Kruskal-Wallis test rejects $H_0$. Performing pairwise tests without first establishing an overall difference inflates the familywise error rate.

## Summary

The Kruskal-Wallis test extends the rank-sum approach to $k \ge 2$ independent groups by comparing the between-group variance of mean ranks. Under the null hypothesis of identical distributions, the $H$ statistic follows an approximate $\chi^2_{k-1}$ distribution. The test achieves an ARE of $3/\pi \approx 0.955$ relative to one-way ANOVA under normality and can be substantially more powerful under non-normal conditions. When $H$ is significant, post-hoc procedures such as [Dunn's test](dunn.md) identify which pairs of groups differ.
