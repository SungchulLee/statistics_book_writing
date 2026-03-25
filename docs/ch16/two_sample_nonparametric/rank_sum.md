# Wilcoxon Rank-Sum Test

The **Wilcoxon rank-sum test** is the most widely used non-parametric test for comparing two independent samples. It tests whether the two populations have the same distribution -- specifically, whether one group tends to produce systematically larger values than the other. The test works by combining both samples, ranking all observations together, and comparing the sum of ranks assigned to each group.

The Wilcoxon rank-sum test is mathematically equivalent to the [Mann-Whitney U test](mann_whitney.md); the two are different formulations of the same procedure.

## Assumptions

1. The two samples are **independent** of each other.
2. The observations are drawn from **continuous** distributions.
3. The two populations have the **same shape** (under $H_0$, they share the same distribution).

!!! note "Location shift model"
    Many textbooks present the rank-sum test under the **location shift model**: $F_Y(x) = F_X(x - \Delta)$, where $\Delta$ is a shift parameter. Under this model, the test is specifically testing $H_0 \colon \Delta = 0$. More generally, the test detects any form of stochastic ordering, not just location shifts.

## Hypotheses

$$
H_0 \colon F_X = F_Y \quad \text{(the two populations have the same distribution)}
$$

$$
H_a \colon F_X \ne F_Y \quad \text{(two-sided)}
$$

One-sided alternatives: $H_a \colon X$ tends to be larger than $Y$, or vice versa.

## Test Statistic

**Step 1.** Combine the two samples of sizes $n_1$ and $n_2$ into a single sample of size $N = n_1 + n_2$.

**Step 2.** Rank all $N$ observations from smallest to largest, assigning midranks to ties.

**Step 3.** Compute the rank sum for group 1:

$$
W = \sum_{i=1}^{n_1} R_i
$$

where $R_i$ is the rank of the $i$-th observation from group 1 in the combined sample.

Since the total rank sum is fixed at $N(N+1)/2$, the rank sum for group 2 is $W_2 = N(N+1)/2 - W$.

## Null Distribution

Under $H_0$, every assignment of ranks to the two groups is equally likely. The number of ways to choose $n_1$ ranks from $\{1, 2, \ldots, N\}$ is $\binom{N}{n_1}$.

**Mean and variance of $W$ under $H_0$:**

$$
\mu_W = \frac{n_1(N + 1)}{2}
$$

$$
\sigma_W^2 = \frac{n_1 \, n_2 \,(N + 1)}{12}
$$

With a tie correction: if there are $g$ groups of tied observations of sizes $t_1, \ldots, t_g$,

$$
\sigma_W^2 = \frac{n_1 \, n_2}{12}\left(N + 1 - \frac{\sum_{j=1}^{g}(t_j^3 - t_j)}{N(N-1)}\right)
$$

## Normal Approximation

For large samples (typically $n_1, n_2 \ge 10$), the standardized statistic

$$
Z = \frac{W - \mu_W}{\sigma_W}
$$

is approximately $\mathcal{N}(0, 1)$ under $H_0$.

## Worked Example

Two teaching methods are compared using exam scores from two independent groups.

**Group A** ($n_1 = 6$): 78, 64, 85, 72, 91, 80

**Group B** ($n_2 = 6$): 55, 68, 61, 70, 66, 58

**Step 1.** Combine and rank all $N = 12$ observations:

| Value | Group | Rank |
|:-----:|:-----:|:----:|
| 55 | B | 1 |
| 58 | B | 2 |
| 61 | B | 3 |
| 64 | A | 4 |
| 66 | B | 5 |
| 68 | B | 6 |
| 70 | B | 7 |
| 72 | A | 8 |
| 78 | A | 9 |
| 80 | A | 10 |
| 85 | A | 11 |
| 91 | A | 12 |

**Step 2.** Rank sum for Group A: $W = 4 + 8 + 9 + 10 + 11 + 12 = 54$.

**Step 3.** Null parameters:

$$
\mu_W = \frac{6 \times 13}{2} = 39
$$

$$
\sigma_W = \sqrt{\frac{6 \times 6 \times 13}{12}} = \sqrt{39} \approx 6.245
$$

**Step 4.** Standardized statistic:

$$
Z = \frac{54 - 39}{6.245} \approx 2.402
$$

**Step 5.** Two-sided $p$-value:

$$
p = 2\,\Phi(-2.402) \approx 0.016
$$

At $\alpha = 0.05$, we reject $H_0$. Group A's scores are significantly higher than Group B's.

## Relationship to the Mann-Whitney U Test

The Mann-Whitney $U$ statistic is a simple linear transformation of $W$:

$$
U_1 = W - \frac{n_1(n_1 + 1)}{2}
$$

The two tests always produce the same $p$-value. See [Mann-Whitney U Test](mann_whitney.md) for the $U$-statistic formulation and its probability interpretation.

## Exact Tables and Software

For small samples ($n_1, n_2 \le 20$), exact critical values are available in published tables or computed by software (e.g., `scipy.stats.ranksums` or `scipy.stats.mannwhitneyu` in Python). The exact distribution is obtained by enumerating all $\binom{N}{n_1}$ equally likely rank assignments.

## Summary

The Wilcoxon rank-sum test compares two independent samples by combining the observations, assigning ranks, and computing the rank sum for one group. Under the null hypothesis, large or small rank sums are unlikely, and the test detects any systematic tendency for one group to produce larger values. The normal approximation is reliable for moderate to large samples, while exact tables handle small samples. The rank-sum test is equivalent to the Mann-Whitney $U$ test and achieves an ARE of $3/\pi \approx 0.955$ relative to the two-sample $t$-test under normality.
