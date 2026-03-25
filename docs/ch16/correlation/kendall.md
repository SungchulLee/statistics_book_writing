# Kendall's Tau (Revisited)

Kendall's rank correlation coefficient $\tau$ was introduced in [Chapter 12](../../ch12/correlation/kendall.md) as a concordance-based measure of monotonic association. This section revisits $\tau$ from the non-parametric testing perspective, covering the concordance/discordance framework, hypothesis testing, handling ties, and the comparison with [Spearman's $r_s$](spearman.md).

## Concordant and Discordant Pairs

Given $n$ paired observations $(X_1, Y_1), \ldots, (X_n, Y_n)$, consider all $\binom{n}{2}$ pairs of observations $(i, j)$ with $i < j$. A pair is:

- **Concordant** if $(X_i - X_j)(Y_i - Y_j) > 0$ -- both variables move in the same direction.
- **Discordant** if $(X_i - X_j)(Y_i - Y_j) < 0$ -- the variables move in opposite directions.
- **Tied** if $X_i = X_j$ or $Y_i = Y_j$ (or both).

Let $C$ denote the number of concordant pairs and $D$ the number of discordant pairs.

## Definition

### Kendall's tau-a (no ties)

When there are no ties:

$$
\tau_a = \frac{C - D}{\binom{n}{2}} = \frac{C - D}{n(n-1)/2}
$$

The numerator $S = C - D$ is called the **Kendall S statistic**.

### Kendall's tau-b (with ties)

When ties are present, tau-b adjusts the denominator:

$$
\tau_b = \frac{C - D}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}
$$

where

- $n_0 = n(n-1)/2$ is the total number of pairs,
- $n_1 = \sum_{i} t_i(t_i - 1)/2$ is the number of pairs tied on $X$ ($t_i$ = size of the $i$-th tied group in $X$),
- $n_2 = \sum_{j} u_j(u_j - 1)/2$ is the number of pairs tied on $Y$ ($u_j$ = size of the $j$-th tied group in $Y$).

Tau-b satisfies $-1 \le \tau_b \le 1$ and equals $\pm 1$ only when the data can be perfectly described by a monotonic relationship (accounting for ties).

## Hypothesis Test

### Hypotheses

$$
H_0 \colon \tau = 0 \quad \text{(no monotonic association)}
$$

$$
H_a \colon \tau \ne 0 \quad \text{(two-sided)}
$$

### Null Distribution of S

Under $H_0$ (independence), the $S$ statistic has

$$
E[S] = 0
$$

$$
\text{Var}(S) = \frac{n(n-1)(2n+5)}{18}
$$

When ties are present, the variance adjusts to

$$
\text{Var}(S) = \frac{1}{18}\left[n(n-1)(2n+5) - \sum_{i} t_i(t_i-1)(2t_i+5) - \sum_{j} u_j(u_j-1)(2u_j+5)\right] + \frac{\sum_{i}t_i(t_i-1)(t_i-2) \cdot \sum_{j}u_j(u_j-1)(u_j-2)}{9n(n-1)(n-2)} + \frac{\sum_{i}t_i(t_i-1) \cdot \sum_{j}u_j(u_j-1)}{2n(n-1)}
$$

### Normal Approximation

For $n \ge 10$, the standardized statistic

$$
Z = \frac{S}{\sqrt{\text{Var}(S)}}
$$

is approximately $\mathcal{N}(0, 1)$ under $H_0$.

For small $n$, exact $p$-values can be computed from the permutation distribution.

## Worked Example

Six students are ranked by two judges on presentation quality.

| Student | Judge 1 rank ($R_i$) | Judge 2 rank ($S_i$) |
|:-------:|:------:|:------:|
| A | 1 | 2 |
| B | 2 | 1 |
| C | 3 | 4 |
| D | 4 | 3 |
| E | 5 | 6 |
| F | 6 | 5 |

**Count concordant and discordant pairs** (no ties, so tau-a applies):

For each pair $(i, j)$ with $i < j$, check whether $(R_i - R_j)$ and $(S_i - S_j)$ have the same sign.

There are $\binom{6}{2} = 15$ pairs. Enumeration yields $C = 12$ concordant and $D = 3$ discordant.

$$
S = C - D = 12 - 3 = 9
$$

$$
\tau_a = \frac{9}{15} = 0.600
$$

**Test:**

$$
\text{Var}(S) = \frac{6 \times 5 \times 17}{18} = \frac{510}{18} \approx 28.33
$$

$$
Z = \frac{9}{\sqrt{28.33}} \approx \frac{9}{5.323} \approx 1.691
$$

$$
p = 2\,\Phi(-1.691) \approx 0.091
$$

At $\alpha = 0.05$, we fail to reject $H_0$. With only 6 observations, the test lacks power to detect a moderate concordance.

## Comparison with Spearman's Correlation

| Feature | Spearman's $r_s$ | Kendall's $\tau$ |
|:--------|:-----------------|:-----------------|
| Based on | Rank differences | Concordant/discordant pair counts |
| Range | $[-1, 1]$ | $[-1, 1]$ |
| Typical magnitude | Larger (closer to $\pm 1$) | Smaller (typically $|\tau| < |r_s|$) |
| Interpretation | Pearson correlation of ranks | Probability of concordance minus discordance |
| Computational complexity | $O(n \log n)$ | $O(n^2)$ (naive), $O(n \log n)$ (merge sort) |
| Tie handling | Midranks | Tau-b adjustment |
| Small-sample distribution | Less tractable | More tractable |

!!! note "Probability interpretation"
    Kendall's $\tau$ has a direct probability interpretation. For a randomly chosen pair $(i, j)$:

    $$\tau = P(\text{concordant}) - P(\text{discordant})$$

    This makes $\tau$ easier to interpret than $r_s$ in many applied settings. A value of $\tau = 0.4$ means concordant pairs outnumber discordant pairs by a margin corresponding to 40% of all pairs.

## When to Prefer Kendall's Tau

- **Small sample sizes** -- the exact distribution of $\tau$ is more tractable than that of $r_s$.
- **Many ties** -- tau-b handles ties more naturally than the midrank adjustment for Spearman.
- **Interpretability** -- the concordance/discordance interpretation is more intuitive in some domains (e.g., comparing judges' rankings, preference orderings).

## Summary

Kendall's $\tau$ measures monotonic association by comparing the number of concordant and discordant observation pairs. The hypothesis test uses the $S = C - D$ statistic, which under independence is approximately normal with mean zero. The tau-b variant adjusts for ties in either variable. Compared to Spearman's $r_s$, Kendall's $\tau$ tends to be smaller in magnitude but offers a clearer probability interpretation and better small-sample distributional properties.
