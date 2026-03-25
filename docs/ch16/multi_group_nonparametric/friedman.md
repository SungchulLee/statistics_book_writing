# Friedman Test (Repeated Measures)

The [Kruskal-Wallis test](kruskal_wallis.md) compares independent groups. When the same subjects are measured under multiple conditions (repeated measures) or when data are collected in matched blocks, the observations are no longer independent across groups. The **Friedman test** is the non-parametric counterpart to repeated-measures ANOVA: it accounts for the block structure by ranking observations *within each block* rather than across the entire sample.

## Assumptions

1. There are $b$ blocks (e.g., subjects) and $k$ treatments (conditions).
2. Within each block, the $k$ observations can be ranked.
3. The blocks are independent of each other.
4. The data are at least ordinal within each block.

## Hypotheses

$$
H_0 \colon \text{All } k \text{ treatments have the same effect (identical distributions within blocks)}
$$

$$
H_a \colon \text{At least one treatment differs from the others}
$$

## Procedure

**Step 1.** For each block $j = 1, \ldots, b$, rank the $k$ observations from 1 to $k$. Assign midranks to ties within a block.

**Step 2.** Compute the rank sum for each treatment $i$:

$$
R_i = \sum_{j=1}^{b} r_{ij}
$$

where $r_{ij}$ is the rank of treatment $i$ in block $j$.

**Step 3.** Compute the Friedman statistic:

$$
\chi^2_F = \frac{12}{bk(k+1)} \sum_{i=1}^{k} R_i^2 - 3b(k+1)
$$

This can equivalently be written as

$$
\chi^2_F = \frac{12}{bk(k+1)} \sum_{i=1}^{k} \left(R_i - \frac{b(k+1)}{2}\right)^2
$$

which shows that $\chi^2_F$ measures how much the treatment rank sums deviate from their common expected value under $H_0$.

## Null Distribution

Under $H_0$, within each block every permutation of the ranks $\{1, 2, \ldots, k\}$ is equally likely. For large $b$, the Friedman statistic is approximately distributed as

$$
\chi^2_F \sim \chi^2_{k-1}
$$

For small $b$ and $k$, exact $p$-values are available from tables or software.

## Worked Example

Four pain medications are tested on 5 patients. Each patient rates their pain relief on a 1--100 scale for each medication.

| Patient | Drug A | Drug B | Drug C | Drug D |
|:-------:|:------:|:------:|:------:|:------:|
| 1 | 30 | 45 | 60 | 50 |
| 2 | 25 | 40 | 55 | 35 |
| 3 | 35 | 50 | 65 | 45 |
| 4 | 20 | 30 | 50 | 40 |
| 5 | 40 | 55 | 70 | 60 |

**Step 1.** Rank within each patient (block):

| Patient | Drug A | Drug B | Drug C | Drug D |
|:-------:|:------:|:------:|:------:|:------:|
| 1 | 1 | 2 | 4 | 3 |
| 2 | 1 | 3 | 4 | 2 |
| 3 | 1 | 3 | 4 | 2 |
| 4 | 1 | 2 | 4 | 3 |
| 5 | 1 | 2 | 4 | 3 |

**Step 2.** Rank sums: $R_A = 5$, $R_B = 12$, $R_C = 20$, $R_D = 13$.

**Check:** $5 + 12 + 20 + 13 = 50 = 5 \times 4 \times (4+1)/2$. $\checkmark$

**Step 3.** Compute $\chi^2_F$:

$$
\chi^2_F = \frac{12}{5 \times 4 \times 5}(5^2 + 12^2 + 20^2 + 13^2) - 3 \times 5 \times 5
$$

$$
= \frac{12}{100}(25 + 144 + 400 + 169) - 75 = \frac{12 \times 738}{100} - 75 = 88.56 - 75 = 13.56
$$

**Step 4.** Compare to $\chi^2_3$: $P(\chi^2_3 > 13.56) \approx 0.0036$.

At $\alpha = 0.05$, we reject $H_0$. There is significant evidence that the medications differ in pain relief effectiveness. The mean ranks (Drug A: 1.0, Drug B: 2.4, Drug C: 4.0, Drug D: 2.6) indicate that Drug C provides the most relief and Drug A the least.

## Post-Hoc Comparisons

After a significant Friedman test, pairwise comparisons can identify which treatments differ. The most common approach is the **Nemenyi test**, which compares all pairs of mean ranks:

$$
|\bar{R}_i - \bar{R}_j| > q_\alpha \sqrt{\frac{k(k+1)}{6b}}
$$

where $q_\alpha$ is the critical value from the studentized range distribution. Alternatively, pairwise Wilcoxon signed-rank tests with a Bonferroni correction can be used.

!!! tip "Connection to the Kruskal-Wallis test"
    The Kruskal-Wallis test ranks all $N$ observations globally, while the Friedman test ranks observations *within each block*. This within-block ranking removes between-block variability, analogous to how repeated-measures ANOVA removes between-subject variability. When the block structure is present, the Friedman test is more powerful than the Kruskal-Wallis test.

## Summary

The Friedman test extends non-parametric group comparison to repeated-measures and randomized block designs by ranking observations within each block. The test statistic $\chi^2_F$ measures the variance of treatment rank sums and follows an approximate $\chi^2_{k-1}$ distribution under the null hypothesis. The test is the non-parametric analogue of repeated-measures ANOVA and is applicable whenever the data within blocks can be meaningfully ranked, even if they are ordinal. Post-hoc procedures such as the Nemenyi test identify which specific treatments differ.
