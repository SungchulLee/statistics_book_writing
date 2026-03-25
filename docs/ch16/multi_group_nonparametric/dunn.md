# Post-Hoc Dunn's Test

When the [Kruskal-Wallis test](kruskal_wallis.md) rejects the null hypothesis, it tells us that at least one group differs from the others -- but not *which* groups differ. **Dunn's test** is a post-hoc multiple comparison procedure designed specifically for this purpose. It compares all pairs of groups using the mean ranks from the Kruskal-Wallis analysis and adjusts the $p$-values for multiple testing.

## Motivation

After a significant Kruskal-Wallis result with $k$ groups, there are $\binom{k}{2} = k(k-1)/2$ possible pairwise comparisons. Performing each at level $\alpha$ without adjustment inflates the familywise error rate (FWER). Dunn's test controls this inflation by using a Bonferroni correction (or other adjustments) on the pairwise $p$-values.

## Test Statistic

For each pair of groups $(i, j)$, Dunn's test compares their mean ranks. Let $\bar{R}_i$ and $\bar{R}_j$ be the mean ranks of groups $i$ and $j$ from the combined ranking used in the Kruskal-Wallis test. The test statistic is

$$
z_{ij} = \frac{\bar{R}_i - \bar{R}_j}{\sigma_{ij}}
$$

where the standard error is

$$
\sigma_{ij} = \sqrt{\frac{N(N+1)}{12}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

Here $N = \sum_{i=1}^k n_i$ is the total sample size and $n_i, n_j$ are the group sizes.

### Tie Correction

When ties are present in the combined sample, the standard error becomes

$$
\sigma_{ij} = \sqrt{\left[\frac{N(N+1)}{12} - \frac{\sum_{l=1}^{g}(t_l^3 - t_l)}{12(N-1)}\right]\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

where $g$ is the number of tied groups and $t_l$ is the number of tied observations in the $l$-th group.

## p-Value and Multiple Testing Adjustment

The unadjusted two-sided $p$-value for each pair is

$$
p_{ij} = 2\,\Phi(-|z_{ij}|)
$$

To control the FWER, the $p$-values are adjusted. Common methods include:

| Method | Adjustment | Properties |
|:-------|:-----------|:-----------|
| **Bonferroni** | $p_{ij}^* = \min\!\bigl(m \cdot p_{ij}, \; 1\bigr)$ where $m = \binom{k}{2}$ | Conservative, controls FWER |
| **Holm** (step-down) | Rank $p$-values; $p_{(r)}^* = \min\!\bigl((m - r + 1) \cdot p_{(r)}, \; 1\bigr)$ | Less conservative than Bonferroni, controls FWER |
| **Benjamini-Hochberg** | $p_{(r)}^* = \min\!\bigl(m \cdot p_{(r)} / r, \; 1\bigr)$ | Controls FDR instead of FWER |

!!! tip "Holm vs Bonferroni"
    The Holm step-down procedure is uniformly more powerful than the Bonferroni correction while still controlling the FWER at level $\alpha$. It should generally be preferred unless the stricter Bonferroni correction is specifically required.

## Worked Example

Continuing the fertilizer example from the [Kruskal-Wallis](kruskal_wallis.md) section, the Kruskal-Wallis test rejected $H_0$ with $H = 11.795$, $p \approx 0.003$. The three groups had:

- Fertilizer A: $n_1 = 5$, $\bar{R}_A = 7.7$
- Fertilizer B: $n_2 = 5$, $\bar{R}_B = 13.0$
- Fertilizer C: $n_3 = 5$, $\bar{R}_C = 3.3$

With $N = 15$ and $m = 3$ pairwise comparisons:

**Standard error** (assuming no ties for simplicity):

$$
\sigma_{ij} = \sqrt{\frac{15 \times 16}{12}\left(\frac{1}{5} + \frac{1}{5}\right)} = \sqrt{20 \times 0.4} = \sqrt{8} \approx 2.828
$$

**Pairwise comparisons:**

| Pair | $|\bar{R}_i - \bar{R}_j|$ | $z_{ij}$ | $p_{ij}$ | $p_{ij}^*$ (Bonferroni) |
|:-----|:----:|:----:|:------:|:------:|
| A vs B | 5.3 | 1.874 | 0.061 | 0.183 |
| A vs C | 4.4 | 1.556 | 0.120 | 0.360 |
| B vs C | 9.7 | 3.430 | 0.0006 | 0.0018 |

**Interpretation at $\alpha = 0.05$:**

- **B vs C:** $p^* = 0.0018 < 0.05$. Fertilizer B and C differ significantly. Fertilizer B produces taller plants (higher mean rank).
- **A vs B:** $p^* = 0.183 > 0.05$. No significant difference between A and B.
- **A vs C:** $p^* = 0.360 > 0.05$. No significant difference between A and C.

The Kruskal-Wallis rejection was driven primarily by the large difference between Fertilizer B and Fertilizer C.

## Procedure Summary

1. Run the [Kruskal-Wallis test](kruskal_wallis.md). Proceed to Dunn's test only if $H_0$ is rejected.
2. For each pair $(i, j)$, compute $z_{ij}$ using the mean ranks from the Kruskal-Wallis analysis.
3. Compute unadjusted $p$-values from the standard normal distribution.
4. Apply a multiple testing correction (Bonferroni, Holm, or Benjamini-Hochberg).
5. Declare pairs significant where the adjusted $p$-value is below $\alpha$.

## Comparison with Pairwise Mann-Whitney

An alternative post-hoc strategy is to conduct pairwise Mann-Whitney $U$ tests with a Bonferroni correction. The key difference is that Dunn's test uses the *combined* ranking from the Kruskal-Wallis analysis, while pairwise Mann-Whitney re-ranks the observations for each pair. Dunn's test is more commonly used because it is consistent with the global ranking used in the omnibus test.

## Summary

Dunn's test performs pairwise comparisons after a significant Kruskal-Wallis result by computing $z$-statistics based on mean rank differences. Multiple testing adjustments (Bonferroni, Holm, or Benjamini-Hochberg) control the error rate across all $\binom{k}{2}$ comparisons. The test uses the same combined ranking as the Kruskal-Wallis test, ensuring consistency between the omnibus and post-hoc analyses.
