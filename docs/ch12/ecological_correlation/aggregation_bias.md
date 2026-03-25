# Aggregation Bias

Whenever data are summarized before analysis -- averaged over groups, binned into intervals, or collapsed across time periods -- the resulting statistics can differ systematically from those computed on the original individual-level data. This systematic distortion is called **aggregation bias**. It encompasses the ecological fallacy and Simpson's paradox as special cases, but also includes subtler effects on variances, regression slopes, and correlations that arise purely from the act of aggregation itself.

---

## What Is Aggregation Bias

Aggregation bias occurs when the statistical relationship estimated from aggregated data differs from the relationship that exists at the individual level. The bias arises because aggregation changes the structure of the data in ways that can inflate correlations, distort regression coefficients, and mask or reverse associations.

The core mechanism is straightforward: averaging within groups reduces within-group variability while preserving between-group variability. Since between-group and within-group relationships can differ, the aggregate analysis captures a mixture that may not represent either component faithfully.

---

## How Aggregation Inflates Correlations

Consider $n$ individuals in $G$ groups, each of size $m$ (for simplicity, assume equal group sizes). Let $X_{gi}$ and $Y_{gi}$ denote the values for individual $i$ in group $g$. Define the group means $\bar{X}_g = \frac{1}{m}\sum_{i=1}^m X_{gi}$ and $\bar{Y}_g = \frac{1}{m}\sum_{i=1}^m Y_{gi}$.

The correlation between the group means $\bar{X}_g$ and $\bar{Y}_g$ is the **ecological correlation**. A key mathematical result is that

$$
\text{Var}(\bar{X}_g) = \frac{\text{Var}(X)}{m} \cdot \frac{\text{Var}_B(X)}{\text{Var}(X)/m}
$$

More intuitively, averaging eliminates individual-level noise, causing group means to cluster more tightly along any between-group trend line. This tighter clustering produces higher correlations.

!!! note "Aggregation almost always increases the absolute value of the correlation"
    When within-group and between-group relationships have the same sign, the ecological correlation is stronger than the individual-level correlation. The inflation can be dramatic: individual correlations of $r = 0.3$ routinely become ecological correlations of $r = 0.9$ or higher when averaging over large groups.

---

## Effect on Regression Slopes

Aggregation also distorts regression coefficients. If we regress $Y$ on $X$ at the individual level, the slope is

$$
\hat{\beta}_{\text{individual}} = \frac{\sum_{g,i}(X_{gi} - \bar{X})(Y_{gi} - \bar{Y})}{\sum_{g,i}(X_{gi} - \bar{X})^2}
$$

If we instead regress the group means $\bar{Y}_g$ on $\bar{X}_g$, the ecological slope is

$$
\hat{\beta}_{\text{ecological}} = \frac{\sum_g (\bar{X}_g - \bar{\bar{X}})(\bar{Y}_g - \bar{\bar{Y}})}{\sum_g (\bar{X}_g - \bar{\bar{X}})^2}
$$

These two slopes are generally different. The ecological slope reflects only the between-group relationship, while the individual slope is a combination of between-group and within-group effects. When these components differ, the ecological slope can overestimate, underestimate, or even reverse the individual-level effect.

---

## Sources of Aggregation Bias

Several mechanisms contribute to aggregation bias:

1. **Loss of within-group variation.** Averaging eliminates individual differences within groups, removing information that may be essential for estimating the true relationship.

2. **Confounding by group membership.** Groups may differ systematically in ways that are correlated with both $X$ and $Y$, introducing confounding at the aggregate level that does not exist (or is weaker) at the individual level.

3. **Nonlinear effects.** When the relationship between $X$ and $Y$ is nonlinear, the average of a function is not the function of the average. Aggregation applies a linear approximation, distorting the estimated relationship.

4. **Unequal group sizes.** When groups have different sizes, the aggregate analysis implicitly assigns different weights to different individuals, which can skew results.

---

## Example: State-Level vs Individual-Level Income and Education

Suppose we correlate average income and average education level across 50 U.S. states and find $r = 0.85$. This ecological correlation is much higher than the individual-level correlation (typically around $r = 0.4$) for two reasons:

- Averaging over millions of people within each state eliminates individual-level noise, tightening the scatter of group means.
- State-level differences in industry, policy, and demographics create between-state variation that aligns income and education more strongly than individual-level variation does.

Concluding that education explains 72% of income variation ($r^2 = 0.72$) would be a serious overstatement of the individual-level relationship ($r^2 = 0.16$).

---

## Mitigating Aggregation Bias

1. **Analyze at the individual level** whenever individual data are available.

2. **Use multilevel models.** Hierarchical models explicitly separate within-group and between-group effects, estimating each appropriately.

3. **Report the level of analysis.** Always state whether statistics are computed on individual or aggregated data.

4. **Avoid cross-level inference.** Do not use aggregate statistics to draw conclusions about individuals, and vice versa.

5. **Check sensitivity to aggregation.** If only aggregate data are available, perform sensitivity analyses to understand how results might change at finer levels of aggregation.

---

## Connection to the Ecological Fallacy and Simpson's Paradox

Aggregation bias is the umbrella concept under which the [ecological fallacy](ecological_fallacy.md) and [Simpson's paradox](simpsons_paradox.md) fall:

- The **ecological fallacy** is the specific error of applying aggregate-level correlations to individuals.
- **Simpson's paradox** is the specific case where aggregation reverses the direction of an association.
- **Aggregation bias** is the general phenomenon that any form of data summarization can systematically distort statistical relationships.

Understanding aggregation bias provides the unified framework for recognizing when and why summarized data can mislead.

---

## Summary

Aggregation bias is the systematic distortion of statistical relationships that occurs when data are analyzed at a more aggregated level than they were generated. Averaging inflates correlations, distorts regression slopes, and can reverse associations. The bias arises from the loss of within-group variation and the potential for confounding at the group level. Individual-level analysis and multilevel models are the primary defenses against aggregation bias.
