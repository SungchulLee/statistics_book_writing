# When and Why to Use Non-Parametric Tests

Parametric tests such as the $t$-test and ANOVA derive their power from strong distributional assumptions -- most commonly that the data follow a normal distribution. When those assumptions hold, parametric methods are the most efficient tools available. However, real-world data frequently violate these assumptions: distributions may be heavily skewed, contaminated by outliers, or measured on an ordinal scale that lacks meaningful numerical distances. Non-parametric tests provide a principled alternative in all of these situations.

This section motivates the use of **distribution-free** (non-parametric) methods by examining the conditions under which parametric assumptions break down and by clarifying the trade-offs involved in choosing between the two families of tests.

## When Parametric Assumptions Fail

### Normality Violations

Many parametric procedures assume that the population distribution is normal, or that the sampling distribution of the test statistic is approximately normal. The central limit theorem guarantees the latter for large samples, but for small samples the normality of the underlying data matters directly. Common violations include:

- **Heavy-tailed distributions** -- Data from Cauchy, $t$ with small degrees of freedom, or contaminated normal distributions produce extreme observations far more often than a Gaussian model predicts.
- **Skewed distributions** -- Income data, survival times, and many biological measurements follow right-skewed distributions (e.g., log-normal, exponential, Weibull).
- **Multimodal distributions** -- Mixture populations create multiple peaks that no single parametric family can capture.

When the sample size is small (say $n < 30$) and the population departs substantially from normality, the Type I error rate of a $t$-test can deviate significantly from the nominal $\alpha$.

### Ordinal or Ranked Data

Some measurement scales are inherently ordinal: pain ratings on a 1--10 scale, letter grades, or Likert-scale survey responses. Arithmetic operations like computing a mean are not meaningful for ordinal data because the distances between categories are not guaranteed to be equal. Non-parametric tests, which operate on ranks rather than raw values, respect the ordinal structure of such data.

### Outliers and Contamination

A single extreme observation can dramatically shift a sample mean and inflate the sample variance, distorting the results of any test that depends on these quantities. Rank-based tests are inherently resistant to outliers because replacing each observation with its rank bounds the influence of any single data point to at most one rank position.

### Small Sample Sizes

With very small samples ($n < 10$), the normal approximation underlying many parametric tests becomes unreliable. Several non-parametric tests offer **exact** $p$-values computed directly from the permutation distribution of the test statistic, eliminating the need for any large-sample approximation.

## What Makes a Test Non-Parametric

A statistical test is called **non-parametric** (or **distribution-free**) if its validity does not depend on the assumption that the data come from a specific parametric family of distributions. More precisely, under the null hypothesis $H_0$, the distribution of the test statistic is the same for all continuous distributions.

!!! note "Distribution-free under the null"
    The term "distribution-free" refers to the null distribution of the test statistic, not to the data themselves. Non-parametric tests still make *some* assumptions -- typically that observations are independent and identically distributed from a continuous distribution.

The most common mechanism for achieving distribution-freeness is the **rank transformation**: replace each observation $X_i$ by its rank $R_i$ among the combined sample. Under the null hypothesis (e.g., that two groups share the same distribution), every permutation of the ranks is equally likely, so the null distribution of any rank-based statistic can be derived without knowing the shape of the population.

## Advantages of Non-Parametric Tests

| Advantage | Explanation |
|:----------|:------------|
| Fewer assumptions | Valid without requiring normality or equal variances |
| Robustness to outliers | Rank transformation limits the influence of extreme values |
| Applicability to ordinal data | Meaningful when only the ordering of observations is available |
| Exact $p$-values for small samples | Permutation-based null distributions avoid reliance on asymptotics |
| Broadly applicable | Can test hypotheses about medians, distributions, or stochastic ordering |

## Disadvantages and Trade-offs

Non-parametric methods are not without cost. The primary trade-off is **statistical power**: when the parametric assumptions are satisfied, a non-parametric test will generally require a larger sample to achieve the same power as its parametric counterpart. This efficiency loss is quantified by the **asymptotic relative efficiency** (ARE), discussed in detail in the [Power Comparison](power_comparison.md) section.

| Limitation | Explanation |
|:-----------|:------------|
| Lower power under normality | Discarding magnitude information (by ranking) loses some efficiency |
| Less familiar confidence intervals | Constructing CIs from rank-based tests is less straightforward than from $t$-based methods |
| Ties complicate exact tests | When observations share the same value, exact permutation distributions require adjustment |
| Fewer diagnostic tools | Residual plots and influence diagnostics are less developed for rank-based tests |

## Decision Framework

Choosing between parametric and non-parametric tests is not an all-or-nothing decision. The following guidelines help navigate the choice:

1. **Check normality first.** Use graphical methods (QQ-plots, histograms) and formal tests (Shapiro-Wilk, Anderson-Darling) from [Chapter 14](../../ch14/index.md) to assess whether the normality assumption is reasonable.
2. **Consider the sample size.** For large samples ($n > 30$), the central limit theorem often makes parametric tests robust to moderate departures from normality. For small samples, non-parametric tests offer safer inference.
3. **Assess the measurement scale.** If the data are ordinal, a rank-based test is the natural choice regardless of sample size.
4. **Evaluate the impact of outliers.** If the data contain extreme values that cannot be removed on substantive grounds, non-parametric tests protect against their distorting influence.
5. **Weigh power against robustness.** When the assumptions hold, parametric tests are more powerful. When they do not, non-parametric tests can be substantially more powerful because the parametric test's actual Type I error rate may differ from the nominal level.

??? example "When to choose non-parametric over parametric"
    **Scenario:** A researcher collects pain scores (1--10 Likert scale) from 12 patients in a treatment group and 10 in a control group. The distribution of scores is heavily right-skewed with a floor effect at 1.

    - The data are **ordinal**, so the mean is not a meaningful summary.
    - The sample is **small** ($n_1 = 12$, $n_2 = 10$), so the CLT provides a weak guarantee.
    - The distribution is **skewed**, violating the normality assumption.

    A Mann-Whitney $U$ test (or equivalently, the Wilcoxon rank-sum test) is the appropriate choice here. It tests whether one group tends to produce systematically higher scores than the other, without assuming anything about the shape of the distribution.

## Summary

Non-parametric tests trade a modest amount of power under ideal conditions for broad applicability and robustness. They are indispensable when the data are ordinal, when samples are small and non-normal, or when outliers threaten the validity of parametric inference. The remainder of this chapter develops the most widely used non-parametric procedures for one-sample, paired-sample, two-sample, and multi-group comparisons.
