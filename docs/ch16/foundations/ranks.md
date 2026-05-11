# Ranks and Rank Transformations

Most non-parametric tests achieve their distribution-free property by replacing raw observations with their **ranks** -- the positions the observations occupy when sorted from smallest to largest. This simple transformation discards the specific numerical values while preserving the ordering, which is exactly the information needed to detect location shifts, stochastic dominance, and monotonic association.

This section defines the rank transformation formally, explains how to handle tied observations using **midranks**, and introduces the key properties that make rank statistics the foundation of non-parametric inference.

## Definition of Ranks

Given a sample $X_1, X_2, \ldots, X_n$, the **rank** of observation $X_i$ is the position of $X_i$ in the ordered sample. Formally, let $X_{(1)} \le X_{(2)} \le \cdots \le X_{(n)}$ denote the order statistics. If all values are distinct, the rank of $X_i$ is the unique integer $R_i$ such that

$$
X_i = X_{(R_i)}
$$

Equivalently,

$$
R_i = \sum_{j=1}^{n} \mathbf{1}(X_j \le X_i)
$$

where $\mathbf{1}(\cdot)$ is the indicator function.

??? example "Ranking a small sample"
    Consider the sample $X = (7.3, \; 2.1, \; 5.8, \; 9.0, \; 4.5)$. Sorting gives the order statistics $(2.1, \; 4.5, \; 5.8, \; 7.3, \; 9.0)$. The ranks are:

    | $i$ | $X_i$ | $R_i$ |
    |:---:|:-----:|:-----:|
    | 1 | 7.3 | 4 |
    | 2 | 2.1 | 1 |
    | 3 | 5.8 | 3 |
    | 4 | 9.0 | 5 |
    | 5 | 4.5 | 2 |

## Handling Ties with Midranks

When two or more observations share the same value, assigning a unique integer rank to each is ambiguous. The standard resolution is to assign each tied observation the **midrank** (also called the **average rank**): the arithmetic mean of the ranks that the tied values would occupy.

If observations at positions $j, j+1, \ldots, j+k-1$ in the sorted sample are all equal, each receives the midrank

$$
\bar{R} = \frac{1}{k} \sum_{l=0}^{k-1} (j + l) = j + \frac{k - 1}{2}
$$

??? example "Midranks with tied values"
    Consider $X = (4, \; 7, \; 7, \; 7, \; 10)$. The sorted values occupy positions 1 through 5. The three 7s occupy positions 2, 3, and 4, so each receives midrank $(2 + 3 + 4)/3 = 3$:

    | $i$ | $X_i$ | Rank position(s) | Midrank $R_i$ |
    |:---:|:-----:|:-----------------:|:-------------:|
    | 1 | 4 | 1 | 1 |
    | 2 | 7 | 2, 3, 4 | 3 |
    | 3 | 7 | 2, 3, 4 | 3 |
    | 4 | 7 | 2, 3, 4 | 3 |
    | 5 | 10 | 5 | 5 |

    Notice that the sum of midranks still equals $1 + 3 + 3 + 3 + 5 = 15 = n(n+1)/2$, a property that always holds.

!!! warning "Effect of ties on test statistics"
    Many non-parametric test statistics assume no ties when deriving their null distributions. When ties are present, a **tie correction factor** must be applied. For example, the Kruskal-Wallis and Wilcoxon rank-sum tests divide by a correction term that depends on the number and size of tied groups. Ignoring ties inflates the variance of the test statistic, making the test conservative.

## Properties of Ranks

Several properties make ranks especially useful for distribution-free inference.

**Sum of ranks.** For any sample of size $n$, the ranks $R_1, R_2, \ldots, R_n$ are a permutation of $\{1, 2, \ldots, n\}$ (or midranks summing to the same total), so

$$
\sum_{i=1}^{n} R_i = \frac{n(n+1)}{2}
$$

**Mean rank.** The average rank is always

$$
\bar{R} = \frac{n+1}{2}
$$

regardless of the shape of the underlying distribution.

**Variance of ranks.** When there are no ties, the variance of the ranks is

$$
\text{Var}(R) = \frac{1}{n} \sum_{i=1}^{n} \left(i - \frac{n+1}{2}\right)^2 = \frac{n^2 - 1}{12}
$$

**Distribution-free property.** Under the null hypothesis that all observations come from the same continuous distribution, every permutation of the ranks is equally likely. This means the null distribution of any statistic that depends only on the ranks can be computed exactly by enumeration, without knowing the population distribution.

## The Rank Transformation in Practice

The rank transformation converts an arbitrary continuous distribution into a discrete uniform distribution on $\{1, 2, \ldots, n\}$. This has two important consequences:

1. **Outlier resistance.** An extreme observation receives rank $n$ regardless of whether it is 10 or 10,000. The rank transformation bounds the influence of every observation.
2. **Scale invariance.** Ranks are invariant under any monotone increasing transformation of the data. If we apply a log or square-root transformation before ranking, the ranks do not change.

!!! tip "Ranks as a normalizing transformation"
    For moderately non-normal data, replacing observations with their ranks (or with the normal scores $\mathcal{N}^{-1}(R_i / (n+1))$, known as the **van der Waerden transformation**) can serve as a practical normalization step. The resulting rank-transformed data can then be analyzed with standard ANOVA or regression methods, yielding tests with good robustness properties.

## Tie Correction Factor

When ties are present, the exact null distributions of rank-based statistics change. Most test statistics incorporate a correction factor. Let $g$ denote the number of distinct tied groups, and let $t_j$ be the number of tied observations in the $j$-th group. The commonly used correction factor is

$$
C = 1 - \frac{\sum_{j=1}^{g} (t_j^3 - t_j)}{n^3 - n}
$$

The corrected test statistic is obtained by dividing the uncorrected statistic by $C$ (or equivalently, dividing its variance by $C$). When there are no ties, $t_j = 1$ for all $j$, so $C = 1$ and no correction is needed.

## Common Rank-Based Statistics

The following rank-based test statistics appear throughout this chapter:

| Statistic | Definition | Used in |
|:----------|:-----------|:--------|
| Wilcoxon $W^+$ | Sum of ranks of positive differences | Signed-rank test |
| Wilcoxon $W$ | Sum of ranks in one group | Rank-sum test |
| Mann-Whitney $U$ | Count of pairwise wins | Two-sample test |
| Kruskal-Wallis $H$ | Between-group rank variance | Multi-group test |
| Friedman $\chi^2_F$ | Within-block rank variance | Repeated measures |
| Spearman $r_s$ | Pearson correlation of ranks | Correlation |
| Kendall $\tau$ | Normalized concordance count | Correlation |

Each of these statistics operates on the ranks rather than the raw data, inheriting the distribution-free property and outlier resistance described above.

## Summary

The rank transformation is the central mechanism behind non-parametric testing. By replacing observations with their positions in the sorted sample, rank-based methods achieve distribution-freeness, robustness to outliers, and applicability to ordinal data. When ties occur, midranks preserve the key properties of the rank transformation, and correction factors ensure that test statistics maintain their nominal significance levels. The specific rank-based procedures built on this foundation are developed in the sections that follow.


## Exercises

**Exercise 1.**
Describe the main concept of Ranks and Rank Transformations and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Ranks and Rank Transformations is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
