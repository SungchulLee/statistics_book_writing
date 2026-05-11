# Levene Test

## Overview

Levene's test assesses the null hypothesis that two or more populations share the same variance. Unlike Bartlett's test and the F-test for equality of variances, Levene's test is robust to departures from normality, making it the preferred choice for checking homoscedasticity in practice. This page presents the test statistic, discusses the choice of centering function, and demonstrates the test across several variance-ratio scenarios using `scipy.stats.levene`.

## Hypotheses and Test Statistic

Consider $k$ groups with sample sizes $n_1, \dots, n_k$ and total sample size $N = \sum_{i=1}^{k} n_i$. The hypotheses are

$$
H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2, \qquad H_1: \text{not all variances are equal}
$$

Define the transformed variable

$$
Z_{ij} = |y_{ij} - \tilde{y}_i|
$$

where $\tilde{y}_i$ is a measure of central tendency for group $i$ (typically the group median). Levene's test statistic is then

$$
W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}
$$

where $\bar{Z}_{i\cdot}$ is the mean of $Z_{ij}$ within group $i$ and $\bar{Z}_{\cdot\cdot}$ is the overall mean of all $Z_{ij}$. Under $H_0$, the statistic $W$ approximately follows an $F(k-1,\, N-k)$ distribution.

## Choice of Center

The centering function determines the robustness and power of the test:

| Center | Notation | Properties |
|---|---|---|
| Mean | $\bar{y}_i$ | Original Levene (1960); most powerful under normality but sensitive to outliers |
| Median | $\tilde{y}_i$ | Brown-Forsythe variant; robust to skewness and outliers |
| Trimmed mean | $\bar{y}_i^{(\text{trim})}$ | Compromise between power and robustness |

In `scipy.stats.levene`, the `center` parameter controls this choice. The default is `'median'`, which is the Brown-Forsythe variant:

```python
from scipy.stats import levene

stat, pval = levene(group1, group2, center='median')
```

## Demonstration

The accompanying script generates $X \sim N(0, 1)$ and $Y \sim N(1, \sigma_Y)$ for $\sigma_Y \in \{1.00, 1.05, 1.10, 1.15, 1.20\}$, then applies Levene's test to each pair:

```python
import numpy as np
import scipy.stats as stats

seed, size = 1, 100
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = stats.levene(x, y)
    print(f"sigma_y={scale:.2f}: F={stat:.2f}, p={pval:.3f}")
```

## Interpretation

- When $\sigma_Y = 1.00$, the variances are equal and the test produces a small $W$ with a large p-value, correctly retaining $H_0$.
- As $\sigma_Y$ increases, the absolute deviations in the $Y$ group grow relative to those in the $X$ group, increasing $W$ and decreasing the p-value.
- Compared to Bartlett's test on the same data, Levene's test is slightly less powerful under exact normality but maintains correct Type I error rates under non-normality.

Levene's test is the standard pre-check before running a classical ANOVA. When it rejects, the analyst should switch to Welch's ANOVA or another heteroscedasticity-robust procedure.

## Exercises

**Exercise 1.**
For two groups with $n_1 = n_2 = 50$, explain intuitively why Levene's test statistic $W$ is essentially a one-way ANOVA F-statistic applied to the transformed data $Z_{ij}$.

??? success "Solution to Exercise 1"
    Levene's test replaces each observation $y_{ij}$ with its absolute deviation from the group center, $Z_{ij} = |y_{ij} - \tilde{y}_i|$. If the group variances are equal, these absolute deviations should have similar means across groups. If one group has larger variance, its absolute deviations will be systematically larger, producing a higher group mean $\bar{Z}_{i\cdot}$.

    The test statistic $W$ measures the ratio of between-group variation to within-group variation in the $Z_{ij}$ values, which is exactly the one-way ANOVA F-statistic applied to the transformed data. A large $W$ means the group means of the absolute deviations differ more than expected by chance, providing evidence against equal variances.

---

**Exercise 2.**
A dataset has three groups with sample sizes $(15, 15, 15)$ and the data are approximately normal. Would you recommend Levene's test or Bartlett's test? What if the sample sizes were $(15, 15, 200)$?

??? success "Solution to Exercise 2"
    For the balanced case with approximate normality, Bartlett's test is slightly more powerful because it is the uniformly most powerful test under exact normality. However, Levene's test would also perform well and is the safer default.

    For the unbalanced case $(15, 15, 200)$, Levene's test is preferred even under normality. Bartlett's test can behave erratically with highly unbalanced designs because the pooled variance $S_p^2$ is dominated by the large group, and small departures from normality in any group can inflate the test statistic. Levene's median-based approach is more robust to both the imbalance and potential distributional issues.

---

**Exercise 3.**
Prove that if all observations in every group have the same value (i.e., zero within-group variance), then $W = 0$.

??? success "Solution to Exercise 3"
    If every observation in group $i$ equals the same value $c_i$, then the group median is $\tilde{y}_i = c_i$ and

    $$
    Z_{ij} = |y_{ij} - \tilde{y}_i| = |c_i - c_i| = 0
    $$

    for all $i, j$. Therefore $\bar{Z}_{i\cdot} = 0$ for every group and $\bar{Z}_{\cdot\cdot} = 0$. The numerator of $W$ becomes

    $$
    \sum_{i=1}^{k} n_i (0 - 0)^2 = 0
    $$

    so $W = 0$ regardless of the denominator (which is also zero, but the convention is that no variability yields no evidence against $H_0$). $\square$

---

**Exercise 4.**
Suppose Levene's test gives $p = 0.03$ for a three-group comparison. The researcher proceeds with a classical one-way ANOVA and finds $p = 0.04$ for the group means. Critique this approach and suggest an alternative.

??? success "Solution to Exercise 4"
    The researcher has identified a violation of the equal-variance assumption (Levene's $p = 0.03 < 0.05$) but then used a procedure that requires that very assumption. The classical ANOVA F-test is unreliable when variances are unequal, especially with unbalanced designs: the actual Type I error rate can be substantially higher or lower than the nominal $\alpha$.

    The correct approach is to use Welch's ANOVA (`scipy.stats.alexandergovern` or a Welch-corrected F-test), which does not assume equal variances. For post-hoc comparisons, Games-Howell should replace Tukey HSD, as it also accounts for unequal variances.

---

**Exercise 5.**
Derive the approximate distribution of $W$ under $H_0$ by arguing from the properties of the one-way ANOVA F-statistic applied to the $Z_{ij}$ values.

??? success "Solution to Exercise 5"
    Under $H_0: \sigma_1^2 = \cdots = \sigma_k^2$, the absolute deviations $Z_{ij} = |y_{ij} - \tilde{y}_i|$ have the same expected value across all groups (since the spread of each group is identical). The $Z_{ij}$ values are not exactly normal, but for moderate to large sample sizes their group means $\bar{Z}_{i\cdot}$ are approximately normal by the Central Limit Theorem.

    The statistic $W$ is the standard one-way ANOVA F-statistic computed on the $Z_{ij}$ values. Under the null hypothesis (equal means of the $Z_{ij}$ across groups), the ANOVA F-statistic follows $F(k-1, N-k)$ approximately. The approximation improves with sample size. The key insight is that even though the $Z_{ij}$ are not normal (they are non-negative by construction), the ratio of mean squares converges to the F-distribution as long as the group sample sizes are not too small.
