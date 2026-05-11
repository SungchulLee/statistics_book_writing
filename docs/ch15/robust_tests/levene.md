# Levene's Test

The chi-square test, F-test, and Bartlett's test all assume normality and break down when the data are non-normal. Levene (1960) proposed an elegantly simple idea: instead of testing variances directly, transform the data into absolute deviations from the group mean and then apply a standard one-way ANOVA to the transformed values. This conversion turns a variance-comparison problem into a mean-comparison problem, which is far less sensitive to the shape of the underlying distribution.

## The Key Idea

If a group has large variance, its observations tend to be far from the group center. If a group has small variance, its observations cluster close to the center. Levene's test formalizes this intuition by defining new variables

$$
Z_{ij} = |X_{ij} - \bar{X}_i|
$$

where $X_{ij}$ is the $j$-th observation in group $i$ and $\bar{X}_i$ is the mean of group $i$. The value $Z_{ij}$ measures how far each observation falls from its group mean. Groups with larger variances produce larger average $Z_{ij}$ values.

Testing $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2$ is then equivalent to testing whether the means of $Z_{ij}$ are equal across groups.

## Test Statistic

Apply the standard one-way ANOVA F-statistic to the transformed values $Z_{ij}$:

$$
W = \frac{(N - k) \sum_{i=1}^{k} n_i (\bar{Z}_i - \bar{Z})^2}{(k - 1) \sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_i)^2}
$$

where:

- $\bar{Z}_i = \frac{1}{n_i} \sum_{j=1}^{n_i} Z_{ij}$ is the mean of the transformed values in group $i$
- $\bar{Z} = \frac{1}{N} \sum_{i=1}^{k} \sum_{j=1}^{n_i} Z_{ij}$ is the overall mean of all transformed values
- $N = \sum_{i=1}^{k} n_i$ is the total sample size
- $k$ is the number of groups

Under $H_0$ and mild regularity conditions, $W$ is approximately distributed as $F_{k-1, N-k}$.

## Hypotheses

$$
H_0\colon \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2
$$

$$
H_1\colon \sigma_i^2 \neq \sigma_j^2 \text{ for at least one pair } i \neq j
$$

Reject $H_0$ if $W > F_{1-\alpha,\, k-1,\, N-k}$.

## Why Levene's Test Is Robust

The robustness comes from two sources:

1. **Absolute deviations are less sensitive to outliers than squared deviations.** The sample variance uses $(X_{ij} - \bar{X}_i)^2$, which gives extreme observations quadratic influence. Absolute deviations $|X_{ij} - \bar{X}_i|$ give them only linear influence.

2. **The ANOVA F-test on means is robust.** By the central limit theorem, the group means $\bar{Z}_i$ become approximately normal for moderate sample sizes, even when the $Z_{ij}$ themselves are not normally distributed. The F-test for means inherits this robustness.

!!! note "Original vs. Modified Levene's Test"
    Levene's original 1960 proposal uses the group mean $\bar{X}_i$ as the center. The Brown-Forsythe modification (covered in the next section) replaces the mean with the group median, providing additional robustness to skewness and outliers. When authors refer to "Levene's test" without qualification, they sometimes mean the Brown-Forsythe version; check the documentation of the software being used.

## Example

Three groups have the following observations:

| Group 1 | Group 2 | Group 3 |
|---|---|---|
| 10, 12, 14, 11, 13 | 20, 28, 22, 35, 25 | 15, 16, 14, 17, 15 |

**Step 1.** Compute group means:

- $\bar{X}_1 = 12.0$, $\bar{X}_2 = 26.0$, $\bar{X}_3 = 15.4$

**Step 2.** Compute absolute deviations $Z_{ij} = |X_{ij} - \bar{X}_i|$:

| Group 1 | Group 2 | Group 3 |
|---|---|---|
| 2, 0, 2, 1, 1 | 6, 2, 4, 9, 1 | 0.4, 0.6, 1.4, 1.6, 0.4 |

**Step 3.** Compute means of transformed values:

- $\bar{Z}_1 = 1.20$, $\bar{Z}_2 = 4.40$, $\bar{Z}_3 = 0.88$
- $\bar{Z} = (1.20 + 4.40 + 0.88) \times 5/15 = 2.16$ (weighted by equal group sizes)

**Step 4.** Compute the $W$ statistic using the ANOVA formula on the $Z_{ij}$ values. With $k = 3$ groups and $N = 15$ observations, $W$ follows an $F_{2, 12}$ distribution under $H_0$.

**Step 5.** Compare $W$ to the critical value $F_{0.95,\, 2,\, 12} = 3.885$. Group 2 has much larger deviations than the other groups, which will produce a large $W$ value, likely leading to rejection of $H_0$.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Group data
group1 = [10, 12, 14, 11, 13]
group2 = [20, 28, 22, 35, 25]
group3 = [15, 16, 14, 17, 15]

# Levene's test using the mean (original Levene)
stat, p_value = stats.levene(group1, group2, group3, center='mean')
print(f"Levene's W statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: variances are significantly different.")
else:
    print("Fail to reject H0: no significant difference in variances.")
```

## Strengths and Limitations

**Strengths:**

- Robust to moderate departures from normality
- Simple to compute (just an ANOVA on absolute deviations)
- Available in all major statistical software packages
- Works well for symmetric non-normal distributions

**Limitations:**

- Still uses the group mean, which is sensitive to outliers and skewness. The Brown-Forsythe modification addresses this by using the median.
- The $F$-distribution approximation is asymptotic; very small samples may show some size distortion.
- Less powerful than Bartlett's test when the data are truly normal.

## Exercises

**Exercise 1.**
Three different teaching methods are applied to three groups of students. After the semester, the students' scores are recorded as follows:

- **Group 1:** $[78, 82, 85, 90, 87]$
- **Group 2:** $[65, 70, 72, 68, 74]$
- **Group 3:** $[92, 88, 94, 89, 91]$

Use Levene's test to determine if the variances in the test scores are equal across the three groups.

??? success "Solution to Exercise 1"

    **Hypotheses:**

    - Null Hypothesis ($H_0$): The variances are equal across the three groups.
    - Alternative Hypothesis ($H_1$): At least one group has a variance that differs from the others.

    **Test Statistic:**

    Levene's test calculates the absolute deviations from the group medians and tests whether the variance of these deviations differs across groups.

    **Python Implementation:**

    ```python
    from scipy.stats import levene

    # Test scores for the three groups
    group1 = [78, 82, 85, 90, 87]
    group2 = [65, 70, 72, 68, 74]
    group3 = [92, 88, 94, 89, 91]

    # Perform Levene's test
    statistic, p_value = levene(group1, group2, group3)

    print(f"Levene's test statistic: {statistic}")
    print(f"P-value: {p_value}")
    ```

    **Interpretation:**

    If the p-value is less than $0.05$, reject the null hypothesis and conclude that the variances are not equal across the groups.

---

**Exercise 2.**
Samples were drawn from two populations and analyzed. A Levene's test was performed, resulting in a high p-value (failure to reject $H_0$: equal variances). However, a t-test assuming equal variances yielded a p-value smaller than 0.001. How should these two test results be interpreted?

??? success "Solution to Exercise 2"

    - The Levene's test supports the assumption that the **variances are equal** across the two populations.
    - The t-test result provides **strong evidence** that the **means** of the two populations are significantly different.
    - These results are not contradictory — two populations can have equal variances while having very different means. The Levene's test validates the equal-variance assumption used in the t-test, which strengthens the conclusion that the observed mean difference is genuine.
