# Welch's One-Way ANOVA

**Welch's ANOVA** (Analysis of Variance) is a statistical test used to determine whether the means of two or more groups are significantly different when the assumption of **equal variances** (homoscedasticity) is violated. It is an alternative to the traditional one-way ANOVA, which assumes equal variances among groups.

## 1. When to Use Welch's ANOVA

- When comparing the means of two or more groups.
- When group variances are unequal (**heteroscedasticity**).
- When the assumption of normality is approximately satisfied for each group (or sample sizes are large enough for the Central Limit Theorem to apply).

## 2. Assumptions

1. The data in each group are **independently** and **randomly sampled**.
2. Each group is approximately **normally distributed** (or the sample size is large enough for the Central Limit Theorem to apply).
3. Group variances are **not assumed to be equal**.

## 3. Hypotheses

- **Null Hypothesis ($H_0$)**: The group means are equal.

$$ H_0: \mu_1 = \mu_2 = \cdots = \mu_k $$

- **Alternative Hypothesis ($H_a$)**: At least one group mean is different from the others.

## 4. Test Statistic

Welch's ANOVA computes a test statistic $F$ using a weighted approach:

$$ F = \frac{\sum_{i=1}^k w_i (\bar{X}_i - \bar{X})^2 / (k-1)}{\sum_{i=1}^k \frac{1 - w_i}{n_i - 1} / (k - \frac{1}{\sum_{i=1}^k \frac{1}{n_i - 1}})} $$

Where:

- $\bar{X}_i$: Mean of the $i$-th group.
- $w_i = \frac{n_i}{s_i^2}$: Weight of the $i$-th group.
- $s_i^2$: Variance of the $i$-th group.
- $n_i$: Sample size of the $i$-th group.
- $\bar{X}$: Weighted mean across all groups.

The degrees of freedom are calculated using the **Welch-Satterthwaite equation**, leading to an $F$-statistic that is compared against the $F$-distribution.

## 5. Welch-Satterthwaite Equation for Computing Degree of Freedom

The **Welch-Satterthwaite equation** provides an approximation for the **degrees of freedom** (df) when comparing group means under unequal variances. This adjusted degrees of freedom is essential for ensuring that the test statistic follows the correct distribution.

The formula for the effective degrees of freedom is:

$$
\text{df} = \frac{\left( \sum_{i=1}^k \frac{w_i}{n_i} \right)^2}{\sum_{i=1}^k \frac{\left( \frac{w_i}{n_i} \right)^2}{n_i - 1}}
$$

**Components of the Formula:**

1. **Weights ($w_i$)**: $w_i = \frac{n_i}{s_i^2}$, where $n_i$ is the sample size and $s_i^2$ is the variance of group $i$. The weights adjust for the inverse variance of each group, giving more influence to groups with lower variances.

2. **Numerator**: The sum of the weights scaled by their respective sample sizes is squared, accounting for the overall influence of all groups.

3. **Denominator**: The variability within each group is divided by its degrees of freedom ($n_i - 1$) and summed.

**Purpose**: The equation accounts for differences in variances and sample sizes, providing an effective degrees of freedom that is not necessarily an integer but is crucial for comparing the F-statistic against the critical value of an F-distribution.

## 6. Steps for Performing Welch's ANOVA

1. **State the Hypotheses**: $H_0$: All group means are equal. $H_a$: At least one group mean is different.
2. **Check Assumptions**: Independence, normality, and heteroscedasticity (no need for equal variances).
3. **Compute the Test Statistic**: Calculate $F$ and its associated degrees of freedom.
4. **Determine the p-value**: Compare the $F$-statistic to the $F$-distribution with the calculated degrees of freedom.
5. **Make a Decision**: If $p \leq \alpha$ (e.g., 0.05), reject $H_0$.
6. **Post Hoc Analysis**: If $H_0$ is rejected, conduct post hoc tests (e.g., Games-Howell test) to identify which groups differ.

## 7. Python Implementation

### Example: Welch's ANOVA with Unequal Variances

```python
import pingouin as pg
import pandas as pd

# Sample data
data = {
    "Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Values": [12, 14, 13, 22, 23, 19, 31, 33, 29, 35],
}
df = pd.DataFrame(data)

# Perform Welch's ANOVA
anova_results = pg.welch_anova(dv="Values", between="Group", data=df)

# Display the results
print(anova_results)
```

Output:
```
         Source  ddof1    ddof2         F         p-unc
0  Group       2.000  5.654  45.236  0.00012
```

- $F$: Test statistic.
- $p$: p-value indicating whether to reject $H_0$.

### Post Hoc Testing

If Welch's ANOVA finds significant differences, use a post hoc test such as the **Games-Howell test**, which does not assume equal variances or equal sample sizes:

```python
# Perform Games-Howell post hoc test
post_hoc = pg.pairwise_gameshowell(dv="Values", between="Group", data=df)
print(post_hoc)
```

## 8. Advantages

1. **No equal variance assumption**: Handles unequal variances (heteroscedasticity).
2. **Robust to unbalanced designs**: Can be applied when group sizes are unequal.
3. **More accurate** than traditional one-way ANOVA when variances differ.

## 9. Limitations

1. Requires the assumption of normality for accurate results.
2. May be less powerful than traditional ANOVA when variances are actually equal.
3. More complex to compute manually.

## 10. Summary

Welch's ANOVA is an extension of one-way ANOVA designed to compare means when the assumption of equal variances is violated. It is robust, flexible, and essential for analyzing data with heteroscedasticity. Post hoc tests like the Games-Howell test are recommended to identify which groups differ.
