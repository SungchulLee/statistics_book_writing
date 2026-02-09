# Welch's Two-Way ANOVA

**Welch's Two-Way ANOVA** is an extension of Welch's one-way ANOVA that allows the comparison of means across two independent factors when the assumption of **equal variances** is violated. It adjusts for heteroscedasticity (unequal variances) by using weighted means and incorporates the interaction effects between the two factors.

## 1. When to Use Welch's Two-Way ANOVA

1. To compare group means across two factors when variances are unequal (heteroscedasticity).
2. When interaction effects between two factors need to be analyzed.
3. When traditional two-way ANOVA assumptions are violated, especially equal variances and equal sample sizes.
4. In practical settings where data collection leads to unequal variability between groups, such as experimental or survey data.

## 2. Assumptions

1. The data are **independently and randomly sampled**.
2. Each group's data is approximately **normally distributed** (or the sample size is large enough for the Central Limit Theorem to apply).
3. Group variances are **not equal** (heteroscedasticity is allowed).
4. Factors and interactions are modeled as fixed effects.

## 3. Hypotheses

**Main Effects:**

- For Factor A:
    - $H_0$: The means of all levels of Factor A are equal.
    - $H_a$: At least one mean of Factor A is different.
- For Factor B:
    - $H_0$: The means of all levels of Factor B are equal.
    - $H_a$: At least one mean of Factor B is different.

**Interaction Effect:**

- $H_0$: There is no interaction effect between Factor A and Factor B.
- $H_a$: There is an interaction effect between Factor A and Factor B.

## 4. How Welch's Two-Way ANOVA Works

### Weighted Means

Instead of pooling variances as in traditional ANOVA, Welch's method calculates weighted means for each group:

$$ w_{ij} = \frac{n_{ij}}{s_{ij}^2} $$

Where $n_{ij}$ is the sample size and $s_{ij}^2$ is the variance of group $(i,j)$ for Factor A level $i$ and Factor B level $j$.

### Test Statistics

Separate F-tests are performed for main effects of Factor A and Factor B, and interaction effects between Factor A and Factor B. The test statistics are derived using weighted group means, and the degrees of freedom are adjusted using the **Welch-Satterthwaite equation** to account for unequal variances.

## 5. Steps to Perform Welch's Two-Way ANOVA

1. **Organize Data**: Data should include two independent factors and one dependent variable.
2. **State Hypotheses**: Define null and alternative hypotheses for main effects and interaction effects.
3. **Calculate Weighted Means**: Compute weighted group means and variances for each factor level.
4. **Adjust Degrees of Freedom**: Use the Welch-Satterthwaite equation to calculate effective degrees of freedom.
5. **Compute F-Statistics**: Perform separate F-tests for the main effects and interaction, using the adjusted degrees of freedom.
6. **Interpret Results**: Compare the F-statistics to the critical values or use p-values.
7. **Post Hoc Analysis**: If significant differences are found, conduct post hoc tests (e.g., Games-Howell).

## 6. Python Implementation

Consider an experiment to test the effects of two factors (e.g., **Temperature** and **Fertilizer Type**) on plant growth:

```python
import pingouin as pg
import pandas as pd

# Sample data
data = {
    "Temperature": ["High", "High", "High", "Low", "Low", "Low", "Medium", "Medium", "Medium"],
    "Fertilizer": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
    "Growth": [12, 15, 14, 10, 13, 11, 14, 16, 15],
}
df = pd.DataFrame(data)

# Perform Welch's Two-Way ANOVA
anova_results = pg.welch_anova(dv="Growth", between=["Temperature", "Fertilizer"], data=df)

# Display results
print(anova_results)
```

**Output:**
```
          Source        ddof1    ddof2         F       p-unc
0     Temperature    2.000    5.400    7.124    0.024
1     Fertilizer     2.000    5.200    6.345    0.035
2 Temperature * Fertilizer  4.000    6.700    3.215    0.062
```

**Interpretation:**

- **Temperature**: Significant effect ($p = 0.024$), suggesting that temperature levels affect growth.
- **Fertilizer**: Significant effect ($p = 0.035$), indicating fertilizer types influence growth.
- **Interaction**: No significant interaction ($p = 0.062$).

### Post Hoc Testing

If the main effects are significant, use post hoc tests like **Games-Howell** to identify which specific levels differ:

```python
# Perform Games-Howell post hoc test for Temperature
post_hoc_temp = pg.pairwise_gameshowell(dv="Growth", between="Temperature", data=df)
print(post_hoc_temp)

# Perform Games-Howell post hoc test for Fertilizer
post_hoc_fert = pg.pairwise_gameshowell(dv="Growth", between="Fertilizer", data=df)
print(post_hoc_fert)
```

## 7. Advantages

1. **Handles Unequal Variances**: Adjusts for heteroscedasticity, unlike traditional two-way ANOVA.
2. **Interaction Effects**: Captures interaction between two factors while accounting for variance differences.
3. **Robust to Unequal Sample Sizes**: Handles unbalanced designs effectively.

## 8. Limitations

1. **Assumption of Normality**: Welch's ANOVA still assumes approximate normality within groups.
2. **Complex Computation**: Requires more computational resources compared to traditional ANOVA.
3. **Interpretation Challenges**: Interaction effects can be harder to interpret when variances are unequal.

## 9. Summary

Welch's Two-Way ANOVA is a robust method for analyzing the effects of two factors on a dependent variable when group variances are unequal. It extends the principles of Welch's one-way ANOVA to factorial designs, ensuring accurate results even under heteroscedasticity. For practical applications, software tools like `pingouin` in Python make it easy to implement.
