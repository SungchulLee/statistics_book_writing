# Non-Parametric Methods

Another strategy for dealing with non-normal data is to use **non-parametric methods**. These methods do not assume a specific distribution for the data and are often used when data is ordinal or when normality assumptions are violated.

## Common Non-Parametric Tests

- **Mann-Whitney U Test**: A non-parametric alternative to the $t$-test for comparing two independent groups.
- **Kruskal-Wallis Test**: A non-parametric alternative to ANOVA for comparing more than two groups.
- **Wilcoxon Signed-Rank Test**: A non-parametric test for comparing two related samples.

## Python Implementation

```python
import numpy as np
from scipy.stats import mannwhitneyu

# Generate two non-normal datasets
group1 = np.random.exponential(scale=2, size=100)
group2 = np.random.exponential(scale=3, size=100)

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(group1, group2)
print(f"Mann-Whitney U Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value > alpha:
    print("Fail to reject H_0: No significant difference between the groups.")
else:
    print("Reject H_0: Significant difference between the groups.")
```

Non-parametric tests offer robust alternatives when normality assumptions are violated or when dealing with ordinal data. They are widely used in situations where data distributions are unknown or non-normal.

## Summary: Dealing with Non-Normal Data

When faced with non-normal data, several strategies can be employed:

- **Transformations** can help bring the data closer to normality.
- **Bootstrapping** provides an alternative approach that does not rely on parametric assumptions.
- **Non-parametric methods** offer powerful alternatives to traditional parametric tests when assumptions are violated.

The choice of method depends on the degree of non-normality, the sample size, and the specific research questions being addressed. In practice, combining these approaches with graphical and formal normality assessments can lead to more reliable statistical analysis.
