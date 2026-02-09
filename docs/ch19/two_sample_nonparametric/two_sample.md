# Two-Sample Non-Parametric Tests

## 1. Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

The **Mann-Whitney U test**, also known as the **Wilcoxon rank-sum test**, is a non-parametric statistical test used to compare the distributions of two independent groups. It is particularly useful when the assumptions of a parametric test, such as the independent two-sample t-test, are not met (e.g., non-normality or ordinal data).

### Key Features

- **Purpose**: Tests whether the distributions of two independent groups are the same, or whether one group tends to have larger (or smaller) values than the other.
- **Assumptions**:
    - The two groups are independent of each other.
    - The observations are ordinal, interval, or ratio (but do not require normality).
    - The two samples are randomly drawn.
- **Null Hypothesis ($H_0$)**: The two groups have the same distribution (no difference in medians or ranks).
- **Alternative Hypothesis ($H_1$)**: The distributions are different, or one group tends to have higher values than the other.

### How the Test Works

**Step 1:** Combine all data from both groups and assign **ranks** to the values. The smallest value gets rank 1, the second smallest rank 2, and so on. If there are ties, assign the average of the tied ranks.

**Step 2:** Calculate the sum of ranks for each group:

- $R_1$: Sum of ranks for group 1
- $R_2$: Sum of ranks for group 2

**Step 3:** Compute the **U-statistic** for each group:

$$U_1 = n_1 n_2 + \frac{n_1 (n_1 + 1)}{2} - R_1$$

$$U_2 = n_1 n_2 + \frac{n_2 (n_2 + 1)}{2} - R_2$$

where $n_1$ and $n_2$ are the sample sizes for groups 1 and 2.

**Step 4:** The Mann-Whitney U statistic is:

$$U = \min(U_1, U_2)$$

**Step 5:** Determine significance:

- For large sample sizes ($n_1, n_2 > 20$), use a **normal approximation** with a Z-score.
- For small sample sizes, use the exact distribution of $U$.

### Interpretation

- If the p-value is small (e.g., $p < 0.05$): Reject $H_0$; the two groups have significantly different distributions.
- If the p-value is large: Fail to reject $H_0$; no evidence of a difference.

### Which Group is Larger if $H_0$ is Rejected?

Compare the **mean rank** of each group. A higher mean rank indicates that the values in that group tend to be larger.

### Python Implementation

```python
import numpy as np
from scipy.stats import mannwhitneyu

# Example: comparing test scores of two teaching methods
group_a = np.array([85, 78, 92, 88, 76, 95, 89, 82, 91, 87])
group_b = np.array([72, 68, 81, 75, 70, 77, 74, 69, 73, 71])

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"U statistic: {stat}")
print(f"P-value: {p_value:.4f}")

# Mean ranks
combined = np.concatenate([group_a, group_b])
ranks = np.argsort(np.argsort(combined)) + 1
mean_rank_a = ranks[:len(group_a)].mean()
mean_rank_b = ranks[len(group_a):].mean()
print(f"Mean rank Group A: {mean_rank_a:.1f}")
print(f"Mean rank Group B: {mean_rank_b:.1f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: Significant difference between groups.")
    if mean_rank_a > mean_rank_b:
        print("Group A tends to have larger values.")
    else:
        print("Group B tends to have larger values.")
else:
    print("Fail to reject H0.")
```

### Equivalence Note

The Mann-Whitney U test and Wilcoxon rank-sum test are **statistically equivalent**. The choice of terminology often depends on the software:

- **SPSS**: "Mann-Whitney U test"
- **R**: "Wilcoxon rank-sum test" (`wilcox.test`)
- **Python scipy**: `mannwhitneyu` or `ranksums`

---

## 2. Mood's Median Test

**Mood's Median Test** is a non-parametric hypothesis test used to compare the medians of two or more groups. It is particularly useful when data is non-normal, ordinal, or contains outliers, as it focuses solely on the median rather than the mean or variance.

### Key Features

- **Non-parametric**: Does not require assumptions about the distribution.
- **Robust to Outliers**: Focuses on the median.
- **Applicable for Two or More Groups**.
- **Null Hypothesis ($H_0$)**: All groups have the same median.
- **Alternative Hypothesis ($H_1$)**: At least one group has a different median.

### Assumptions

1. The samples are independent.
2. The data is continuous or ordinal.
3. The groups are random samples from the population.

### How It Works

**Step 1:** Calculate the overall median of the combined dataset.

**Step 2:** For each group, classify data points as "above" or "below" the overall median.

**Step 3:** Construct a contingency table showing the counts of values above and below the overall median for each group.

**Step 4:** Apply the chi-square test of independence to the contingency table:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

where $O$ is the observed frequency and $E$ is the expected frequency under the null hypothesis.

**Step 5:** Evaluate p-value to determine significance.

### Worked Example

**Scenario**: Compare median test scores of two student groups.

- Group A: $[50, 55, 60, 65, 70]$
- Group B: $[45, 50, 55, 60, 65]$

### Python Implementation

```python
import numpy as np
from scipy.stats import chi2_contingency

def moods_median_test(*groups):
    """
    Perform Mood's Median Test to compare medians of multiple groups.

    Parameters:
    - groups: Variable number of arrays representing the groups.

    Returns:
    - chi2_stat: The chi-square statistic.
    - p_value: The p-value for the test.
    - contingency_table: The contingency table used.
    """
    combined_data = np.concatenate(groups)
    overall_median = np.median(combined_data)

    contingency_table = []
    for group in groups:
        above_median = np.sum(group > overall_median)
        below_median = np.sum(group < overall_median)
        contingency_table.append([above_median, below_median])

    contingency_table = np.array(contingency_table).T
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    return chi2_stat, p_value, contingency_table

# Example Data
group_a = np.array([50, 55, 60, 65, 70])
group_b = np.array([45, 50, 55, 60, 65])

chi2_stat, p_value, contingency_table = moods_median_test(group_a, group_b)

print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Contingency Table:\n{contingency_table}")
```

**Output:**

```
Chi-Square Statistic: 0.4000
P-value: 0.5271
Contingency Table:
[[3 2]
 [2 3]]
```

**Interpretation**: The p-value (0.5271) is greater than 0.05, so we fail to reject the null hypothesis. No significant difference in medians.

### Advantages and Limitations

| Advantages | Limitations |
|---|---|
| Robust to outliers | Ignores variability within groups |
| Non-parametric | Low power compared to parametric tests |
| Simple to implement | Requires independence |
| Works with multiple groups | |

---

## 3. Kruskal-Wallis Test

The **Kruskal-Wallis test** is the non-parametric extension of one-way ANOVA, used to compare the distributions of three or more independent groups.

### Key Features

- **Purpose**: Test whether the medians (or distributions) of three or more groups differ.
- **Null Hypothesis**: All groups have the same distribution.
- **Alternative**: At least one group differs.
- **Relation**: When there are only two groups, the Kruskal-Wallis test is equivalent to the Mann-Whitney U test.

### Test Statistic

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $N$ is the total number of observations, $k$ is the number of groups, $n_i$ is the size of group $i$, and $R_i$ is the sum of ranks for group $i$.

Under $H_0$, $H$ follows approximately a $\chi^2$ distribution with $k - 1$ degrees of freedom.

### Python Implementation

```python
from scipy.stats import kruskal

group_a = [85, 78, 92, 88, 76]
group_b = [72, 68, 81, 75, 70]
group_c = [90, 95, 88, 92, 87]

stat, p_value = kruskal(group_a, group_b, group_c)

print(f"H statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: At least one group differs significantly.")
else:
    print("Fail to reject H0.")
```

### Post-Hoc Testing

If the Kruskal-Wallis test is significant, use pairwise Mann-Whitney U tests with a correction for multiple comparisons (e.g., Bonferroni correction) to determine which specific groups differ.

---

## Comparison: Two-Sample Parametric vs Non-Parametric Tests

| Feature | Two-Sample t-Test | Mann-Whitney U | Mood's Median |
|---|---|---|---|
| **Assumption** | Normality, equal variances | None (ordinal+) | None |
| **Tests** | Means | Distributions/ranks | Medians |
| **Power** | Highest (when normal) | Moderate | Low |
| **Robustness** | Sensitive to outliers | Robust | Very robust |
| **Multiple groups** | Use ANOVA | Use Kruskal-Wallis | Can handle |

### Guideline for Choosing

- **Normal data, equal variances**: Use the **two-sample t-test** (or Welch's t-test for unequal variances).
- **Non-normal data, continuous**: Use the **Mann-Whitney U test**.
- **Data with extreme outliers**: Use **Mood's Median Test**.
- **Three or more groups**: Use **Kruskal-Wallis** (non-parametric) or **ANOVA** (parametric).
