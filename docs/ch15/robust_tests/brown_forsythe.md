# Brown-Forsythe Test

Levene's test replaces each observation with its absolute deviation from the group mean and then runs an ANOVA on the deviations. Brown and Forsythe (1974) proposed a single but impactful change: use the group **median** instead of the group mean as the center of the absolute deviations. This modification makes the test substantially more robust to skewed distributions and outliers, because the median is not pulled toward extreme values the way the mean is.

## Modification from Levene's Test

In Levene's original test, the transformed observations are

$$
Z_{ij}^{(\text{Levene})} = |X_{ij} - \bar{X}_i|
$$

In the Brown-Forsythe test, they are

$$
Z_{ij}^{(\text{BF})} = |X_{ij} - \tilde{X}_i|
$$

where $\tilde{X}_i$ denotes the median of group $i$. The test statistic is then the standard one-way ANOVA $F$-statistic computed on the $Z_{ij}^{(\text{BF})}$ values:

$$
W^* = \frac{(N - k) \sum_{i=1}^{k} n_i (\bar{Z}_i^* - \bar{Z}^*)^2}{(k - 1) \sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij}^* - \bar{Z}_i^*)^2}
$$

where $Z_{ij}^* = Z_{ij}^{(\text{BF})}$, $\bar{Z}_i^*$ is the mean of the transformed values in group $i$, and $\bar{Z}^*$ is the grand mean.

Under $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2$, the statistic $W^*$ follows approximately an $F_{k-1, N-k}$ distribution.

## Why the Median Improves Robustness

The mean $\bar{X}_i$ is sensitive to outliers: a single extreme value shifts the mean substantially, which inflates the absolute deviations for all observations in the group. This distortion propagates into the test statistic and can cause spurious rejections.

The median $\tilde{X}_i$ has a breakdown point of 50%, meaning up to half the observations can be contaminated before the median is arbitrarily affected. When the data are skewed, the median lies closer to the bulk of the data than the mean does, producing deviations that more faithfully reflect the spread of the majority of the observations.

!!! note "Mean vs. Median vs. Trimmed Mean"
    Some implementations of Levene's test offer three choices of center: mean, median, and 10% trimmed mean. The median version is the Brown-Forsythe test. The trimmed mean provides a compromise between power (favoring the mean) and robustness (favoring the median). For general-purpose use, the median is the recommended default.

## Hypotheses and Decision Rule

$$
H_0\colon \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2
$$

$$
H_1\colon \sigma_i^2 \neq \sigma_j^2 \text{ for at least one pair } i \neq j
$$

Reject $H_0$ at significance level $\alpha$ if $W^* > F_{1-\alpha,\, k-1,\, N-k}$.

## Example

Consider three groups:

| Group 1 | Group 2 | Group 3 |
|---|---|---|
| 8, 10, 12, 9, 11 | 5, 30, 18, 22, 15 | 14, 16, 15, 17, 14 |

**Step 1.** Compute group medians:

- $\tilde{X}_1 = 10$, $\tilde{X}_2 = 18$, $\tilde{X}_3 = 15$

**Step 2.** Compute absolute deviations from the medians:

| Group 1 | Group 2 | Group 3 |
|---|---|---|
| 2, 0, 2, 1, 1 | 13, 12, 0, 4, 3 | 1, 1, 0, 2, 1 |

**Step 3.** Compute group means of the deviations:

- $\bar{Z}_1^* = 1.20$, $\bar{Z}_2^* = 6.40$, $\bar{Z}_3^* = 1.00$

**Step 4.** The grand mean is $\bar{Z}^* = (1.20 + 6.40 + 1.00) \times 5/15 = 2.867$.

**Step 5.** Compute $W^*$ using the ANOVA formula on these deviations. Group 2 has much larger deviations, reflecting its greater spread. The resulting $W^*$ is compared to $F_{0.95,\, 2,\, 12} = 3.885$.

Notice that Group 2 contains an outlier (30). The median-based deviations are less affected by this outlier than mean-based deviations would be, because the median (18) is closer to the bulk of the data than the mean (18) in this case. For more skewed data, the difference between mean and median centers becomes more pronounced.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Group data
group1 = [8, 10, 12, 9, 11]
group2 = [5, 30, 18, 22, 15]
group3 = [14, 16, 15, 17, 14]

# Brown-Forsythe test (Levene's test with center='median')
stat, p_value = stats.levene(group1, group2, group3, center='median')
print(f"Brown-Forsythe W* statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: variances are significantly different.")
else:
    print("Fail to reject H0: no significant difference in variances.")
```

## Performance Characteristics

The Brown-Forsythe test has been studied extensively through simulation:

- **Type I error control.** The actual rejection rate stays close to the nominal $\alpha$ across a wide range of distributions, including skewed and heavy-tailed populations.
- **Power under normality.** The Brown-Forsythe test has slightly lower power than Bartlett's test or the original Levene's test when the data are truly normal. The power loss is typically small (1--3 percentage points).
- **Power under non-normality.** The Brown-Forsythe test maintains good power for detecting variance differences even when the data are non-normal, because it does not waste power on false alarms caused by distributional shape.

## When to Use the Brown-Forsythe Test

The Brown-Forsythe test is the recommended default for testing homogeneity of variances in most practical situations:

- As a **pre-test before ANOVA**, it provides reliable variance diagnostics regardless of the population shape.
- When the **distribution is unknown**, it offers the best balance of robustness and power among the tests in this chapter.
- When the **data contain outliers**, the median-based deviations prevent spurious rejections.

The main situation where another test is preferable is when normality has been confirmed and maximum power is desired. In that case, Bartlett's test is the most powerful option.
