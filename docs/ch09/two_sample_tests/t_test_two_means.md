# Two-Sample t-Test (Pooled and Welch)

## Overview

The two-sample t-test compares the means of two independent samples to determine if they differ significantly. It is widely used in A/B testing, clinical trials, and experimental design.

## Hypotheses

- **Null Hypothesis** ($H_0$): $\mu_1 = \mu_2$ (the means are equal)
- **Alternative Hypothesis** ($H_a$): $\mu_1 \neq \mu_2$ (the means differ)

## Test Statistics

### Pooled t-Test (Assumes Equal Variances)

When both groups are assumed to have the same population variance:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}}
$$

where the pooled standard deviation is:

$$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

**Degrees of freedom**: $df = n_1 + n_2 - 2$

### Welch's t-Test (Unequal Variances Assumed)

When variances may differ, Welch's test is more robust:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

**Degrees of freedom** (Satterthwaite approximation):

$$df = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}$$

**Note**: Welch's test is generally preferred as it does not assume equal variances and maintains Type I error control.

## Practical Considerations

### Equal Variance Assumption

Before choosing between pooled and Welch's tests, one might be tempted to perform a pre-test for equality of variances (like Levene's test). However, modern statistical practice recommends using Welch's test as the default because:

1. It is robust to violations of the equal variance assumption
2. It has nearly identical power to the pooled test when variances are actually equal
3. It provides better protection when variances differ

### Effect Size

For two-sample comparisons, **Cohen's d** measures practical significance:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$

Interpretation:
- $|d| < 0.2$: Small effect
- $0.2 \leq |d| < 0.5$: Small to medium effect
- $0.5 \leq |d| < 0.8$: Medium effect
- $|d| \geq 0.8$: Large effect

## Example: Web Page A/B Test

Suppose we test whether users spend more time on a redesigned web page (Page B) than the current version (Page A):

```python
import numpy as np
from scipy import stats

# Session times in seconds
page_a = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167])
page_b = np.array([173, 181, 182, 170, 169, 177, 168, 183, 169, 164])

# Welch's t-test (default: equal_var=False)
t_stat, p_value = stats.ttest_ind(page_a, page_b, equal_var=False)

print(f"Page A: mean = {page_a.mean():.2f}, std = {page_a.std(ddof=1):.2f}")
print(f"Page B: mean = {page_b.mean():.2f}, std = {page_b.std(ddof=1):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (two-sided): {p_value:.4f}")

# One-sided test: H_a: μ_B > μ_A
p_one_sided = p_value / 2 if page_b.mean() > page_a.mean() else 1 - p_value / 2
print(f"p-value (one-sided): {p_one_sided:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(page_a) - 1) * page_a.std(ddof=1)**2 +
                       (len(page_b) - 1) * page_b.std(ddof=1)**2) /
                      (len(page_a) + len(page_b) - 2))
cohens_d = (page_b.mean() - page_a.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

## Implementation in Python

### Using scipy.stats

```python
from scipy import stats

# Welch's t-test (recommended)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Pooled t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)

# One-sided tests
if t_stat > 0:
    p_one_sided = p_value / 2  # Upper tail
else:
    p_one_sided = 1 - p_value / 2  # Lower tail
```

### Using statsmodels

```python
import statsmodels.api as sm

# Welch's t-test with more details
t_stat, p_value, df = sm.stats.ttest_ind(group1, group2,
                                         usevar='unequal',
                                         alternative='two-sided')
```

## Assumptions

1. **Independence**: Observations within each group are independent
2. **Normality**: Data in each group are approximately normally distributed (less critical with n > 30)
3. **Random Sampling**: Samples are randomly drawn from their respective populations

## When to Use Each Test

| Scenario | Test to Use |
|----------|------------|
| Small samples, variances appear equal | Pooled t-test |
| Any sample size or unclear variances | **Welch's t-test** |
| Non-normal data, small samples | Permutation test or Mann-Whitney U test |
| Large samples (n > 30) | Either test (both work well) |

## Related Tests

- **Paired t-test**: For dependent samples (matched pairs)
- **Mann-Whitney U test**: Non-parametric alternative for non-normal data
- **Permutation test**: Assumption-free resampling approach
- **Bootstrap confidence interval**: For confidence intervals without distributional assumptions
