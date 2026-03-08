# Bartlett's Test for Equality of Variances

Bartlett's test assesses the equality of variances across multiple groups. It is most appropriate when the data are normally distributed, as it is highly sensitive to deviations from normality. While Bartlett's test is powerful under the assumption of normality, it can lead to incorrect conclusions if the data are non-normal.

## Hypotheses

**Null Hypothesis ($H_0$):** The variances are equal across the groups:

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**Alternative Hypothesis ($H_1$):** At least one group has a variance that differs from the others:

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{for at least one pair} \quad i \neq j
$$

## Assumptions

1. **Normality:** Bartlett's test assumes the data are normally distributed in each group. This is critical because the test is highly sensitive to deviations from normality.
2. **Independence:** The observations must be independent within and between groups.
3. **Random Sampling:** The data should be drawn from random samples.

## Test Statistic

The test statistic for Bartlett's test is derived from the ratio of the pooled variance to the individual sample variances:

$$
T = \frac{(N - k) \ln(S_p^2) - \sum_{i=1}^k (n_i - 1) \ln(S_i^2)}{1 + \frac{1}{3(k - 1)} \left( \sum_{i=1}^k \frac{1}{n_i - 1} - \frac{1}{N - k} \right)} \sim \chi^2_{k-1}
$$

where:

- $N$ is the total number of observations,
- $k$ is the number of groups,
- $S_p^2$ is the pooled variance:

$$
S_p^2 = \frac{\sum_{i=1}^k (n_i - 1) S_i^2}{N - k}
$$

- $S_i^2$ is the sample variance for group $i$.

The test statistic $T$ follows a chi-square distribution with $k - 1$ degrees of freedom.

## Decision Rule

$$
T > \chi^2_{\text{critical}} \quad \Rightarrow \quad \text{Reject } H_0
$$

## Limitations

Bartlett's test is not robust to violations of the normality assumption. If the data are not normally distributed, alternative tests like Levene's test or the Brown–Forsythe test should be considered.

## Python Implementation

### Using SciPy

```python
import numpy as np
from scipy.stats import bartlett

# Example data
group1 = [12, 15, 14, 10, 13, 14, 12, 11]
group2 = [22, 25, 20, 18, 24, 23, 19, 21]
group3 = [32, 35, 34, 30, 33, 34, 32, 31]

# Perform Bartlett's test
statistic, p_value = bartlett(group1, group2, group3)

# Output the results
print(f"Bartlett's test statistic: {statistic}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in variances.")
```

### Manual Computation

```python
import numpy as np
from scipy import stats

# Example data
group1 = np.array([12, 15, 14, 10, 13, 14, 12, 11])
group2 = np.array([22, 25, 20, 18, 24, 23, 19, 21])
group3 = np.array([32, 35, 34, 30, 33, 34, 32, 31])

# Step 1: Calculate variances for each group
variance_group1 = group1.var(ddof=1)
variance_group2 = group2.var(ddof=1)
variance_group3 = group3.var(ddof=1)

# Step 2: Calculate sample sizes
sample_size1 = group1.size
sample_size2 = group2.size
sample_size3 = group3.size

# Step 3: Calculate pooled variance
total_sample_size = sample_size1 + sample_size2 + sample_size3
number_of_groups = 3
degrees_of_freedom_pooled = total_sample_size - number_of_groups

pooled_variance = (
    ((sample_size1 - 1) * variance_group1) +
    ((sample_size2 - 1) * variance_group2) +
    ((sample_size3 - 1) * variance_group3)
) / degrees_of_freedom_pooled

# Step 4: Calculate the numerator (logarithmic terms)
numerator = (
    (total_sample_size - number_of_groups) * np.log(pooled_variance) -
    ((sample_size1 - 1) * np.log(variance_group1) +
     (sample_size2 - 1) * np.log(variance_group2) +
     (sample_size3 - 1) * np.log(variance_group3))
)

# Step 5: Calculate the denominator (correction term)
correction_term = (
    (1 / (sample_size1 - 1)) +
    (1 / (sample_size2 - 1)) +
    (1 / (sample_size3 - 1)) -
    (1 / degrees_of_freedom_pooled)
)
denominator = 1 + correction_term / (3 * (number_of_groups - 1))

# Step 6: Calculate Bartlett's test statistic
bartlett_statistic = numerator / denominator

# Step 7: Calculate p-value using chi-square distribution
degrees_of_freedom = number_of_groups - 1
p_value = stats.chi2.sf(bartlett_statistic, degrees_of_freedom)

# Display results
print(f"Bartlett's Test Statistic (T): {bartlett_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in variances.")
```
