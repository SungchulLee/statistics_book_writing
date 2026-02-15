# Two-Sample Permutation Tests

## Overview

The two-sample permutation test is a non-parametric method for testing whether two independent samples differ in location (typically their means). Unlike the t-test, it requires no assumptions about normality or equal variances. It is particularly useful in A/B testing and causal inference.

## Null Hypothesis

Under the null hypothesis of no difference between groups:
- The observed difference in means (or other statistic) is due to random variation
- Group labels are arbitrary—shuffling them should produce test statistics as extreme as the observed value with the given p-value probability

## Test Procedure

### Algorithm

1. **Compute the observed test statistic** from the actual data:
   $$T_{obs} = |\bar{X}_1 - \bar{X}_2|$$

2. **Combine all observations** into a pooled dataset of size $n_1 + n_2$

3. **Permute B times** (typically B = 1,000 to 10,000):
   - Randomly shuffle group labels
   - Randomly allocate $n_1$ observations to group 1 and $n_2$ to group 2
   - Calculate the test statistic for this permutation

4. **Compute the p-value**:
   $$p\text{-value} = \frac{\#\{T_b \geq T_{obs}\}}{B}$$

## Example: Web Page A/B Test

### Data

Two web pages are tested for user engagement (session time in seconds):

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# Session times for Page A and Page B
page_a = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167])
page_b = np.array([173, 181, 182, 170, 169, 177, 168, 183, 169, 164])

# Observed difference
obs_diff = np.abs(page_a.mean() - page_b.mean())
print(f"Page A: μ̂ = {page_a.mean():.2f}")
print(f"Page B: μ̂ = {page_b.mean():.2f}")
print(f"Observed |difference|: {obs_diff:.2f}")
```

### Permutation Test

```python
def two_sample_permutation_test(x, y, n_perms=1000):
    """
    Perform a two-sample permutation test.

    Parameters:
    -----------
    x : array-like
        First sample
    y : array-like
        Second sample
    n_perms : int
        Number of permutations

    Returns:
    --------
    p_value : float
        Two-sided p-value
    perm_diffs : array
        All permuted test statistics
    """
    obs_diff = np.abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    nx = len(x)

    perm_diffs = np.zeros(n_perms)
    for i in range(n_perms):
        # Randomly shuffle pooled data
        np.random.shuffle(pooled)
        # Assign first nx to group 1, rest to group 2
        perm_diffs[i] = np.abs(pooled[:nx].mean() - pooled[nx:].mean())

    # Two-sided p-value
    p_value = np.mean(perm_diffs >= obs_diff)
    return p_value, perm_diffs

# Run permutation test
np.random.seed(42)
p_val, perm_stats = two_sample_permutation_test(page_a, page_b, n_perms=1000)

print(f"\nPermutation test p-value: {p_val:.4f}")
print(f"Conclusion: {'Reject H₀' if p_val < 0.05 else 'Fail to reject H₀'}")
```

### Visualization

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of permuted test statistics
ax.hist(perm_stats, bins=30, alpha=0.7, color='steelblue', edgecolor='black',
        label='Permuted test statistics')

# Mark observed statistic
obs_stat = np.abs(page_a.mean() - page_b.mean())
ax.axvline(obs_stat, color='red', linewidth=2, label=f'Observed difference = {obs_stat:.2f}')

# Highlight rejection region
rejection_region = perm_stats[perm_stats >= obs_stat]
ax.hist(rejection_region, bins=30, alpha=0.5, color='red',
        label=f'P-value region (p = {p_val:.3f})')

ax.set_xlabel('Absolute Difference in Means')
ax.set_ylabel('Frequency')
ax.set_title('Two-Sample Permutation Test Distribution')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## Example: A/B Conversion Rate Test

For binary outcomes (conversion = 1, no conversion = 0), the permutation test works similarly:

```python
# A/B test: Control vs. Treatment
# Control: 200 conversions out of 23,739 users
# Treatment: 182 conversions out of 22,588 users

n_control, conv_control = 23739, 200
n_treatment, conv_treat = 22588, 182

# Create binary response vector
binary_response = np.zeros(n_control + n_treatment)
binary_response[:conv_control + conv_treat] = 1
np.random.shuffle(binary_response)

# Observed difference in rates
rate_control = conv_control / n_control
rate_treatment = conv_treat / n_treatment
obs_diff_rates = rate_treatment - rate_control

print(f"Control conversion rate: {rate_control:.4f}")
print(f"Treatment conversion rate: {rate_treatment:.4f}")
print(f"Observed difference: {obs_diff_rates:.4f}")

# Permutation test
perm_diffs_rates = []
for _ in range(5000):
    np.random.shuffle(binary_response)
    perm_rate_control = binary_response[:n_control].mean()
    perm_rate_treatment = binary_response[n_control:].mean()
    perm_diffs_rates.append(perm_rate_treatment - perm_rate_control)

p_value_ab = np.mean(np.abs(perm_diffs_rates) >= np.abs(obs_diff_rates))
print(f"A/B test p-value: {p_value_ab:.4f}")
```

## Comparison to Parametric Tests

### Two-Sample Permutation Test vs. t-Test

Both tests address the same hypothesis, but with different approaches:

| Aspect | Permutation Test | t-Test |
|--------|-----------------|--------|
| **Assumptions** | None (distribution-free) | Normality or large n |
| **Variance** | Unequal variances handled naturally | Requires Welch's adjustment |
| **Intuition** | Randomization-based | Probability theory-based |
| **p-value precision** | Limited by number of permutations | Exact (continuous) |
| **Power** | Similar or slightly lower | Slightly higher if assumptions hold |
| **Computational cost** | Higher | Negligible |

### Example: Comparing Results

```python
from scipy import stats

# t-test
t_stat, p_ttest = stats.ttest_ind(page_a, page_b, equal_var=False)
print(f"Welch's t-test p-value: {p_ttest:.4f}")

# Permutation test (from previous code)
print(f"Permutation test p-value: {p_val:.4f}")

# Both should give similar results
```

## Advantages

1. **No distributional assumptions**: Works with any data type
2. **Intuitive**: Direct interpretation under randomization
3. **Robust**: Naturally handles outliers and non-normality
4. **Flexible**: Can be applied to any test statistic

## Disadvantages

1. **Computational**: More intensive than parametric tests
2. **Discrete p-values**: p-values are multiples of $1/B$
3. **Small sample concerns**: With very small samples, p-value granularity can be coarse

## When to Use

- **Always valid**: Small samples, non-normal data, or unknown distributions
- **A/B testing**: Standard approach in tech industry
- **Robustness check**: Compare against parametric results
- **Custom test statistics**: When you need unusual statistics (median, trimmed mean, etc.)

## One-Sided vs. Two-Sided Tests

### Two-Sided Test (Default)

$$p\text{-value} = \frac{\#\{|T_b| \geq |T_{obs}|\}}{B}$$

Tests if $H_a: \mu_1 \neq \mu_2$

### One-Sided Test

For $H_a: \mu_1 > \mu_2$:

$$p\text{-value} = \frac{\#\{T_b \geq T_{obs}\}}{B}$$

where $T_b = \bar{X}_{1,b} - \bar{X}_{2,b}$ (without absolute value)

## Summary

Two-sample permutation tests provide a powerful, assumption-free alternative to t-tests. They are especially valuable in A/B testing contexts where the randomization design naturally leads to the null hypothesis of no difference between treatment groups. The permutation distribution directly reflects what we would expect to observe if groups were randomly shuffled under the null hypothesis.
