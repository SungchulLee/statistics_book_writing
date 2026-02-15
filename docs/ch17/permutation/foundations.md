# Permutation Tests: Foundations

## Overview

Permutation tests are a resampling-based approach to hypothesis testing that does not rely on parametric assumptions. Rather than assuming a particular probability distribution (like the normal distribution), permutation tests use the observed data to generate a distribution of the test statistic under the null hypothesis by randomly shuffling or rearranging the data.

## Key Concepts

### Why Permutation Tests?

Permutation tests are valuable because they:

1. **Avoid Distributional Assumptions**: No need to assume normality or equal variances
2. **Work with Small Samples**: Valid even with modest sample sizes
3. **Are Intuitive**: The logic is straightforward—if the null hypothesis is true, the labels are arbitrary
4. **Are General**: Can be applied to any test statistic we define

### The Core Logic

Under the null hypothesis of no difference between groups:
- The group labels (e.g., "Page A" vs "Page B") are arbitrary
- Randomly rearranging these labels should produce test statistics as extreme as (or more extreme than) the observed value with probability related to the p-value
- This random rearrangement creates an empirical null distribution

### General Algorithm

1. **Calculate the observed test statistic** from the actual data
2. **Permute the data** many times (typically 1,000 to 10,000 times):
   - Randomly shuffle group labels or resample data
   - Recalculate the test statistic for each permuted dataset
3. **Compare**: Count how many permuted statistics are as extreme as the observed statistic
4. **Calculate p-value**: $p = \frac{\text{# extreme permuted statistics}}{N_{\text{permutations}}}$

## Example: Web Page Stickiness (A/B Testing)

A company tests whether users spend more time on Page B compared to Page A. This is a classic application of permutation testing in A/B testing.

### Data and Observed Difference

Suppose we have session times (in seconds) for users on each page:

```python
import pandas as pd
import numpy as np
import random

# Sample data (from Practical Statistics for Data Scientists)
session_times = pd.DataFrame({
    'Time': [185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
             173, 181, 182, 170, 169, 177, 168, 183, 169, 164],
    'Page': ['Page A']*10 + ['Page B']*10
})

mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()
observed_diff = mean_b - mean_a

print(f"Page A mean: {mean_a:.2f} seconds")
print(f"Page B mean: {mean_b:.2f} seconds")
print(f"Observed difference: {observed_diff:.2f} seconds")
```

### Permutation Test Implementation

```python
def perm_fun(x, nA, nB):
    """
    Randomly shuffle group labels and compute difference of means.

    Parameters:
    -----------
    x : array-like
        Combined data (times for all users)
    nA : int
        Number of observations in group A
    nB : int
        Number of observations in group B

    Returns:
    --------
    float : Difference in means (B - A) for the permuted assignment
    """
    n = nA + nB
    # Randomly select nB indices for group B; rest go to group A
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B

    return x.loc[list(idx_B)].mean() - x.loc[list(idx_A)].mean()

# Number of observations in each group
nA = session_times[session_times.Page == 'Page A'].shape[0]
nB = session_times[session_times.Page == 'Page B'].shape[0]

# Run permutation test (1,000 permutations)
random.seed(42)
perm_diffs = [perm_fun(session_times.Time, nA, nB) for _ in range(1000)]

# Calculate p-value: proportion of permuted diffs as extreme as observed
p_value = np.mean(np.abs(perm_diffs) > np.abs(observed_diff))
print(f"Permutation test p-value: {p_value:.4f}")
```

### Interpretation

If p-value ≤ 0.05, we reject the null hypothesis and conclude that the pages have significantly different session times.

## Example: A/B Test for Conversion Rate

Another common A/B testing scenario: testing whether a change in web interface increases conversion rate.

```python
# Example: 200 conversions out of 23,739 users (Control)
#          182 conversions out of 22,588 users (Treatment)

# Observed conversion rates
conv_rate_control = 200 / 23739
conv_rate_treatment = 182 / 22588
observed_diff = conv_rate_treatment - conv_rate_control

# Create a binary response vector: 1 = converted, 0 = did not convert
conversion = [0] * (23739 + 22588 - 200 - 182)  # Non-converters
conversion.extend([1] * (200 + 182))  # Converters
conversion = np.array(conversion)

# Permutation test
def perm_fun_conv(x, n_control, n_treatment):
    """Permutation for proportions test."""
    n = len(x)
    idx_t = set(random.sample(range(n), n_treatment))
    idx_c = set(range(n)) - idx_t

    rate_t = x[list(idx_t)].mean()
    rate_c = x[list(idx_c)].mean()
    return rate_t - rate_c

random.seed(42)
perm_diffs_conv = [perm_fun_conv(conversion, 23739, 22588)
                   for _ in range(1000)]

p_value_conv = np.mean(np.abs(perm_diffs_conv) > np.abs(observed_diff))
print(f"Conversion A/B test p-value: {p_value_conv:.4f}")
```

## Advantages and Disadvantages

### Advantages

- **Model-free**: No distributional assumptions required
- **Intuitive**: Direct interpretation of randomization under null hypothesis
- **Flexible**: Can use any test statistic (mean, median, proportion, etc.)
- **Conservative**: Generally controls Type I error rate well

### Disadvantages

- **Computational**: Requires many simulations (though fast on modern computers)
- **Discrete p-values**: Limited by number of permutations used
- **Loss of power**: Sometimes less powerful than parametric tests (when their assumptions hold)

## Computational Considerations

For exact p-values with small samples, consider enumeration of all possible permutations. However, for larger samples:

- 1,000 permutations typically gives p-value precision to ±0.01
- 10,000 permutations gives precision to ±0.001
- More permutations improve precision but face diminishing returns

## Connection to Other Methods

- **Bootstrap Confidence Intervals**: Permutation tests use resampling similar to bootstrap
- **Exact Tests**: Permutation tests are more practical than exact tests for moderate sample sizes
- **Parametric Tests**: Compare permutation p-values to t-test or ANOVA p-values to validate assumptions

## Summary

Permutation tests provide a powerful, assumption-free approach to hypothesis testing. They are especially valuable in A/B testing, where the data generation process is controlled and the null hypothesis (no difference between groups) is natural to assume. By repeatedly shuffling group labels, we can determine whether the observed difference is likely due to chance alone.
