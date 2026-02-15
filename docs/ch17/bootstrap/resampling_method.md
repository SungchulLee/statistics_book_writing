# The Bootstrap Resampling Method

## Overview

The **bootstrap** is a powerful, distribution-free method for estimating the sampling distribution of a statistic without making parametric assumptions. By repeatedly resampling (with replacement) from a single sample, we can approximate the sampling distribution and compute standard errors, confidence intervals, and other inferential quantities.

## Core Idea

The bootstrap rests on a simple principle: the **empirical distribution** of our sample is a reasonable estimate of the true population distribution. If we repeatedly sample from our observed sample (with replacement), the variability in our computed statistics approximates the true sampling variability.

**Key insight**: The bootstrap trades computational effort for avoiding strong distributional assumptions.

## Bootstrap Algorithm

Given a sample of $n$ observations:

1. **Resample**: Draw $n$ observations from the sample **with replacement**, creating a bootstrap sample
2. **Compute**: Calculate the statistic of interest on the bootstrap sample
3. **Repeat**: Steps 1-2 a large number of times (typically 500-10,000)
4. **Analyze**: The collection of computed statistics forms the bootstrap distribution

```
Original Sample: X₁, X₂, ..., Xₙ
    ↓
    ├→ Bootstrap Sample 1* → Statistic θ₁*
    ├→ Bootstrap Sample 2* → Statistic θ₂*
    ├→ Bootstrap Sample 3* → Statistic θ₃*
    └→ Bootstrap Sample B* → Statistic θ_B*
    ↓
Bootstrap Distribution: {θ₁*, θ₂*, ..., θ_B*}
```

## Example: Bootstrap Distribution of the Median

The median is particularly useful when data contain outliers or aren't normally distributed. Unlike the mean, the median has **no simple formula** for its standard error. The bootstrap solves this problem elegantly.

```python
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Set random seed
np.random.seed(seed=1)

# Simulate income data (realistic for illustrating robustness of median)
loans_income = np.random.exponential(scale=50000, size=5000) + 20000

# Compute original sample median
original_median = loans_income.median()
print(f"Original sample median: ${original_median:,.0f}")
print()

# Bootstrap procedure: resample 1000 times
bootstrap_medians = []
for nrepeat in range(1000):
    # Resample with replacement
    bootstrap_sample = resample(loans_income)  # size=len(loans_income) by default
    # Compute median of bootstrap sample
    bootstrap_medians.append(bootstrap_sample.median())

bootstrap_medians = pd.Series(bootstrap_medians)

# Compute bootstrap statistics
bootstrap_mean = bootstrap_medians.mean()
bootstrap_std = bootstrap_medians.std()
bias = bootstrap_mean - original_median

print(f"Bootstrap Statistics:")
print(f"  Mean of bootstrap distribution:  ${bootstrap_mean:,.0f}")
print(f"  Standard error of median:        ${bootstrap_std:,.0f}")
print(f"  Bias of median estimator:        ${bias:,.0f}")
print()

# The bootstrap distribution can be visualized
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(bootstrap_medians, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(original_median, color='darkred', linestyle='--', linewidth=2.5, label=f'Original median: ${original_median:,.0f}')
ax.axvline(bootstrap_mean, color='green', linestyle='--', linewidth=2.5, label=f'Bootstrap mean: ${bootstrap_mean:,.0f}')
ax.set_xlabel('Median Income ($)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Bootstrap Distribution of the Sample Median', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## Interpreting Bootstrap Results

### Standard Error

The standard deviation of the bootstrap distribution is the **standard error** of the statistic:

$$SE(\text{median}) \approx \text{std}(\{\theta_1^*, \theta_2^*, \ldots, \theta_B^*\})$$

This estimates the typical variation in the median across different samples from the population.

### Bias

If the mean of the bootstrap distribution differs from the original statistic, the estimator has **bias**:

$$\text{Bias} = E[\text{Estimator}] - \text{True Parameter} \approx \text{mean}(\text{Bootstrap distribution}) - \text{Original statistic}$$

In the example above, a small negative bias suggests the median is slightly downward-biased (though this depends on the specific data).

### Bootstrap Distribution Shape

The shape of the bootstrap distribution reveals:
- **Skewness**: Non-symmetry indicates asymmetric sampling distribution
- **Heavy tails**: Suggests the statistic is sensitive to outliers
- **Multimodality**: Can indicate clustered data or multiple population modes

## Advantages of Bootstrap

### 1. Distribution-Free
No assumption of normality or specific distributional form required. The bootstrap works for:
- Non-normal data
- Skewed distributions
- Heavy-tailed distributions
- Any arbitrary population shape

### 2. General Applicability
Works for any statistic, not just the mean:
- Median
- Correlation coefficient
- Ratio statistics
- Quantiles and percentiles
- Custom estimators

### 3. Intuitive and Transparent
The procedure directly estimates sampling variability without requiring theoretical derivations. This makes it:
- Easy to understand conceptually
- Easy to explain to non-statisticians
- Easy to implement and verify

## Comparison with Parametric Approaches

| Aspect | Bootstrap | Parametric (Theory-based) |
|:---|:---|:---|
| **Assumptions** | Minimal (IID sample) | Strong (normality, known variance, etc.) |
| **Applicability** | Any statistic | Limited to standard statistics |
| **Computation** | Resampling (intensive) | Formulas (fast) |
| **Validity** | Asymptotic, improves with B | Exact or asymptotic |
| **Implementation** | Simple code | Requires mathematical knowledge |

## Practical Considerations

### Sample Size Requirements

The bootstrap requires that the original sample be reasonably representative of the population. It works poorly when:
- Sample size is very small ($n < 30$) relative to population variability
- Extreme values are missing from the sample
- The sample is biased or non-random

### Number of Bootstrap Replications

**Rule of thumb**: Use $B = 1000$ for confidence intervals or $B \geq 500$ for standard errors.

For extreme quantiles (e.g., 99th percentile), use $B \geq 5000$.

```python
# Standard error with different B values
np.random.seed(1)
np.random.exponential(scale=50000, size=5000)

for B in [100, 500, 1000, 5000]:
    boot_medians = np.array([
        np.median(np.random.choice(loans_income, size=len(loans_income), replace=True))
        for _ in range(B)
    ])
    se = boot_medians.std()
    print(f"B = {B:5d}: SE = ${se:8,.0f}")
```

### Computational Cost

Modern computers can easily handle 10,000 bootstrap replications for standard statistics. For more computationally intensive operations (e.g., fitting complex models), start with $B = 500$ and increase if necessary.

## Limitations and Pitfalls

1. **Bootstrap can't estimate extremes well**: For the maximum of a sample, the bootstrap max is always ≤ the observed max
2. **Dependent data**: The standard bootstrap assumes independent observations; time series or clustered data require modifications (block bootstrap)
3. **Bias not accounted for**: The bootstrap estimates standard error, not bias; biased estimators remain biased in the bootstrap
4. **Small samples**: With very small samples, the empirical distribution may poorly represent the population

## Extensions and Variants

### Block Bootstrap
For time series or clustered data:
```python
def block_bootstrap(data, block_size, n_bootstrap):
    n = len(data)
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        blocks = [data[i:(i+block_size)] for i in np.arange(0, n, block_size)]
        resampled_blocks = np.random.choice(len(blocks), size=len(blocks), replace=True)
        bootstrap_sample = np.concatenate([blocks[i] for i in resampled_blocks])[:n]
        bootstrap_samples.append(bootstrap_sample)
    return bootstrap_samples
```

### Percentile-t Bootstrap
More accurate confidence intervals for some statistics:
```python
# Compute t-statistics and use t-quantiles instead of percentile quantiles
bootstrap_t_stats = (bootstrap_statistics - original_statistic) / bootstrap_ses
ci_lower = original_statistic - np.percentile(bootstrap_t_stats, 97.5) * original_se
ci_upper = original_statistic - np.percentile(bootstrap_t_stats, 2.5) * original_se
```

## Summary

The bootstrap is a foundational tool in modern statistics:
- **Intuitive method** based on resampling from the data
- **Widely applicable** to any statistic and any distribution
- **Computationally accessible** with modern computing power
- **Distribution-free** and makes minimal assumptions

It trades computational effort for avoiding strong parametric assumptions, making it invaluable when theory-based methods are inadequate or unavailable.
