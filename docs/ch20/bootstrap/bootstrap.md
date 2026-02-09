# Bootstrap

## Introduction

The **Bootstrap Method** is a powerful, computer-intensive statistical technique used to estimate the distribution of a statistic (e.g., mean, median) by repeatedly resampling with replacement from the observed data. Unlike parametric methods, it makes no assumptions about the underlying distribution of the data.

### Key Features

1. **Distribution-Free**: Does not rely on the data following any specific distribution.
2. **Versatile**: Can be used to estimate confidence intervals, standard errors, and hypothesis tests for any statistic.
3. **Core Idea**: Simulate multiple datasets by resampling the observed data with replacement.

---

## How It Works

### Step 1: Resample with Replacement

From the original dataset of size $n$, generate a new sample (a "bootstrap sample") of the same size $n$ by sampling with replacement. Some data points may appear multiple times; others may be excluded.

### Step 2: Compute the Statistic

Calculate the desired statistic (e.g., mean, median) for the bootstrap sample.

### Step 3: Repeat

Repeat the resampling process a large number of times (e.g., 1,000 or 10,000 iterations) to create a distribution of the statistic.

### Step 4: Analyze the Bootstrap Distribution

Use the bootstrap distribution to estimate properties of the statistic:

- **Confidence Intervals**: Determine the range within which the true parameter likely falls.
- **Standard Error**: The standard deviation of the bootstrap distribution estimates the standard error.
- **Hypothesis Testing**: Compare the observed statistic to the bootstrap distribution under the null hypothesis.

---

## Bootstrap for Confidence Intervals

### Percentile Method

The simplest approach: use the percentiles of the bootstrap distribution directly.

```python
import numpy as np

def bootstrap_confidence_interval(data, statistic=np.mean, n_resamples=10000, ci=95):
    """
    Compute the bootstrap confidence interval for a given statistic.

    Parameters:
    - data: List or array of data points.
    - statistic: Function to compute the statistic (default is np.mean).
    - n_resamples: Number of bootstrap resamples.
    - ci: Desired confidence level (e.g., 95 for 95% CI).

    Returns:
    - lower_bound, upper_bound, bootstrap_distribution
    """
    bootstrap_distribution = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])

    lower_bound = np.percentile(bootstrap_distribution, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_distribution, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound, bootstrap_distribution

# Example
data = [5, 7, 9, 12, 15]
lower, upper, bootstrap_dist = bootstrap_confidence_interval(data)
print(f"95% Confidence Interval for the Mean: ({lower:.2f}, {upper:.2f})")
```

### Bootstrap Standard Error

The standard error of a statistic is estimated as the standard deviation of the bootstrap distribution:

```python
se = np.std(bootstrap_dist, ddof=1)
print(f"Bootstrap Standard Error: {se:.4f}")
```

### BCa (Bias-Corrected and Accelerated) Method

The BCa method adjusts for both bias and skewness in the bootstrap distribution, providing more accurate confidence intervals than the simple percentile method:

```python
from scipy.stats import bootstrap
import numpy as np

data = np.array([5, 7, 9, 12, 15])
result = bootstrap((data,), np.mean, n_resamples=10000,
                    confidence_level=0.95, method='BCa')
print(f"BCa 95% CI: ({result.confidence_interval.low:.2f}, "
      f"{result.confidence_interval.high:.2f})")
```

---

## Bootstrap for Hypothesis Testing

To test $H_0: \mu = \mu_0$:

1. **Shift the data** so the sample mean equals $\mu_0$: $X_i^* = X_i - \bar{X} + \mu_0$.
2. **Bootstrap from the shifted data** to get the null distribution of $\bar{X}^*$.
3. **Compute the p-value** as the proportion of bootstrap means at least as extreme as the observed $\bar{X}$.

```python
import numpy as np

def bootstrap_hypothesis_test(data, mu_0, n_resamples=10000, alternative='two-sided'):
    """
    Bootstrap hypothesis test for the mean.

    Parameters:
    - data: Observed data.
    - mu_0: Hypothesized mean under H0.
    - n_resamples: Number of bootstrap resamples.
    - alternative: 'two-sided', 'greater', or 'less'.

    Returns:
    - p_value: Bootstrap p-value.
    """
    data = np.array(data)
    observed_mean = data.mean()
    shifted_data = data - observed_mean + mu_0  # Center at mu_0

    bootstrap_means = np.array([
        np.random.choice(shifted_data, size=len(data), replace=True).mean()
        for _ in range(n_resamples)
    ])

    if alternative == 'two-sided':
        p_value = np.mean(np.abs(bootstrap_means - mu_0) >= np.abs(observed_mean - mu_0))
    elif alternative == 'greater':
        p_value = np.mean(bootstrap_means >= observed_mean)
    elif alternative == 'less':
        p_value = np.mean(bootstrap_means <= observed_mean)

    return p_value

# Example
data = [5, 7, 9, 12, 15]
p_value = bootstrap_hypothesis_test(data, mu_0=8, alternative='two-sided')
print(f"Bootstrap p-value: {p_value:.4f}")
```

---

## Bootstrap for Complex Statistics

One of bootstrap's greatest strengths is handling statistics that have no simple analytical formula for their sampling distribution.

### Bootstrap for the Median

```python
data = np.array([3, 7, 8, 12, 15, 20, 45])
lower, upper, _ = bootstrap_confidence_interval(data, statistic=np.median)
print(f"95% CI for Median: ({lower:.2f}, {upper:.2f})")
```

### Bootstrap for Correlation

```python
def bootstrap_correlation(x, y, n_resamples=10000, ci=95):
    n = len(x)
    boot_corrs = []
    for _ in range(n_resamples):
        idx = np.random.choice(n, size=n, replace=True)
        boot_corrs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    boot_corrs = np.array(boot_corrs)
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)
    return lower, upper

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 3, 5, 4, 6, 8, 7, 9])
lower, upper = bootstrap_correlation(x, y)
print(f"95% CI for Correlation: ({lower:.4f}, {upper:.4f})")
```

---

## Pros and Cons

| Pros | Cons |
|---|---|
| No distributional assumptions | Computationally intensive |
| Works with small samples | Sample must be representative of population |
| Handles any statistic (mean, median, correlation, etc.) | Very small datasets may lack variability |
| Provides confidence intervals directly | Multiple bootstrap methods exist; choosing can be confusing |

---

## Applications

1. **Confidence Intervals**: For means, medians, variances, regression coefficients, correlations, and any other statistic.
2. **Hypothesis Testing**: Non-parametric hypothesis testing via shifted bootstrap.
3. **Model Validation**: Validate machine learning models by resampling data and assessing performance.
4. **Time Series**: Adapted versions (e.g., block bootstrap) for dependent data.
5. **Finance**: Estimating Value at Risk (VaR), portfolio return distributions.
