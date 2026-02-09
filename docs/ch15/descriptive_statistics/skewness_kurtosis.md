# Skewness and Kurtosis

Descriptive statistics quantify the shape of a distribution and assess how closely it resembles a normal distribution. Two key measures for evaluating normality are **skewness** and **kurtosis**. These metrics describe the asymmetry and peakedness of the data distribution, respectively.

## Skewness

**Skewness** measures the asymmetry of the distribution. For a perfectly normal distribution, skewness is 0. A positive skewness indicates a long right tail (data skewed to the right), while a negative skewness indicates a long left tail (data skewed to the left).

The formula for skewness is:

$$
\text{Skewness} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

where

- $n$ is the number of data points,
- $x_i$ is each data point,
- $\bar{x}$ is the sample mean,
- $s$ is the sample standard deviation.

If the skewness is close to zero, the population distribution will likely be symmetric, indicating normality. Significant deviations from zero suggest non-normality.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
data = np.random.normal(0, 3, 1000)
# data = np.random.exponential(1, 1000)

# Calculate skewness
skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")
```

## Kurtosis

**Kurtosis** describes the "tailedness" of the distribution. A normal distribution has a kurtosis value of 3 (also called **mesokurtic**). Kurtosis values above 3 indicate a distribution with heavy tails (**leptokurtic**), while values below 3 indicate lighter tails (**platykurtic**).

The formula for kurtosis is:

$$
\text{Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4
$$

and the formula for **excess kurtosis** is:

$$
\text{Excess Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 - 3
$$

The subtraction of 3 ensures that a normal distribution has an excess kurtosis of 0 (for easier comparison). The `scipy.stats.kurtosis` function computes this **excess kurtosis**, not the raw kurtosis.

A kurtosis value (computed by `scipy.stats.kurtosis`) near zero suggests a normal distribution. Larger values indicate heavier tails, while smaller values suggest lighter tails than normal.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.exponential(1, 1000)

# Calculate skewness
skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")

# Calculate kurtosis
kurtosis_value = stats.kurtosis(data)
print(f"Kurtosis: {kurtosis_value:.4}")
```
