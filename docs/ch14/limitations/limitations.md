# Limitations and Pitfalls of Normality Tests

While formal normality tests provide valuable insights, they have limitations. In practice, these tests can be influenced by factors such as sample size, the power of the test, and the distribution of real-world data. Understanding these pitfalls is crucial for making informed decisions about the appropriateness of normality tests.

## Sensitivity to Sample Size

Normality tests are highly sensitive to sample size. For small samples, these tests often lack the power to detect deviations from normality, potentially leading to a **Type II error** (failing to reject the null hypothesis when it is false). On the other hand, for very large samples, even minor deviations from normality can result in rejecting the null hypothesis, leading to a **Type I error** (incorrectly rejecting the null hypothesis).

- **Small samples**: May not detect actual deviations from normality.
- **Large samples**: May detect even trivial deviations that are not practically significant.

```python
import numpy as np
from scipy.stats import shapiro

# Small sample size
small_sample = np.random.normal(0, 1, 20)
stat_small, p_value_small = shapiro(small_sample)
print(f"Shapiro-Wilk Test (small sample): p-value={p_value_small}")

# Large sample size
large_sample = np.random.normal(0, 1, 10000)
stat_large, p_value_large = shapiro(large_sample)
print(f"Shapiro-Wilk Test (large sample): p-value={p_value_large}")
```

In this example, a small sample size may fail to detect deviations from normality, while a large sample size could reject the null hypothesis due to minor deviations.

## Power of the Tests

The **power** of a normality test refers to its ability to correctly reject the null hypothesis when the data is not normally distributed. Some tests, like the **Shapiro-Wilk test**, are more powerful than others. However, no test is perfect, and their power depends on both the sample size and the degree of deviation from normality.

In some cases, the data may be slightly skewed or kurtotic, but the test lacks sufficient power to detect the deviation, especially with small sample sizes.

```python
from scipy.stats import skew, kurtosis, shapiro

# Generate a sample with slight skewness and kurtosis
skewed_data = np.random.gamma(2, 2, 1000)

# Check skewness and kurtosis
print(f"Skewness: {skew(skewed_data)}, Kurtosis: {kurtosis(skewed_data)}")

# Perform Shapiro-Wilk test
stat, p_value = shapiro(skewed_data)
print(f"Shapiro-Wilk Test: p-value={p_value}")
```

## Handling Skewed Distributions

Many real-world datasets are not normally distributed and may exhibit skewness. In such cases, normality tests may reject the null hypothesis even when the data is still suitable for many parametric tests, such as $t$-tests or ANOVA, due to their robustness to moderate deviations from normality.

Financial data, biological measurements, and income distributions often have long tails or skewness that do not strictly adhere to the normal distribution assumption, yet many parametric techniques can still be applied with reasonable confidence.

```python
# Generate a sample with positive skew
income_data = np.random.exponential(scale=50000, size=1000)

# Perform Shapiro-Wilk test
stat, p_value = shapiro(income_data)
print(f"Shapiro-Wilk Test on Skewed Data: p-value={p_value}")
```

Despite the skewness of the data, many statistical tests remain robust and reliable. Relying purely on normality tests could lead to unnecessary data transformations or rejections of valid methods.

## Practical Considerations in Real-World Data

In practice, real-world data rarely follows a perfect normal distribution. Formal tests can be overly strict, and minor deviations from normality do not always invalidate the use of parametric methods. Statistical methods like the **Central Limit Theorem (CLT)** can help mitigate concerns about normality for large samples by ensuring that the sampling distribution of the mean is approximately normal, even if the data itself is not.

```python
# Generate a highly skewed dataset
skewed_data = np.random.gamma(2, 2, 100)

# Mean and standard error of the sample mean
mean_sample = np.mean(skewed_data)
std_error = np.std(skewed_data) / np.sqrt(len(skewed_data))

# Central Limit Theorem ensures that the distribution of the sample mean approaches normality
print(f"Sample mean: {mean_sample}, Standard error: {std_error}")
```

The CLT justifies using normal-based methods for large samples, even when the data itself is not normally distributed. Thus, it is important to consider the context and the goal of the analysis before strictly relying on normality tests.

## Conclusion

Normality tests are useful tools for assessing whether data is approximately normal, but they should not be applied blindly. Sample size, test power, and the practical relevance of normality to the analysis are critical considerations. In many cases, particularly with large sample sizes or moderately skewed data, parametric methods remain robust even when normality is not strictly met.
