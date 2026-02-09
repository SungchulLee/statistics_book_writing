# D'Agostino's K-Squared Test

## Overview

**D'Agostino's K-squared test** is a formal statistical test used to evaluate whether a given sample follows a normal distribution by combining two key measures: **skewness** and **kurtosis**. The test examines both the asymmetry (skewness) and the "tailedness" (kurtosis) of the data distribution and computes a combined test statistic that assesses the overall deviation from normality. It is especially useful when considering skewness and kurtosis in a single test.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data is normally distributed.
- **Alternative Hypothesis** ($H_1$): The data is not normally distributed.

The test statistic combines the z-scores of the skewness and kurtosis into a single value. If this test statistic is significant and the associated $p$-value is small (below a given significance level, typically 0.05), we reject the null hypothesis and conclude that the data is not normally distributed.

## How D'Agostino's K-Squared Test Works

1. **Calculate Skewness**: The test first measures the sample's skewness, which refers to the asymmetry of the distribution. For a perfectly normal distribution, skewness should be close to zero.

2. **Calculate Kurtosis**: Next, the test measures kurtosis. Kurtosis describes the "tailedness" of the distribution. A normal distribution has a kurtosis of 3 (mesokurtic).

3. **Combine the Measures**: We compute the z-scores $Z_{\text{skewness}}$ and $Z_{\text{kurtosis}}$ of both the skewness and kurtosis, and the test statistic $K^2$ is formed by summing the squares of these z-scores:

    $$
    K^2 = Z_{\text{skewness}}^2 + Z_{\text{kurtosis}}^2
    $$

    This test statistic follows a chi-squared distribution with 2 degrees of freedom.

4. **Compute the p-value**: Based on the value of $K^2$, the test computes a $p$-value, which tells us the probability of observing such a deviation from normality under the null hypothesis.

## How to Compute $Z_{\text{skewness}}$

The computation of $Z_{\text{skewness}}$ involves standardizing the skewness of a sample to test whether it significantly deviates from zero under the null hypothesis of normality.

Given a sample of size $n$:

**1. Compute the sample skewness $S$:**

$$
S = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}
$$

where $x_i$ is the $i$-th observation, $\bar{x}$ is the sample mean, the numerator measures the third moment (asymmetry), and the denominator normalizes it to make it dimensionless.

**2. Standardize $S$:**

2.1 Compute the standard error of skewness $\text{SE}_{S}$:

$$
\text{SE}_{S} = \sqrt{\frac{6n (n-1)}{(n-2)(n+1)(n+3)}}
$$

2.2 Standardize $S$ to obtain $Z_{\text{skewness}}$:

$$
Z_{\text{skewness}} = \frac{S}{\text{SE}_{S}}
$$

**Key Notes:**

- $Z_{\text{skewness}}$ follows a standard normal distribution ($N(0, 1)$) under the null hypothesis of normality.
- If $|Z_{\text{skewness}}|$ is large, it indicates significant departure from normality due to skewness.

## How to Compute $Z_{\text{kurtosis}}$

Given a sample of size $n$:

**1. Compute the sample kurtosis $K$:**

$$
K = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}{\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^2}
$$

where $x_i$ is the $i$-th observation, $\bar{x}$ is the sample mean, the numerator measures the fourth moment (tailedness), and the denominator normalizes it.

**2. Adjust $K$ to excess kurtosis:**

$$
K_{\text{excess}} = K - 3
$$

**3. Compute the standard error of kurtosis $\text{SE}_{K}$:**

$$
\text{SE}_{K} = \sqrt{\frac{24n(n-1)^2}{(n-3)(n-2)(n+3)(n+5)}}
$$

**4. Standardize $K_{\text{excess}}$:**

$$
Z_{\text{kurtosis}} = \frac{K_{\text{excess}}}{\text{SE}_{K}}
$$

**Key Notes:**

- $Z_{\text{kurtosis}}$ follows a standard normal distribution ($N(0, 1)$) under the null hypothesis of normality.
- If $|Z_{\text{kurtosis}}|$ is large, it indicates significant departure from normality due to kurtosis.

## Python Implementation

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
data = np.random.normal(0, 1, 1000)

Z_skewtest, p_value = stats.skewtest(data)
Z_kurtosistest, p_value = stats.kurtosistest(data)
print(f"{Z_skewtest**2 + Z_kurtosistest**2 = }")

# Perform D'Agostino's K-squared test
stat, p_value = stats.normaltest(data)

print(f"D'Agostino's K-squared Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value > alpha:
    print("Fail to reject H_0: The data is normally distributed.")
else:
    print("Reject H_0: The data is not normally distributed.")
```

## Applications

D'Agostino's K-squared test is commonly used in situations where the assumption of normality is critical, such as in:

- **Parametric statistical tests** (e.g., $t$-tests, ANOVA) that require normally distributed data.
- **Regression analysis**, where the normality of residuals is assumed.
- **Quality control** and **financial modeling**, where normality is often assumed in modeling and decision-making processes.

## Limitations

- **Sample Size Sensitivity**: Like many normality tests, D'Agostino's K-squared test is sensitive to sample size. For small sample sizes, the test might not have enough power to detect deviations from normality. For large samples, even slight deviations from normality may result in rejecting the null hypothesis.
- **Assumes Continuous Data**: The test is designed for continuous data. Applying it to categorical or ordinal data is not appropriate.

D'Agostino's K-squared test is a powerful method for checking whether data is normally distributed, accounting for skewness and kurtosis. By combining these two important aspects of distribution shape, the test provides a robust assessment of normality. However, like all normality tests, it should be used with graphical methods (e.g., Q-Q plots) and other statistical tests to understand the data's distribution comprehensively.
