# Shapiro-Wilk Test

## Overview

The Shapiro-Wilk test is a popular method for assessing the normality of a dataset. It evaluates whether the sample data comes from a normally distributed population by calculating a test statistic and corresponding $p$-value.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data is normally distributed.
- **Alternative Hypothesis** ($H_1$): The data is not normally distributed.

## Computation of the Test Statistic $W$

We compute the Shapiro-Wilk test statistic $W$ using the following steps:

1. **Order the Data**: Let $X_1, X_2, \dots, X_n$ be the sample data sorted in ascending order such that $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$.

2. **Expected Values**: Calculate the expected values $m_i$ for a sample of size $n$ from a standard normal distribution. These values represent the means of the order statistics. Let $\mathbf{m}^T = [m_1, m_2, \dots, m_n]$ be the vector of these expected values.

3. **Covariance Matrix**: We use a covariance matrix $\Sigma$ of the order statistics from the normal distribution to generate weights $a_i$. These weights are computed to optimize the sensitivity of the test to departures from normality. The vector of weights is denoted by $\mathbf{a}^T = [a_1, a_2, \dots, a_n]$:

    $$
    [a_1, a_2, \dots, a_n] = \frac{[m_1, m_2, \dots, m_n]\Sigma^{-1}}{\sqrt{[m_1, m_2, \dots, m_n]\Sigma^{-1}\Sigma^{-1}[m_1, m_2, \dots, m_n]^T}}
    $$

4. **Test Statistic $W$**: Then, we compute the test statistic $W$ as:

    $$
    W = \frac{\left( \sum_{i=1}^{n} a_i X_{(i)} \right)^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
    $$

    where

    - $X_{(i)}$ is the $i$-th ordered data point,
    - $a_i$ is the corresponding weight from the vector $\mathbf{a}$,
    - $\bar{X}$ is the sample mean.

    The numerator represents the squared linear combination of the ordered sample, and the denominator is the total variance of the sample data.

## Deriving the $p$-Value

Once we compute the test statistic $W$, the $p$-value is obtained by comparing $W$ to the distribution of $W$ under the null hypothesis of normality. The $p$-value represents the probability of observing a test statistic as extreme as $W$ under the assumption that the null hypothesis is true.

- We reject the null hypothesis if the $p$-value is small (e.g., less than $\alpha = 0.05$). This suggests the data does not follow a normal distribution.
- If the $p$-value is large (greater than or equal to $\alpha$), we do not reject the null hypothesis.

### Decision Rule

- If $p$-value $\leq \alpha$, reject $H_0$ (the data is not normally distributed).
- If $p$-value $> \alpha$, fail to reject $H_0$ (the data is normally distributed).

## Python Implementation

```python
import numpy as np
from scipy import stats

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

# Perform Shapiro-Wilk test
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

In summary, the Shapiro-Wilk test uses the ordered sample data and precomputed weights to compute the test statistic $W$ to determine whether the data is likely to have come from a normal distribution. It is generally considered one of the most powerful normality tests, particularly for small to moderate sample sizes.
