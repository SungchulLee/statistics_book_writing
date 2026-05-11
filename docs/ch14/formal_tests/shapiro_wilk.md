# Shapiro-Wilk Test


## Overview

The Shapiro-Wilk test is a popular method for assessing the normality of a dataset. It evaluates whether the sample data comes from a normally distributed population by calculating a test statistic and corresponding $p$-value.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data is normally distributed.
- **Alternative Hypothesis** ($H_1$): The data is not normally distributed.

## Computation of the Test Statistic W
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

## Deriving the p-Value
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

## Exercises

**Exercise 1.**
The Shapiro-Wilk test statistic $W$ ranges from 0 to 1. Explain what values close to 1 and values far from 1 indicate.

??? success "Solution to Exercise 1"
    The Shapiro-Wilk statistic $W$ measures how well the order statistics match the expected normal order statistics. $W$ close to 1 indicates the data are consistent with normality (the Q-Q plot is approximately linear). $W$ significantly less than 1 indicates departures from normality.

    Formally, $W = (\sum a_i x_{(i)})^2 / \sum(x_i - \bar{x})^2$, where the $a_i$ are optimal coefficients for the normal distribution. The numerator is the squared regression of the order statistics on expected normal scores; the denominator is the total variance. When data are normal, these are nearly equal, giving $W \approx 1$.

---

**Exercise 2.**
A Shapiro-Wilk test on $n = 25$ observations yields $W = 0.94$ with $p = 0.15$. Interpret this result.

??? success "Solution to Exercise 2"
    With $W = 0.94$ and $p = 0.15 > 0.05$, we fail to reject the null hypothesis of normality. The data are consistent with having come from a normal distribution.

    $W = 0.94$ is reasonably close to 1, suggesting only minor departures (if any) from normality. With $n = 25$, the test has moderate power, so large departures would likely have been detected. However, subtle non-normality might be missed.

    As always, supplement with a Q-Q plot for visual assessment.

---

**Exercise 3.**
Why is the Shapiro-Wilk test considered the most powerful normality test for small to moderate sample sizes?

??? success "Solution to Exercise 3"
    The Shapiro-Wilk test achieves high power because:

    1. **Uses all order statistics:** Unlike the KS test (which uses only the maximum deviation), Shapiro-Wilk uses the entire ordered sample in a linear combination, extracting maximum information.
    2. **Optimal weights:** The coefficients $a_i$ are derived from the expected values and covariance matrix of normal order statistics, making the test statistic optimally sensitive to departures from normality.
    3. **Correlation-based:** $W$ is essentially the squared correlation between the ordered data and the expected normal quantiles (the Q-Q correlation). This directly measures "how normal" the data look.

    Simulation studies consistently show Shapiro-Wilk outperforms KS, Lilliefors, and Anderson-Darling for $n < 50$, and remains competitive for larger $n$.

---

**Exercise 4.**
Can the Shapiro-Wilk test distinguish between different types of non-normality (skewness vs. heavy tails)? How can you determine the nature of the departure?

??? success "Solution to Exercise 4"
    The Shapiro-Wilk test is **omnibus**: it detects any departure from normality but does not indicate the type. A small $p$-value tells you the data are non-normal but not whether the issue is skewness, heavy tails, bimodality, or something else.

    To determine the nature of the departure:

    1. **Q-Q plot:** S-shaped curves indicate heavy/light tails; asymmetric curvature indicates skewness; steps or jumps indicate discreteness or rounding.
    2. **Skewness and kurtosis:** Compute sample skewness (asymmetry) and excess kurtosis (tail heaviness) to identify the direction.
    3. **Separate tests:** Run `skewtest` and `kurtosistest` individually to isolate the source.
    4. **Histogram:** Visual inspection can reveal multimodality, which moment-based diagnostics miss.
