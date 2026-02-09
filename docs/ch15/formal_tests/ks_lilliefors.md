# Kolmogorov-Smirnov Test and Lilliefors Test

While graphical methods and descriptive statistics provide insight into data distribution, formal statistical tests offer more rigorous methods for assessing normality. These tests evaluate whether the observed data significantly deviates from the expected normal distribution, providing a statistical basis for the decision.

## Kolmogorov-Smirnov Test

The **Kolmogorov-Smirnov (K-S) test** is a non-parametric test used to compare the empirical cumulative distribution function (ECDF) of the sample data to the cumulative distribution function (CDF) of a reference distribution (in this case, the normal distribution). The test is sensitive to discrepancies in both the distributions' central tendency and overall shape (variance).

### Hypotheses

- **Null Hypothesis** ($H_0$): The data follows the specified distribution (normal distribution).
- **Alternative Hypothesis** ($H_1$): The data does not follow the specified distribution.

### Computation of the K-S Test Statistic

We compute the Kolmogorov-Smirnov test statistic $D$ based on the largest absolute difference between the empirical CDF of the sample and the CDF of the reference distribution. The steps are:

1. **Order the Data**: Let $X_1, X_2, \dots, X_n$ be the sample data sorted in ascending order such that $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$.

2. **Compute the Empirical CDF**: For each ordered data point $X_{(i)}$, define the ECDF as the proportion of data points less than or equal to $X_{(i)}$:

    $$
    F_{\text{emp}}(X_{(i)}) = \frac{i}{n}
    $$

    where $n$ is the total sample size and $i$ is the rank of the data point.

3. **Compute the Theoretical CDF**: The theoretical CDF for the normal distribution $F_{\text{norm}}(X_{(i)})$ evaluated at $X_{(i)}$ is:

    $$
    F_{\text{norm}}(X_{(i)}) = \Phi\left( \frac{X_{(i)} - \mu}{\sigma} \right)
    $$

    where $\Phi$ is the standard normal CDF, and $\mu$ and $\sigma$ are the sample mean and standard deviation, respectively.

4. **Test Statistic $D$**: The K-S test statistic $D$ is the maximum absolute difference between the empirical CDF and the theoretical CDF at any sample point:

    $$
    D = \max_i \left| F_{\text{emp}}(X_{(i)}) - F_{\text{norm}}(X_{(i)}) \right|
    $$

    In other words, $D$ measures the largest vertical distance between the two CDFs over the data range.

### Decision Rule

- If $D$ exceeds the critical value for the chosen significance level $\alpha$, reject $H_0$ (the data does not follow the normal distribution).
- If $D$ is less than the critical value, fail to reject $H_0$ (the data follows the normal distribution).

The K-S test is effective for detecting differences in the central location and shape of the distribution. However, it is less sensitive to deviations in the tails compared to tests like the Anderson-Darling test.

### Python Implementation

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

data_ks = (data - data.mean()) / data.std()

# Perform Kolmogorov-Smirnov test
stat, p_value = stats.kstest(data_ks, 'norm')
print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

---

## Kolmogorov-Smirnov Test vs Lilliefors Test

The `stats.kstest` function in SciPy performs the **Kolmogorov-Smirnov (K-S) test**, not the **Lilliefors test**. Here is how they differ:

### Kolmogorov-Smirnov Test (`stats.kstest`)

- **Purpose**: General **goodness-of-fit test**, comparing a sample to a known distribution with **fixed parameters**.
- **Parameter Assumptions**: Assumes that the distribution parameters (e.g., mean and standard deviation) are known **a priori**.
- **Usage**: Appropriate if you want to see if a dataset fits a specific distribution with predefined parameters.

### Lilliefors Test (`statsmodels.stats.diagnostic.lilliefors`)

- **Purpose**: Modified K-S test used for **normality testing** when the population parameters (mean and standard deviation) are **unknown and estimated from the sample**.
- **Parameter Assumptions**: Adjusts for the fact that the parameters are estimated from the sample, providing different critical values tailored for this situation.
- **Availability**: Not available in SciPy; provided by `statsmodels`.

### Comparison

| Feature                   | `stats.kstest` (K-S Test)                        | Lilliefors Test                                 |
|---------------------------|--------------------------------------------------|-------------------------------------------------|
| **Purpose**               | General goodness-of-fit (any distribution)       | Normality test when parameters are unknown      |
| **Parameter Knowledge**   | Assumes parameters are known                     | Assumes parameters are unknown                  |
| **Parameter Estimation**  | Not designed for estimated parameters            | Adjusted for estimated parameters               |
| **SciPy Implementation**  | Yes, as `stats.kstest`                           | No direct implementation in SciPy               |

### Python Implementation

```python
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

data_ks = (data - data.mean()) / data.std()

# Perform Kolmogorov-Smirnov test
stat, p_value = stats.kstest(data_ks, 'norm')
print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")

stat, p_value = lilliefors(data)
print(f"Lilliefors Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

---

## Which Test to Choose?

### When to Choose `stats.kstest` (Kolmogorov-Smirnov Test)

- **Use Case**: General goodness-of-fit test when comparing a sample to a known theoretical distribution with **fixed parameters**.
- **Flexibility**: Can test against any distribution (e.g., normal, exponential) as long as the distribution's parameters are specified beforehand.
- **Advantage**: Broadly applicable for comparing empirical data to many theoretical distributions beyond normality.
- **Limitation**: Inaccurate if used for normality testing with estimated parameters, as it does not adjust for parameter estimation.

### When to Choose `statsmodels.stats.diagnostic.lilliefors` (Lilliefors Test)

- **Use Case**: Specifically designed for **normality testing** when distribution parameters are **unknown and estimated from the sample**.
- **Flexibility**: Limited to normality testing, but highly accurate in that context.
- **Advantage**: Offers a more accurate approach to normality testing when parameters are estimated.
- **Limitation**: Limited to normality testing; not applicable for testing other distributions.

### Recommendation

- For **general distribution testing** (e.g., checking if data fits a specific distribution like exponential or Weibull with fixed parameters), **`stats.kstest`** is the better choice.
- For **normality testing with unknown parameters**, **`lilliefors` from `statsmodels`** is preferred because it provides a more accurate assessment, accounting for the parameter estimation process.
