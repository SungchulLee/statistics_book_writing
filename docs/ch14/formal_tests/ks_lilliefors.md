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
    F_{\text{norm}}(X_{(i)}) = \mathcal{N}\left( \frac{X_{(i)} - \mu}{\sigma} \right)
    $$

    where $\mathcal{N}$ is the standard normal CDF, and $\mu$ and $\sigma$ are the sample mean and standard deviation, respectively.

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

## Exercises

**Exercise 1.**
Explain the difference between the Kolmogorov-Smirnov test and the Lilliefors test. When must you use Lilliefors instead of KS?

??? success "Solution to Exercise 1"
    The **Kolmogorov-Smirnov test** compares the empirical CDF to a fully specified theoretical CDF (all parameters known). For example, testing whether data come from $N(0, 1)$ exactly.

    The **Lilliefors test** is a modification for composite hypotheses where parameters are estimated from the data. For normality testing, you typically estimate $\mu$ and $\sigma$ from the data, then compare to $N(\hat{\mu}, \hat{\sigma}^2)$.

    You must use Lilliefors (not KS) whenever parameters are estimated from the same data being tested. Using KS critical values with estimated parameters is invalid: estimating parameters makes the empirical CDF closer to the theoretical CDF (the fit is optimized), so KS p-values are too large (conservative) and the test loses power.

---

**Exercise 2.**
The KS test statistic is $D_n = \sup_x |F_n(x) - F_0(x)|$. Explain geometrically what this measures.

??? success "Solution to Exercise 2"
    $D_n$ is the maximum vertical distance between the empirical CDF (a step function) and the theoretical CDF (a smooth curve) over all values of $x$. Geometrically, it is the tallest gap between the two curves.

    If the data come from $F_0$, the empirical CDF should closely track the theoretical CDF by the Glivenko-Cantelli theorem, and $D_n$ should be small. A large $D_n$ indicates that at some point in the distribution, the observed data deviate substantially from what the theoretical distribution predicts -- either too many or too few observations in some region.

---

**Exercise 3.**
Why is the KS/Lilliefors test generally less powerful than the Shapiro-Wilk or Anderson-Darling test for detecting non-normality?

??? success "Solution to Exercise 3"
    The KS test uses only the maximum deviation $D_n$, which is a single number summarizing the worst-case discrepancy. This has two drawbacks:

    1. **No tail weighting:** The KS test treats deviations in the center of the distribution (where data are dense) the same as deviations in the tails (where they matter more for normality). The Anderson-Darling test upweights tail deviations.

    2. **Single-point focus:** By using only the supremum, KS ignores the pattern of deviations. The Shapiro-Wilk test uses all order statistics in a correlation-based calculation, extracting more information from the data.

    The KS test was designed as a general goodness-of-fit test (for any distribution), not specifically for normality. Specialized tests like Shapiro-Wilk exploit the specific structure of the normal distribution.

---

**Exercise 4.**
A Lilliefors test on $n = 30$ observations gives $D_n = 0.14$. The critical value at $\alpha = 0.05$ is $0.161$. What is the conclusion?

??? success "Solution to Exercise 4"
    Since $D_n = 0.14 < 0.161$, we fail to reject $H_0$ at $\alpha = 0.05$. The data are consistent with normality according to the Lilliefors test.

    However, with $n = 30$, the test has limited power (especially against subtle alternatives like mild heavy tails). The failure to reject does not prove normality -- it may simply reflect insufficient sample size. Supplementing with a Q-Q plot provides additional visual evidence about the nature and degree of any departures.
