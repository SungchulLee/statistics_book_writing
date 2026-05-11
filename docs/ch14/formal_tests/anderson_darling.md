# Anderson-Darling Test


## Overview

The **Anderson-Darling test** is an enhancement of the Kolmogorov-Smirnov test, designed to assess whether a sample comes from a specific distribution, such as the normal distribution. It is particularly sensitive to deviations in the tails, making it especially useful for detecting departures from normality in smaller samples.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data is normally distributed.
- **Alternative Hypothesis** ($H_1$): The data is not normally distributed.

## Computation of the Anderson-Darling Test Statistic

To compute the Anderson-Darling test statistic $A^2$, follow these steps:

1. **Order the Data**: Start with a sample dataset $X_1, X_2, \dots, X_n$ and sort it in ascending order, giving $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$.

2. **Standardize the Data**: Standardize each data point to have a mean of 0 and a variance of 1. Note that this transformation does not impose normality on the data. For each data point $X_i$, calculate the standardized value $Z_i$:

    $$
    Z_i = \frac{X_i - \mu}{\sigma}
    $$

    where $\mu$ is the sample mean and $\sigma$ is the sample standard deviation.

3. **Calculate the Empirical Distribution Function (EDF)**: For each ordered standardized value $Z_{(i)}$, compute the CDF of the normal distribution, $F(Z_{(i)})$.

4. **Compute Test Statistic $A^2$**:

    $$
    A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} \left[ (2i-1) \left( \ln(F(Z_{(i)})) + \ln(1 - F(Z_{(n+1-i)})) \right) \right]
    $$

    where $n$ is the sample size and $F(Z_{(i)})$ is the CDF of the normal distribution at each ordered standardized data point $Z_{(i)}$.

    This formula adjusts for the lower and upper tails, giving the test higher sensitivity to deviations in the distribution tails.

## Deriving the p-Value
After calculating the test statistic $A^2$, it is compared to critical values specific to the Anderson-Darling distribution for the desired distribution type and sample size. Critical values are selected based on the significance level $\alpha$ (e.g., 0.01, 0.05, or 0.10).

- If $A^2$ is greater than the critical value for a given $\alpha$, the $p$-value will be below $\alpha$, leading to rejection of the null hypothesis.
- If $A^2$ is less than the critical value, the $p$-value will be above $\alpha$, indicating insufficient evidence to reject the null hypothesis.

### Decision Rule

- If $A^2 > \text{critical value}$ at significance level $\alpha$, reject $H_0$ (the data is not normally distributed).
- If $A^2 < \text{critical value}$, fail to reject $H_0$ (the data is normally distributed).

## Python Implementation with `stats.anderson`

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

# Perform Anderson-Darling test
result = stats.anderson(data)
statistic = result.statistic
print(f"Anderson-Darling Test: Statistic={statistic}")

# Display critical values
for significance_level, critical_value in zip(result.significance_level, result.critical_values):
    if statistic >= critical_value:
        print(f"At {significance_level}% significance level: Reject H_0. The data is not normally distributed.")
    else:
        print(f"At {significance_level}% significance level: Fail to reject H_0. The data is normally distributed.")
```

---

## Can I Get a p-Value from `stats.anderson`?
The `stats.anderson()` function in SciPy does **not** directly provide a $p$-value; it only returns the test statistic and critical values at specific significance levels.

### Why No Direct p-Value?
The Anderson-Darling test has predefined critical values based on simulation or theoretical distribution tables for each significance level (e.g., 15%, 10%, 5%, 2.5%, and 1% for the normal distribution). Since the Anderson-Darling statistic distribution varies depending on the sample size and the specific distribution being tested, calculating an exact $p$-value is complex.

### Approximate p-Values (for Normality Testing)
If you need an approximate $p$-value for normality testing, there are two options:

**Option 1: Use `statsmodels`**

The `statsmodels` library provides an implementation with an approximate $p$-value:

```python
import numpy as np
from statsmodels.stats.diagnostic import normal_ad

np.random.seed(0)
data = np.random.normal(0, 1, 1000)

statistic, p_value = normal_ad(data)
print(f"Anderson-Darling Test: Statistic={statistic}, p-value={p_value}")
```

**Option 2: Interpret Based on Critical Values**

If you want a rough approximation without additional libraries, you can interpret the $p$-value range based on how the test statistic compares to the provided critical values:

- If the test statistic is **below** the critical value for a certain significance level, the $p$-value is **higher** than that significance level.
- If the test statistic is **above** the critical value for a certain significance level, the $p$-value is **lower** than that significance level.

### Python Implementation with `normal_ad`

```python
import numpy as np
from statsmodels.stats.diagnostic import normal_ad

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

# Perform Anderson-Darling test for normality with p-value
statistic, p_value = normal_ad(data)
print(f"Anderson-Darling Test: Statistic={statistic}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

---

## `stats.anderson` vs `normal_ad`: Which to Choose?

1. **If you need the Anderson-Darling test for general distribution testing** (not limited to normality) and do not need a $p$-value, then **`stats.anderson`** is the better choice. It allows testing for normality, exponential, Weibull, logistic, and extreme value distributions and provides critical values for each.

2. **If your goal is specifically normality testing and you need a $p$-value**, then **`normal_ad` from `statsmodels`** is ideal. This implementation is designed specifically for normality testing and includes an approximate $p$-value, making it easier to interpret results in a traditional hypothesis testing framework.

### Recommendation

If your focus is **normality testing** with a **clear $p$-value interpretation**, use `normal_ad`. For flexibility to test against multiple distributions, use `stats.anderson` and interpret results based on the provided critical values.

## Exercises

**Exercise 1.**
The Anderson-Darling test gives more weight to the tails of the distribution than the Kolmogorov-Smirnov test. Explain how the weighting function achieves this.

??? success "Solution to Exercise 1"
    The Anderson-Darling statistic is:

    $$
    A^2 = -n - \frac{1}{n}\sum_{i=1}^n (2i-1)[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))]
    $$

    This can be expressed as a weighted integral of the squared difference between the empirical and theoretical CDFs: $A^2 = n\int_{-\infty}^{\infty} \frac{[F_n(x) - F(x)]^2}{F(x)(1-F(x))}dF(x)$.

    The weighting function $w(x) = 1/[F(x)(1-F(x))]$ is large in the tails (where $F(x)$ is near 0 or 1) and small in the center. This gives the Anderson-Darling test higher sensitivity to tail departures than the KS test (which weights all parts of the distribution equally), making it better at detecting heavy-tailed or light-tailed alternatives.

---

**Exercise 2.**
An Anderson-Darling test on 50 observations yields $A^2 = 0.85$. Using critical values $0.631$ (10%), $0.752$ (5%), $1.035$ (1%), determine the conclusion at $\alpha = 0.05$.

??? success "Solution to Exercise 2"
    Since $A^2 = 0.85 > 0.752$ (the 5% critical value), we reject $H_0$ (normality) at the 5% significance level. However, $A^2 = 0.85 < 1.035$ (the 1% critical value), so we would not reject at the 1% level.

    Conclusion: there is evidence of non-normality at the 5% level ($0.01 < p < 0.05$). Visual inspection (Q-Q plot) should be used to identify the nature of the departure.

---

**Exercise 3.**
Compare the power of the Anderson-Darling, Shapiro-Wilk, and Kolmogorov-Smirnov tests for detecting departures from normality.

??? success "Solution to Exercise 3"
    Simulation studies generally show:

    1. **Shapiro-Wilk** has the highest power overall, especially for small to moderate $n$ and for a wide range of alternatives (skewed, heavy-tailed, light-tailed).
    2. **Anderson-Darling** has nearly comparable power to Shapiro-Wilk and is superior to KS. Its tail-weighting makes it particularly good at detecting heavy-tailed alternatives.
    3. **Kolmogorov-Smirnov (Lilliefors)** has the lowest power because it uses the maximum deviation without weighting, giving equal attention to the center (where departures are small) and the tails (where they matter most).

    Recommendation: use Shapiro-Wilk as the primary test, Anderson-Darling as an alternative (especially for larger $n$), and KS only when testing against a fully specified distribution (not for composite normality testing).

---

**Exercise 4.**
The Anderson-Darling test can be adapted for distributions other than the normal (e.g., exponential, Weibull). Explain the general principle.

??? success "Solution to Exercise 4"
    The Anderson-Darling test is a general goodness-of-fit test: it compares the empirical CDF to any specified theoretical CDF $F_0(x)$. For normality testing, $F_0 = \mathcal{N}((x-\hat{\mu})/\hat{\sigma})$ with estimated parameters.

    To test whether data follow an exponential distribution: use $F_0(x) = 1 - e^{-x/\hat{\lambda}}$. For Weibull: use the Weibull CDF with estimated shape and scale parameters.

    The test statistic formula is the same in all cases; only $F_0$ changes. Critical values differ by distribution family because the null distribution of $A^2$ depends on the number of estimated parameters and the shape of the reference distribution. Specialized critical value tables or simulation-based p-values are used for each family.
