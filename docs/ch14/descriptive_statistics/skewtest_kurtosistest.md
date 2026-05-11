# Skewtest and Kurtosistest


## Skewtest

The **skewness test**, provided by `scipy.stats.skewtest()`, is a formal statistical test that evaluates whether the skewness of a dataset significantly deviates from zero, indicating whether the data is symmetric or not.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data has zero skewness (is symmetrically distributed).
- **Alternative Hypothesis** ($H_1$): The data has non-zero skewness (is asymmetrical).

The skewness test provides a test statistic and a $p$-value. If the $p$-value is small (typically less than 0.05), we reject the null hypothesis and conclude that the data is not symmetrically distributed.

### Step-by-Step Explanation

1. **Calculate the sample skewness**: We first compute the sample skewness using the formula

    $$
    \text{Skewness} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^3
    $$

    where $n$ is the number of data points, $x_i$ are the individual data points, $\bar{x}$ is the mean, and $\sigma$ is the standard deviation.

2. **Compute the z-score**: The test statistic, or **z-score**, is computed by dividing the observed skewness by its standard error. The z-score tells us how far the skewness deviates from the expected value (zero for a normal distribution).

3. **Convert z-score to p-value**: We compute the p-value using the cumulative distribution function (CDF) of the standard normal distribution. We compare the z-score obtained in the previous step to the standard normal distribution to get the **two-tailed p-value**. This p-value quantifies how likely it is to observe a skewness as extreme as the observed value under the null hypothesis of zero skewness (i.e., symmetric distribution).

### Applications

The skewness test is practical when assessing whether the data can be analyzed using parametric statistical methods that assume symmetry (such as specific versions of the $t$-test or ANOVA). If we conclude that the data is skewed, it may be necessary to transform the data (e.g., using a log or Box-Cox transformation) or use non-parametric methods that do not assume symmetry.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset with skewness
# data = np.random.gamma(2, 2, 1000)
data = np.random.normal(2, 2, 1000)

# Calculate skewness
skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")

# Perform skewness test
stat, p_value = stats.skewtest(data)
print(f"Skewness Test: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not symmetrically distributed (significant skewness).")
else:
    print("Fail to reject H_0: The data is symmetrically distributed (no significant skewness).")
```

---

## Kurtosistest

The **kurtosis test**, provided by `scipy.stats.kurtosistest()`, is a formal statistical test that evaluates whether the excess kurtosis of a dataset significantly deviates from the kurtosis of a normal distribution.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data has normal kurtosis.
- **Alternative Hypothesis** ($H_1$): The data does not have normal kurtosis.

The kurtosis test provides a test statistic and a $p$-value. If the $p$-value is small (typically less than 0.05), we reject the null hypothesis and conclude that the data does not have normal kurtosis.

### Step-by-Step Explanation

1. **Calculate the sample excess kurtosis**: We first compute the sample excess kurtosis using the formula

    $$
    \text{Excess Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4 - 3
    $$

    where $n$ is the number of data points, $x_i$ are the individual data points, $\bar{x}$ is the mean, and $\sigma$ is the standard deviation.

2. **Compute the z-score**: The test statistic, or **z-score**, is computed by dividing the observed excess kurtosis by its standard error. The z-score tells us how far the excess kurtosis deviates from the expected value (zero for a normal distribution).

3. **Convert z-score to p-value**: We compute the p-value using the CDF of the standard normal distribution to get the **two-tailed p-value**. This p-value quantifies how likely it is to observe an excess kurtosis as extreme as the observed value under the null hypothesis.

### Applications

We use the kurtosis test when assessing the suitability of parametric methods (such as the $t$-test or ANOVA), which assume normal kurtosis. If the kurtosis test indicates that the data has significantly heavy or light tails, transformations (e.g., log transformation or Box-Cox transformation) or non-parametric methods may be needed.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.gamma(2, 2, 1000)
data = np.random.normal(2, 2, 1000)

# Calculate kurtosis
kurtosis_value = stats.kurtosis(data)
print(f"Kurtosis: {kurtosis_value:.4f}")

# Perform kurtosis test
stat, p_value = stats.kurtosistest(data)
print(f"Kurtosis Test: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data does not have normal kurtosis.")
else:
    print("Fail to reject H_0: The data has normal kurtosis.")
```

## Exercises

**Exercise 1.**
Using `scipy.stats.skewtest`, a researcher obtains a z-score of 3.2 and p-value of 0.001. Interpret this result.

??? success "Solution to Exercise 1"
    The skewtest tests $H_0$: the population skewness is zero (consistent with normality). The z-score of 3.2 is far from zero, and $p = 0.001$ is well below any conventional significance level.

    Interpretation: there is strong evidence that the data come from a skewed (non-symmetric) distribution. The positive z-score indicates right-skewness (the right tail is heavier than the left). This is one component of non-normality; the data may require transformation or nonparametric methods.

---

**Exercise 2.**
Explain the difference between `skewtest` and simply computing the sample skewness. Why is a formal test necessary?

??? success "Solution to Exercise 2"
    The sample skewness $g_1 = \frac{m_3}{m_2^{3/2}}$ is a point estimate that is always nonzero (even for truly normal data) due to sampling variability. Without a test, there is no way to judge whether the observed skewness is statistically significant or just sampling noise.

    `skewtest` transforms $g_1$ into a z-statistic using D'Agostino's transformation, which accounts for the sampling distribution of skewness under normality. The p-value then quantifies whether the observed skewness is unlikely under the null. For example, $g_1 = 0.3$ might be significant with $n = 500$ but not with $n = 20$.

---

**Exercise 3.**
The `kurtosistest` has a minimum sample size requirement ($n \geq 20$). Explain why small samples are problematic for kurtosis estimation.

??? success "Solution to Exercise 3"
    Sample kurtosis involves the fourth power of deviations $(x_i - \bar{x})^4$, making it extremely sensitive to individual observations. In small samples:

    1. **High variance:** The sampling distribution of the kurtosis estimator is very wide, meaning the estimate is unreliable.
    2. **Bias:** The sample kurtosis is biased in small samples, and bias-correction formulas have large uncertainty.
    3. **Non-normality of the estimator:** The z-transformation used in `kurtosistest` assumes $n$ is large enough for the transformation to be approximately standard normal. Below $n \approx 20$, this approximation fails.

    For small samples, visual methods (Q-Q plots) and the Shapiro-Wilk test are more reliable than moment-based tests.

---

**Exercise 4.**
If `skewtest` gives $p = 0.15$ and `kurtosistest` gives $p = 0.03$, what can you conclude about the nature of the non-normality?

??? success "Solution to Exercise 4"
    The skewtest does not reject ($p = 0.15$), suggesting the data are approximately symmetric. The kurtosistest rejects ($p = 0.03$), indicating the data have abnormal tail behavior (likely heavy or light tails).

    This pattern suggests the distribution is **symmetric but leptokurtic** (heavy-tailed) or **symmetric but platykurtic** (light-tailed). Examples include the $t$-distribution with small $\nu$ (symmetric, heavy-tailed) or the uniform distribution (symmetric, light-tailed).

    The practical implication depends on the direction: heavy tails (positive excess kurtosis) mean more outliers than expected, affecting mean-based inference. Light tails (negative excess kurtosis) are generally less problematic for standard methods.
