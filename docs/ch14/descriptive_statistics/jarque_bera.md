# Jarque-Bera Test


## Overview

The Jarque-Bera test is a statistical test used to assess whether a dataset follows a normal distribution by evaluating two key features of the data: skewness and kurtosis. It is widely used in econometrics and financial applications, where normality is a crucial assumption for many statistical models and methods.

### Hypotheses

- **Null Hypothesis** ($H_0$): The data is normally distributed.
- **Alternative Hypothesis** ($H_1$): The data is not normally distributed.

## Computation of the Jarque-Bera Test Statistic

We compute the test statistic $JB$ using the following formula:

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
$$

where

- $n$ is the sample size,
- $S$ is the sample skewness,
- $K$ is the sample kurtosis,
- The factor 6 in the denominator is a scaling constant to normalize the contribution of skewness and kurtosis to the statistic.

### Steps to Compute the Test Statistic

1. **Compute the Sample Mean**:

    $$
    \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
    $$

2. **Compute the Sample Skewness ($S$)**:

    $$
    S = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{X_i - \bar{X}}{\sigma} \right)^3
    $$

    where $\sigma^2$ is the sample variance:

    $$
    \sigma^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2
    $$

3. **Compute the Sample Kurtosis ($K$)**:

    $$
    K = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{X_i - \bar{X}}{\sigma} \right)^4
    $$

4. **Calculate the Jarque-Bera Statistic ($JB$)**:

    $$
    JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
    $$

    This statistic combines the squared skewness and the squared deviation of the kurtosis from 3 (the kurtosis of a normal distribution), weighted by sample size $n$.

## Deriving the p-Value
The Jarque-Bera test statistic $JB$ follows a chi-square ($\chi^2$) distribution with 2 degrees of freedom under the null hypothesis (since we compute it based on two components: skewness and kurtosis). To obtain the $p$-value:

1. Compare the computed $JB$ statistic to the critical values of the chi-square distribution with 2 degrees of freedom.

2. The **$p$-value** is the probability that the test statistic $JB$ would be as extreme as or more extreme than the observed value, under the assumption that the null hypothesis ($H_0$) is true.

    - A small $p$-value (typically less than a significance level $\alpha = 0.05$) suggests that the data deviates significantly from normality, and we reject the null hypothesis.
    - A large $p$-value suggests insufficient evidence to reject the null hypothesis, meaning the data could reasonably come from a normal distribution.

### Decision Rule

- If the $p$-value is less than the significance level $\alpha$ (e.g., 0.05), reject $H_0$ and conclude that the data is not normally distributed.
- If the $p$-value is greater than or equal to $\alpha$, fail to reject $H_0$ and conclude that the data may be normally distributed.

### Interpretation

- A high **$JB$ statistic** implies that the data has skewness or kurtosis (or both) that deviates significantly from a normal distribution.
- A **low $JB$ statistic** indicates that the sample data's skewness and kurtosis are consistent with a normal distribution.

## Python Implementation

```python
import numpy as np
from scipy import stats

np.random.seed(0)

n = 1000

# Generate a sample dataset
data = np.random.normal(0, 1, n)
# data = np.random.exponential(1, n)

skewness_value = stats.skew(data)
kurtosis_value = stats.kurtosis(data)
JB = n / 6 * (skewness_value**2 + kurtosis_value**2 / 4)
print(f"{JB = }")

# Perform Jarque-Bera test
stat, p_value = stats.jarque_bera(data)
print(f"Jarque-Bera Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

---

## Jarque-Bera Test vs D'Agostino's K-Squared Test

The Jarque-Bera test is **not** an approximation of D'Agostino's K-squared test. While both tests assess normality by examining skewness and kurtosis, they are fundamentally different in how they compute the test statistics.

### Key Differences

**Jarque-Bera Test**:

- Directly uses the sample's **skewness** and **kurtosis** to compute its test statistic.
- The test statistic is:

    $$
    JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
    $$

- Combines skewness and kurtosis into a single test statistic, assuming the sample follows a chi-square distribution with 2 degrees of freedom under the null hypothesis.

**D'Agostino's K-Squared Test**:

- Transforms both skewness and kurtosis into independent **Z-scores**:
    - **$Z_{\text{skewness}}$**: A transformation that normalizes the sample's skewness.
    - **$Z_{\text{kurtosis}}$**: A transformation that normalizes the sample's kurtosis.
- The test statistic is:

    $$
    K^2 = Z_{\text{skewness}}^2 + Z_{\text{kurtosis}}^2
    $$

- Like the Jarque-Bera test, $K^2$ follows a chi-square distribution with 2 degrees of freedom, but D'Agostino's test uses separate transformations for skewness and kurtosis, which makes it more robust and sensitive to deviations from normality.

### Why They Are Distinct

- **Different Approaches**: The Jarque-Bera test applies a simple, direct formula based on the raw skewness and kurtosis values, whereas D'Agostino's K-squared test applies transformations that adjust for sample size and normalize the distribution of skewness and kurtosis.

- **Test Statistic Construction**: Jarque-Bera straightforwardly combines skewness and kurtosis into a single statistic, while D'Agostino's test separates them, transforming them into individual test statistics, which are then squared and summed.

- **Sensitivity**: D'Agostino's K-squared test is more sensitive to deviations from normality than the Jarque-Bera test, especially in larger sample sizes. The Z-transformations make it more accurate when the sample size increases, whereas Jarque-Bera's performance may degrade in small samples or be less sensitive to tail deviations.

### Conclusion

The Jarque-Bera test does not approximate D'Agostino's K-squared test; they are distinct methods for testing normality with different underlying statistical foundations. Both tests use skewness and kurtosis, but D'Agostino's test is considered more robust due to its transformation-based approach, while Jarque-Bera is more straightforward and commonly used in econometrics.

## Exercises

**Exercise 1.**
The test statistic $JB$ for testing the normality of data using the Jarque-Bera test is defined as:

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2_2
$$

where:

- $n$: Sample size
- $S$: Sample skewness
- $K$: Sample kurtosis
- The denominator $6$ is a scaling constant that normalizes the contributions of skewness and kurtosis.

**(a)** Find the theoretical value of $S$ for a normal distribution.

**(b)** Evaluate the integral $\int_{-\infty}^\infty x^4 e^{-x^2} dx$.

**(c)** Why is $3$ subtracted from $K$ when calculating $JB$?

**(d)** If $JB = 3.2189$, express the $p$-value using the cumulative distribution function (CDF) $F$ of $\chi^2_2$.

**(e)** If the $p$-value is 0.2 and the significance level ($\alpha$) is 5%, what is the conclusion of the test?

??? success "Solution to Exercise 1"

    **(a)** For a normal distribution, the theoretical skewness $S$ is:

    $$
    S = \mathbb{E}\left(\frac{X - \mu}{\sigma}\right)^3 = \mathbb{E}Z^3 = \int_{-\infty}^\infty x^3 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 0
    $$

    **(b)**

    $$
    I
    = \int_{-\infty}^\infty x^4 e^{-x^2} dx
    = 2 \int_0^\infty x^4 e^{-x^2} dx
    $$

    Let $u = x^2$, so that $x = u^{1/2}$ and $dx = \frac{1}{2} u^{-1/2} du$. The integral becomes:

    $$
    I
    = 2 \int_0^\infty (u^{1/2})^4 e^{-u} \cdot \frac{1}{2} u^{-1/2} du
    = \int_0^\infty u^{\frac{5}{2}-1} e^{-u} du
    = \Gamma\left(\frac{5}{2}\right)
    = \frac{3}{2}\cdot\Gamma\left(\frac{3}{2}\right)
    = \frac{3}{2}\cdot\frac{1}{2}\cdot\Gamma\left(\frac{1}{2}\right)
    = \frac{3}{2}\cdot\frac{1}{2}\cdot\sqrt{\pi}
    $$

    where the Gamma function is defined as:

    $$
    \Gamma(n) = \int_0^\infty x^{n-1} e^{-x} dx
    $$

    The Gamma function satisfies:

    $$
    \Gamma(n+1) = n \cdot \Gamma(n),\quad
    \Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}
    $$

    **(c)** For a normal distribution, the theoretical kurtosis $K$ is $3$. Subtracting $3$ ensures that for normal data, $K - 3 = 0$. This simplifies the test statistic under the null hypothesis of normality.

    **(d)** The $p$-value is:

    $$
    p = 1 - F(3.2189)
    $$

    **(e)** Since $p = 0.2 > \alpha = 0.05$, there is insufficient evidence to reject the null hypothesis. Therefore, we conclude that the data does not violate normality and maintain the assumption of normality at the given significance level.
