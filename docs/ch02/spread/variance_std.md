# Variance and Standard Deviation

## Overview

Variance and standard deviation are the most widely used measures of statistical dispersion. They quantify how much individual data points deviate from the mean, providing essential information about the spread and consistency of a dataset.

---

## 1. Variance

### Definition

Variance measures the average of the squared deviations from the mean. By squaring, it ensures all deviations contribute positively and penalizes larger deviations more heavily.

### Formulas

**Population variance:**

$$
\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}
$$

**Sample variance (with Bessel's correction):**

$$
s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}
$$

The denominator $n - 1$ corrects for the downward bias that arises from using the sample mean $\bar{x}$ instead of the true population mean $\mu$.

### Example

For the dataset 70, 85, 90, 95, 100 with mean $\bar{x} = 88$:

1. Squared deviations: $(70-88)^2 = 324$, $(85-88)^2 = 9$, $(90-88)^2 = 4$, $(95-88)^2 = 49$, $(100-88)^2 = 144$
2. Sum: $324 + 9 + 4 + 49 + 144 = 530$
3. Sample variance: $s^2 = 530 / 4 = 132.5$

### Computing Variance in Python

```python
import numpy as np

sample_data = np.array([1.5, 2.5, 4, 2, 1, 1])

# Population variance (ddof=0, the default)
population_variance = sample_data.var()
print(f"Population Variance (ddof=0): {population_variance}")

# Sample variance (ddof=1)
sample_variance = sample_data.var(ddof=1)
print(f"Sample Variance (ddof=1): {sample_variance}")
```

### Interpretation

A variance of 132.5 means the exam scores vary, on average, by a squared distance of 132.5 units from the mean. Because variance is expressed in squared units, it can be difficult to interpret directly—which is why the standard deviation is often preferred.

---

## 2. Standard Deviation

### Definition

The standard deviation is the square root of the variance. It returns the measure of spread to the original units of the data, making it directly interpretable.

### Formulas

**Population standard deviation:**

$$
\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}
$$

**Sample standard deviation:**

$$
s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}}
$$

### Example

Using the variance from above: $s = \sqrt{132.5} \approx 11.51$.

This means the exam scores deviate from the mean by about 11.51 points on average.

### Computing Standard Deviation in Python

```python
import numpy as np

sample_data = np.array([1.5, 2.5, 4, 2, 1, 1])

# Population standard deviation (ddof=0)
population_std = sample_data.std()
print(f"Population Standard Deviation (ddof=0): {population_std}")

# Sample standard deviation (ddof=1)
sample_std = sample_data.std(ddof=1)
print(f"Sample Standard Deviation (ddof=1): {sample_std}")
```

### Applications

Standard deviation is used across many domains: assessing the volatility of financial returns, measuring the spread of scientific measurements, evaluating manufacturing consistency, and more. In a normal distribution, approximately 68% of data falls within one standard deviation of the mean, 95% within two, and 99.7% within three (the empirical rule).

---

## 3. Population vs. Sample: The `ddof` Parameter

When computing variance and standard deviation in NumPy and pandas, the `ddof` (delta degrees of freedom) parameter controls the denominator:

| Context | Denominator | `ddof` | Use When |
|---|---|---|---|
| Population | $N$ | 0 | You have the entire population |
| Sample | $n - 1$ | 1 | You have a sample from a larger population |

NumPy defaults to `ddof=0` (population), while pandas defaults to `ddof=1` (sample). Always be explicit about which you are computing.

---

## 4. Practical Considerations

**Data Distribution:** In a normal distribution, standard deviation has a clean interpretation via the empirical rule. In skewed distributions, it may not accurately reflect the typical spread.

**Sensitivity to Outliers:** Both variance and standard deviation are sensitive to extreme values because squaring amplifies large deviations. For skewed data or data with outliers, the IQR is a more robust alternative.

**Real-Life Examples:**

- **Stock Market Volatility:** Standard deviation of returns measures risk. Higher standard deviation means greater price fluctuation and higher investment risk.
- **Student Test Scores:** Low standard deviation indicates most students scored similarly; high standard deviation reveals wide performance variation.
- **Manufacturing Quality:** Standard deviation of product measurements indicates process consistency. Lower values mean tighter quality control.

## Summary

Variance and standard deviation provide a complete picture of data variability by considering every observation's distance from the mean. The standard deviation, being in the original units, is more interpretable and widely used. Understanding the distinction between population and sample formulas—and setting `ddof` correctly—is essential for accurate statistical analysis.

## Exercises

**Exercise 1.**
For the dataset $\{3, 7, 7, 9, 14\}$: (a) compute $\bar{x}$; (b) compute $s^2$ using Bessel's correction; (c) compute $s$; (d) what happens to the mean, variance, and SD if every observation is increased by $c = 10$?

??? success "Solution to Exercise 1"
    (a) $\bar{x} = 40/5 = 8$.

    (b) Squared deviations: $25, 1, 1, 1, 36$; sum $= 64$; $s^2 = 64/4 = 16$.

    (c) $s = \sqrt{16} = 4$.

    (d) Adding a constant $c$ shifts the mean by $c$ but leaves variance and SD unchanged: $(x_i + c) - (\bar{x} + c) = x_i - \bar{x}$, so all squared deviations are identical. The new mean is 18; $s^2 = 16$, $s = 4$ still.

---

**Exercise 2.**
Prove that the sample variance with Bessel's correction is unbiased: $\mathbb{E}[s^2] = \sigma^2$ for i.i.d. data with mean $\mu$ and variance $\sigma^2$.

??? success "Solution to Exercise 2"
    Use the identity $\sum_i (X_i - \bar{X})^2 = \sum_i X_i^2 - n\bar{X}^2$. Take expectations:

    $$
    \mathbb{E}\!\sum_i X_i^2 = n(\sigma^2 + \mu^2), \qquad \mathbb{E}[n\bar{X}^2] = n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2
    $$

    Subtracting,

    $$
    \mathbb{E}\!\sum_i (X_i - \bar{X})^2 = (n-1)\sigma^2
    $$

    Hence $\mathbb{E}[s^2] = (n-1)\sigma^2/(n-1) = \sigma^2$. $\square$

    The ML estimator (dividing by $n$) instead yields $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$ — biased downward by a factor that vanishes only as $n \to \infty$.

---

**Exercise 3.**
**Scale and shift.** If $X$ has variance $\sigma^2$ and $Y = aX + b$, derive $\mathrm{Var}(Y)$. What does this mean for the units of $\sigma$?

??? success "Solution to Exercise 3"
    $\mathbb{E}[Y] = a\mu + b$. Then

    $$
    \mathrm{Var}(Y) = \mathbb{E}\!\left[(aX + b - a\mu - b)^2\right] = a^2 \mathbb{E}[(X - \mu)^2] = a^2 \sigma^2
    $$

    So $\mathrm{Var}(Y) = a^2 \sigma^2$ — the variance scales by the *square* of the multiplicative constant. The additive shift $b$ has no effect.

    **Units:** if $X$ is in dollars, $\sigma^2$ is in dollars-squared (not interpretable as a "typical deviation") and $\sigma$ is in dollars. The SD restores the original units, which is why SD is preferred for reporting "typical spread" while variance is preferred for algebraic manipulation (e.g., adding variances of independent variables).

---

**Exercise 4.**
**Variance of a sum.** For independent random variables $X_1, \ldots, X_n$, show that $\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i)$. Where does independence enter, and what is the formula when the variables are *correlated*?

??? success "Solution to Exercise 4"
    Let $\mu_i = \mathbb{E}[X_i]$. Then

    $$
    \mathrm{Var}\!\left(\sum_i X_i\right) = \mathbb{E}\!\left[\left(\sum_i (X_i - \mu_i)\right)^2\right] = \sum_i \mathbb{E}[(X_i - \mu_i)^2] + \sum_{i \ne j} \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]
    $$

    The first term is $\sum_i \sigma_i^2$. The second term, $\sum_{i \ne j} \mathrm{Cov}(X_i, X_j)$, is **zero** under independence (since the cross terms factor as $\mathbb{E}[X_i - \mu_i] \mathbb{E}[X_j - \mu_j] = 0$).

    **For correlated variables:**

    $$
    \mathrm{Var}\!\left(\sum_i X_i\right) = \sum_i \sigma_i^2 + 2 \sum_{i < j} \mathrm{Cov}(X_i, X_j)
    $$

    Positive correlations *inflate* the sum's variance; negative correlations reduce it. Portfolio diversification exploits this: combining assets with low or negative correlations reduces total variance for the same expected return.

---

**Exercise 5.**
The **standard error** of the sample mean is $\mathrm{SE}(\bar{X}) = \sigma/\sqrt{n}$. Why is this $\sqrt{n}$ and not $n$ in the denominator? Use this to explain why quadrupling sample size only halves the SE.

??? success "Solution to Exercise 5"
    The variance of the sample mean for i.i.d. data is

    $$
    \mathrm{Var}(\bar{X}) = \mathrm{Var}\!\left(\frac{1}{n}\sum_i X_i\right) = \frac{1}{n^2} \cdot n \sigma^2 = \frac{\sigma^2}{n}
    $$

    The standard error is the square root: $\mathrm{SE}(\bar{X}) = \sigma/\sqrt{n}$. The $\sqrt{n}$ appears because we're square-rooting a variance that scales as $1/n$.

    **Quadrupling rule:** if $n$ goes from $n$ to $4n$, then $\sqrt{n} \to 2\sqrt{n}$, and SE is halved. To halve the SE again you need another 4× in sample size — i.e., $16n$. Precision improves with $\sqrt{n}$, an unforgiving rate that drives the cost of high-precision surveys.

---

**Exercise 6.**
The **coefficient of variation** $\mathrm{CV} = \sigma/\mu$ is a unitless measure of relative spread. When is CV more useful than $\sigma$ alone? Give an example where two distributions have the same $\sigma$ but very different CVs.

??? success "Solution to Exercise 6"
    The CV is invariant under multiplicative rescaling: doubling all data values doubles both $\sigma$ and $\mu$, leaving CV unchanged. This makes it a "scale-free" spread measure useful for **comparing variability across distributions with different units or magnitudes**.

    **Example:**

    - Annual returns of stock A: $\mu = 5\%$, $\sigma = 5\%$, so CV $= 1.0$.
    - Annual returns of stock B: $\mu = 50\%$, $\sigma = 5\%$, so CV $= 0.1$.

    Both have the same absolute SD (5 percentage points). But A's returns are far more *variable relative to their typical value* — a one-SD downward move could wipe out a year's return entirely. B's returns are tightly clustered around a high level. CV captures this difference.

    **Caveats:** CV is undefined or unstable when $\mu$ is near zero or can change sign (e.g., investment returns that include losses). For such data, alternative measures like the **Sharpe ratio** ($\mu/\sigma$ for positive-$\mu$ comparisons) or **interquartile range / median** (robust analog) are preferred.
