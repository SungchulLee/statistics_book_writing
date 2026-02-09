# Sampling Distribution of $S^2$

## Overview

The **sampling distribution of the sample variance** $S^2$ describes how the variance computed from a random sample behaves across repeated samples drawn from a population. This concept is critical for understanding how precisely we can estimate the true population variance $\sigma^2$.

## Mathematical Definition

Let $X_1, X_2, \dots, X_n$ be i.i.d. from a population with mean $\mu$ and variance $\sigma^2$. The sample variance is:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2.
$$

## Properties

### Expected Value (Unbiasedness)

$$
E[S^2] = \sigma^2.
$$

This unbiasedness is why we divide by $n-1$ (degrees of freedom) rather than $n$ — Bessel's correction compensates for the loss of one degree of freedom when estimating $\mu$ from the sample.

### Chi-Square Connection (Normal Populations)

If the population is **normal**, the scaled sample variance follows a chi-square distribution:

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

Equivalently:

$$
S^2 \sim \frac{\sigma^2}{n-1} \cdot \chi^2_{n-1}.
$$

### Variance of $S^2$

Under normality:

$$
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}.
$$

**Derivation.** Since $\text{Var}(\chi^2_{n-1}) = 2(n-1)$:

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1) \;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}.
$$

### Standard Error of $S^2$

$$
\text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}} \sim O\!\left(\frac{1}{\sqrt{n}}\right).
$$

## Speed of Convergence

Both $\bar{X}$ and $S^2$ converge at the same asymptotic rate:

| Estimator | Standard Error | Rate |
|-----------|---------------|------|
| $\bar{X}$ | $\sigma / \sqrt{n}$ | $O(1/\sqrt{n})$ |
| $S^2$ | $\sigma^2 \sqrt{2/(n-1)}$ | $O(1/\sqrt{n})$ |

There is **no** faster or slower rate between them in terms of convergence speed.

## The Major Limitation of $S^2$

The critical distinction between $\bar{X}$ and $S^2$ lies not in speed but in **distributional robustness**:

✅ **Sample mean $\bar{X}$** benefits from the Central Limit Theorem (CLT), which guarantees approximate normality of $\bar{X}$ regardless of the population shape, as long as $n$ is large enough.

❌ **Sample variance $S^2$** relies on:

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1},
$$

which **only holds under normality**. For skewed, heavy-tailed, or otherwise non-normal populations, this chi-squared result no longer applies, and $S^2$ can behave unpredictably even with large samples.

## Worked Examples

### Example 1: Expected Value and Variance of $S^2$

**Problem.** A sample of $n = 10$ from $N(\mu, 25)$. Find $E[S^2]$ and $\text{Var}(S^2)$.

**Solution.** For $Y \sim \chi^2_{n-1}$, $EY = n-1$ and $\text{Var}(Y) = 2(n-1)$.

$$
E\!\left[\frac{(n-1)S^2}{\sigma^2}\right] = n - 1
\;\;\Longrightarrow\;\;
E[S^2] = \sigma^2 = 25
$$

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1)
\;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1} = \frac{2(25^2)}{9} = \frac{1250}{9} \approx 138.89
$$

### Example 2: Probability Involving $S^2$ (Normal Population)

**Problem.** A sample of $n = 10$ from $N(\mu, 25)$. Find $P(S^2 > 30)$.

**Solution.**

$$
\frac{(n-1)S^2}{\sigma^2} = \frac{9 \times 30}{25} = 10.8
$$

$$
P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.2897
$$

```python
from scipy import stats

chi2_stat = 9 * 30 / 25
p_value = stats.chi2(df=9).sf(chi2_stat)
print(f"P(S^2 > 30) = {p_value:.4f}")
```

### Example 3: Without Normality Assumption

**Problem.** A sample of $n = 10$ from a population with variance 25 (no normality assumed). What can be said about $P(S^2 > 30)$?

**Solution.** Without normality, $\frac{(n-1)S^2}{\sigma^2}$ does **not** follow a chi-square distribution. We know $E[S^2] = 25$, but we cannot determine $P(S^2 > 30)$ without additional information about the population's shape.

Chebyshev's inequality could provide a bound if $\text{Var}(S^2)$ were known, but that quantity depends on higher moments of the non-normal population, which are unavailable.

## Confidence Interval for $\sigma^2$

Using the chi-square pivot (under normality):

$$
P\!\left(\chi^2_{\alpha/2, \, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, \, n-1}\right) = 1 - \alpha
$$

Rearranging:

$$
\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, \, n-1}}, \;\; \frac{(n-1)S^2}{\chi^2_{\alpha/2, \, n-1}}\right]
$$

!!! note
    Because the chi-square distribution is asymmetric, this confidence interval is **not** symmetric around $S^2$.

## Simulation: Sampling Distribution of $S^2$

### Normal Population

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.norm().rvs(100_000)
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Normal)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### Income (Skewed) Population

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)
population = df['x'].values
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Income — Skewed)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## Summary

| Property | Result |
|----------|--------|
| $E[S^2]$ | $\sigma^2$ (unbiased, always) |
| $\text{Var}(S^2)$ | $2\sigma^4/(n-1)$ (under normality) |
| Distribution | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ (under normality only) |
| Robustness | ❌ No CLT-like guarantee — sensitive to non-normality |
| CI for $\sigma^2$ | Asymmetric, based on chi-square quantiles |
