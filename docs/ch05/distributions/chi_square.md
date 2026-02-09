# Chi-Square Distribution ($\chi^2$)

## Overview

The **chi-square distribution** arises naturally as the distribution of a sum of squared standard normal random variables. It plays a central role in inference about population variance, goodness-of-fit tests, and tests of independence.

---

## Definition

If $Z_1, Z_2, \ldots, Z_d$ are independent standard normal random variables, then:

$$
\sum_{i=1}^d Z_i^2 \sim \chi^2_d
$$

The parameter $d$ is called the **degrees of freedom**.

---

## Degrees of Freedom and Shape

The shape of the $\chi^2$ distribution depends critically on $d$:

- **Low $d$ (e.g., 1–2):** Highly right-skewed with a mode near zero.
- **High $d$:** Becomes more symmetric and approaches a normal distribution (by the CLT, since it is a sum of i.i.d. variables).

---

## Properties

### Basic Properties

$$
\begin{aligned}
\text{Mean} &= d \\
\text{Variance} &= 2d \\
\end{aligned}
$$

For $d = 1$, the distribution is highly skewed. For larger $d$, it becomes more symmetric.

### Additivity

If $X_1 \sim \chi^2_{d_1}$ and $X_2 \sim \chi^2_{d_2}$ are **independent**, then:

$$
X_1 + X_2 \sim \chi^2_{d_1 + d_2}
$$

This is useful when analyzing total variability across independent components.

---

## PDF

$$
f(x; d) = \frac{1}{2^{d/2}\,\Gamma(d/2)} \, x^{(d/2)-1} \, e^{-x/2}, \quad x > 0
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 20, 500)
fig, ax = plt.subplots(figsize=(12, 5))

for df in range(1, 11):
    ax.plot(x, chi2.pdf(x, df), label=f'df = {df}', alpha=0.7)

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Chi-Square PDF for Various Degrees of Freedom')
ax.legend(title='df')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## CDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 20, 500)
fig, ax = plt.subplots(figsize=(12, 5))

for df in range(1, 11):
    ax.plot(x, chi2.cdf(x, df), label=f'df = {df}', alpha=0.7)

ax.set_xlabel('x')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Chi-Square CDF')
ax.legend(title='df')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## PPF (Inverse CDF)

```python
from scipy import stats

df = 10
chi2_975 = stats.chi2(df).ppf(0.975)
print(f"97.5th percentile of χ²(10): {chi2_975:.4f}")

chi2_99 = stats.chi2(df).ppf(0.99)
print(f"99th percentile of χ²(10): {chi2_99:.4f}")
```

---

## Random Samples

### Direct Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
df = 5
data = stats.chi2(df).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.7, label='χ² Samples')
ax.plot(bins, stats.chi2(df).pdf(bins), '--r', lw=3, label='χ² PDF')
ax.legend()
plt.show()
```

### Sampling from Definition (Sum of Squared Normals)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
df = 5
data = np.sum(stats.norm().rvs((df, 10_000))**2, axis=0)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.7, label='Sum of Z² Samples')
ax.plot(bins, stats.chi2(df).pdf(bins), '--r', lw=3, label='χ² PDF')
ax.legend()
plt.show()
```

---

## Why Chi-Square?

The chi-square distribution arises in the study of **sample variance**. For i.i.d. $X_i \sim N(\mu, \sigma^2)$:

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

This result allows us to construct confidence intervals and hypothesis tests for $\sigma^2$.

### Dependence on Normality

This exact chi-square result **depends critically on the normality assumption**:

- For **normal populations**: $\bar{X}$ and $S^2$ are independent, and $(n-1)S^2/\sigma^2$ is exactly chi-square.
- For **non-normal populations**: the chi-square approximation is unreliable, especially for small $n$. The distribution of $S^2$ may differ dramatically.

### Simulation: Normal Population

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

n, n_sim, mu, sigma = 10, 10_000, 1, 2
samples = stats.norm(loc=mu, scale=sigma).rvs(size=(n_sim, n))
s = samples.std(axis=1, ddof=1)
data = (n - 1) * s**2 / sigma**2

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.7)
ax.plot(bins, stats.chi2(n-1).pdf(bins), '--r', lw=3, label='χ²(n-1) PDF')
ax.set_title('(n-1)S²/σ² from Normal Population → χ² Exact')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

### Simulation: Non-Normal Population

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

n, n_sim = 10, 10_000
samples = stats.expon().rvs(size=(n_sim, n))  # Exponential, not normal
s = samples.std(axis=1, ddof=1)
data = (n - 1) * s**2  # σ² = 1 for Exp(1)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.7)
ax.plot(bins, stats.chi2(n-1).pdf(bins), '--r', lw=3, label='χ²(n-1) PDF')
ax.set_title('(n-1)S²/σ² from Exponential Population → χ² Approximation Fails')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Practical Implications

| Scenario | Chi-Square Validity |
|:---|:---|
| Normal population | Exact |
| Large $n$, non-normal | May be approximately valid via CLT |
| Small $n$, skewed/binary population | Unreliable; use exact or resampling methods |
| Binary data | Use $np \geq 5$ and $n(1-p) \geq 5$ rule |

---

## Key Takeaways

- The chi-square distribution is the sum of squared standard normal variables.
- It governs inference about population variance when the population is normal.
- The additivity property makes it useful for combining independent variance components.
- The exactness of the chi-square result for $S^2$ depends critically on normality.
