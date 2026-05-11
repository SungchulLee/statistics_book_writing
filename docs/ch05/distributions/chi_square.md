# Chi-Square Distribution (chi-squared)

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

## Exercises

**Exercise 1.**
If $X \sim \chi^2_5$ and $Y \sim \chi^2_8$ are independent, what is the distribution of $X + Y$? Compute $E[X+Y]$ and $\text{Var}(X+Y)$.

??? success "Solution to Exercise 1"
    By the additivity property of the chi-square distribution, the sum of independent chi-square random variables is chi-square with degrees of freedom equal to the sum:

    $$
    X + Y \sim \chi^2_{5+8} = \chi^2_{13}
    $$

    $$
    E[X+Y] = 13, \quad \text{Var}(X+Y) = 2 \times 13 = 26
    $$

---

**Exercise 2.**
A random sample of $n = 20$ observations is drawn from a $N(\mu, 9)$ population. What is the exact distribution of $\frac{(n-1)S^2}{\sigma^2}$? Find the probability that $S^2 > 15$ (where $\sigma^2 = 9$).

??? success "Solution to Exercise 2"
    Since the population is normal, the sampling distribution is exact:

    $$
    \frac{(n-1)S^2}{\sigma^2} = \frac{19 S^2}{9} \sim \chi^2_{19}
    $$

    We need $P(S^2 > 15) = P\!\left(\frac{19 S^2}{9} > \frac{19 \times 15}{9}\right) = P(\chi^2_{19} > 31.67)$.

    From chi-square tables or software, $P(\chi^2_{19} > 31.67) \approx 0.034$. There is approximately a 3.4% chance that the sample variance exceeds 15 when $\sigma^2 = 9$.

---

**Exercise 3.**
Explain why the chi-square distribution is right-skewed for small degrees of freedom but becomes approximately symmetric for large degrees of freedom.

??? success "Solution to Exercise 3"
    The chi-square distribution with $k$ degrees of freedom is a sum of $k$ independent $\chi^2_1$ variables, each of which is the square of a standard normal. The $\chi^2_1$ distribution is heavily right-skewed (it can only be non-negative, with most mass near 0 and a long right tail).

    For small $k$, the sum of a few such skewed variables remains skewed. As $k$ increases, the CLT applies: the sum of many independent random variables converges to a normal distribution. Specifically, the skewness of $\chi^2_k$ is $\sqrt{8/k}$, which decreases to 0 as $k \to \infty$. For $k = 2$, skewness is 2 (highly skewed); for $k = 50$, skewness is 0.4 (nearly symmetric).

---

**Exercise 4.**
A researcher draws a sample of $n = 10$ from an exponential population and computes $\frac{(n-1)S^2}{\sigma^2}$. She assumes it follows a $\chi^2_9$ distribution. Is this valid? Explain what goes wrong.

??? success "Solution to Exercise 4"
    This is **not valid**. The result $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ holds **only when the population is normal**. The exponential distribution is right-skewed with excess kurtosis $\kappa = 6$, which causes the distribution of $S^2$ to have much heavier tails than a $\chi^2_9$ distribution.

    Specifically, the variance of $S^2$ for a non-normal population depends on the kurtosis: $\text{Var}(S^2) \approx \frac{2\sigma^4}{n-1}(1 + \kappa/2)$. For the exponential distribution, this is $\frac{2\sigma^4}{9}(1 + 3) = \frac{8\sigma^4}{9}$, which is 4 times larger than the chi-square theory predicts ($\frac{2\sigma^4}{9}$). Confidence intervals and hypothesis tests based on the chi-square assumption would have incorrect coverage and inflated Type I error rates.
