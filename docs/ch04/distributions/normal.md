# Normal Distribution

## Overview

The **normal distribution** (also called the Gaussian distribution) is one of the most fundamental probability distributions in statistics. It describes continuous data that cluster around a central value, tapering off symmetrically on both sides in a characteristic "bell curve."

---

## Definition

A normal distribution is characterized by its mean $\mu$ (center) and variance $\sigma^2$ (spread). The PDF is:

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

We write $X \sim N(\mu, \sigma^2)$.

### Standard Normal Distribution

The special case with $\mu = 0$ and $\sigma = 1$ is the **standard normal distribution**:

$$
Z \sim N(0, 1), \qquad f(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)
$$

---

## Standardization

Any normal variable can be converted to a standard normal via the **Z-score transformation**:

$$
\begin{aligned}
\textbf{Standardization:} \quad & X \sim N(\mu, \sigma^2) \implies Z = \frac{X - \mu}{\sigma} \sim N(0, 1) \\[6pt]
\textbf{Reverse:} \quad & Z \sim N(0, 1) \implies X = Z\sigma + \mu \sim N(\mu, \sigma^2)
\end{aligned}
$$

---

## Properties of the Normal Distribution

### Closure Properties

$$
\begin{aligned}
(1) &\quad X \sim \text{Normal} \implies aX + b \sim \text{Normal} \\[4pt]
(2) &\quad X \sim \text{Normal}, \; Y \sim \text{Normal}, \; X \perp Y \implies X + Y \sim \text{Normal} \\[4pt]
(3) &\quad (X, Y) \sim \text{Multivariate Normal} \implies X + Y \sim \text{Normal}
\end{aligned}
$$

**Warning:** $X \sim \text{Normal}$ and $Y \sim \text{Normal}$ does **not** imply $X + Y \sim \text{Normal}$ unless independence or joint normality holds.

### Proof of Property (1)

For $a > 0$ and $X \sim N(\mu, \sigma^2)$:

$$
P(aX + b \leq x) = P\left(X \leq \frac{x - b}{a}\right) = \int_{-\infty}^{(x-b)/a} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(s-\mu)^2}{2\sigma^2}} ds
$$

Differentiating with respect to $x$:

$$
f_{aX+b}(x) = \frac{1}{\sqrt{2\pi(a\sigma)^2}} \exp\left(-\frac{(x - (a\mu + b))^2}{2a^2\sigma^2}\right)
$$

Therefore $aX + b \sim N(a\mu + b, \, a^2\sigma^2)$.

### Key Geometric Properties

- **Symmetry:** Perfectly symmetric about $\mu$; mean = median = mode = $\mu$.
- **Bell-shaped:** Most data is concentrated near the mean.
- **Infinite tails:** The tails extend to $\pm\infty$ but with rapidly decreasing probability.

### 68–95–99.7 Rule

$$
\begin{aligned}
P(\mu - \sigma < X < \mu + \sigma) &\approx 68\% \\
P(\mu - 2\sigma < X < \mu + 2\sigma) &\approx 95\% \\
P(\mu - 3\sigma < X < \mu + 3\sigma) &\approx 99.7\%
\end{aligned}
$$

---

## PDF of the Standard Normal: Verifying Key Properties

The PDF of $N(0, 1)$ is $f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$. We verify:

### (1) Total mass is 1

Let $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$. Then:

$$
I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy = \int_0^{2\pi}\!\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta = 2\pi
$$

So $I = \sqrt{2\pi}$, confirming $\int f(x)\,dx = 1$.

### (2) Mean is 0

The integrand $x \cdot e^{-x^2/2}$ is an **odd function**, so the integral over $(-\infty, \infty)$ is 0.

### (3) Variance is 1

By integration by parts:

$$
\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} x^2 e^{-x^2/2}\,dx = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-x^2/2}\,dx = 1
$$

---

## CDF of the Standard Normal

The CDF has no closed form and is computed numerically:

$$
\Phi(x) = N(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-s^2/2}\,ds
$$

### Properties of $\Phi$

$$
\begin{aligned}
(1) &\quad P(a \leq Z \leq b) = \Phi(b) - \Phi(a) \\
(2) &\quad P(Z \geq x) = P(Z \leq -x) = \Phi(-x) \\
(3) &\quad P(Z \geq x) = 1 - \Phi(x) \\
(4) &\quad P(Z \leq 0) = P(Z \geq 0) = 0.5
\end{aligned}
$$

---

## Integration Trick Related to Normal PDF

**Problem:** Compute $\int_{-\infty}^{\infty} e^{-x^2 - 2x}\,dx$.

**Solution:** Complete the square: $-x^2 - 2x = -(x+1)^2 + 1$. Then:

$$
\int_{-\infty}^{\infty} e^{-x^2-2x}\,dx = e \int_{-\infty}^{\infty} e^{-(x+1)^2}\,dx = e\sqrt{2\pi \cdot \tfrac{1}{2}} \cdot \underbrace{\int \frac{1}{\sqrt{\pi}} e^{-(x+1)^2}\,dx}_{=1 \text{ (PDF of } N(-1, 1/2))} = e\sqrt{\pi}
$$

---

## Python: Plotting PDF, CDF, and Sampling

### PDF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, stats.norm(mu, sigma).pdf(x), label='PDF')
ax.plot(x, stats.norm(mu, sigma).cdf(x), label='CDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Sampling with Estimated PDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
data = stats.norm(loc=0, scale=1).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(data, bins=100, density=True, color='blue', alpha=0.7, label="Samples")
ax.plot(bins, stats.norm(data.mean(), data.std()).pdf(bins),
        '--r', lw=3, label="Estimated Normal PDF")
ax.legend()
plt.show()
```

### 68–95–99.7 Rule Verification

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)
mean, std, n = df.x.mean(), df.x.std(), len(df.x)

n1 = len(df.x[(mean - std < df.x) & (df.x < mean + std)])
n2 = len(df.x[(mean - 2*std < df.x) & (df.x < mean + 2*std)])
n3 = len(df.x[(mean - 3*std < df.x) & (df.x < mean + 3*std)])

print(f"Within 1σ: {n1/n*100:.2f}%")   # ≈ 68%
print(f"Within 2σ: {n2/n*100:.2f}%")   # ≈ 95%
print(f"Within 3σ: {n3/n*100:.2f}%")   # ≈ 99.7%
```

---

## Area Under the Standard Normal Curve

### scipy.stats Methods

| Method | Description |
|:---|:---|
| `rvs` | Generate random samples |
| `pdf` | Compute the PDF |
| `cdf` | Compute $P(X \leq x)$ |
| `sf` | Survival function: $1 - \text{cdf}(x)$ |
| `ppf` | Percent point function (inverse of CDF) |

### Left-Tail, Right-Tail, and Center Areas

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def shade_area(z_bounds, side='left', ax=None):
    """Shade a region under the standard normal curve."""
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm().pdf(x), color='k', alpha=0.9)

    if side == 'left':
        x_shade = np.linspace(-4, z_bounds, 200)
    elif side == 'right':
        x_shade = np.linspace(z_bounds, 4, 200)
    else:  # center
        x_shade = np.linspace(z_bounds[0], z_bounds[1], 200)

    ax.fill_between(x_shade, stats.norm().pdf(x_shade), alpha=0.2, color='k')
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.set_yticks([])

# Left area
z = -1.2
print(f"P(Z ≤ {z}) = {stats.norm().cdf(z):.4f}")

# Right area
z = 1.2
print(f"P(Z ≥ {z}) = {stats.norm().sf(z):.4f}")

# Center area
z1, z2 = -2.1, 1.2
print(f"P({z1} ≤ Z ≤ {z2}) = {stats.norm().cdf(z2) - stats.norm().cdf(z1):.4f}")
```

---

## Why Normal?

The Central Limit Theorem explains the ubiquity of the normal distribution: the distribution of sample means is approximately normal for large $n$, regardless of the original population distribution:

$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty
$$

This makes the normal distribution the foundation for confidence intervals, hypothesis testing, and quality control.

---

## Fixing the Random Seed

`scipy.stats` uses NumPy's random number generator, so setting `np.random.seed()` ensures reproducibility:

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)
samples = stats.norm.rvs(size=10)
print(samples)  # Same output every time with seed 42
```

---

## Key Takeaways

- The normal distribution $N(\mu, \sigma^2)$ is characterized by its bell shape, symmetry, and the 68–95–99.7 rule.
- The standard normal $N(0,1)$ serves as a universal reference via Z-score standardization.
- Linear transformations and sums of independent normals remain normal.
- The CDF has no closed form but is efficiently computed numerically.
- The CLT explains why the normal distribution appears so frequently in nature and statistics.
