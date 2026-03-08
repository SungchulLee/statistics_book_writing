# Uniform Distribution

## Overview

The **uniform distribution** assigns equal probability to all values in an interval $[a, b]$. It is the simplest continuous distribution and serves as the foundation for random number generation, simulation, and probability integral transforms.

---

## Definition

A random variable $X$ follows a continuous uniform distribution on $[a, b]$:

$$
X \sim \text{Uniform}(a, b), \qquad f(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}
$$

The PDF is constant over the interval, reflecting equal likelihood for all values.

### CDF

$$
F(x) = \begin{cases} 0 & x < a \\ \frac{x - a}{b - a} & a \leq x \leq b \\ 1 & x > b \end{cases}
$$

---

## Properties

$$
\begin{aligned}
E[X] &= \frac{a + b}{2} \\[4pt]
\text{Var}(X) &= \frac{(b - a)^2}{12} \\[4pt]
\text{SD}(X) &= \frac{b - a}{2\sqrt{3}}
\end{aligned}
$$

### Derivation of Mean

$$
E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\bigg|_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}
$$

### Derivation of Variance

$$
E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^3}{3}\bigg|_a^b = \frac{a^2 + ab + b^2}{3}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2 + ab + b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}
$$

---

## Standard Uniform Distribution

The special case $U \sim \text{Uniform}(0, 1)$ is the **standard uniform distribution**. Any uniform variable can be related to it:

$$
X = a + (b - a)U \sim \text{Uniform}(a, b) \quad \text{where } U \sim \text{Uniform}(0, 1)
$$

Conversely:

$$
U = \frac{X - a}{b - a} \sim \text{Uniform}(0, 1) \quad \text{where } X \sim \text{Uniform}(a, b)
$$

---

## Probability Integral Transform

The uniform distribution plays a central role in simulation through the **probability integral transform**:

**Theorem:** If $X$ is a continuous random variable with CDF $F$, then $F(X) \sim \text{Uniform}(0, 1)$.

**Converse (Inverse Transform Sampling):** If $U \sim \text{Uniform}(0, 1)$, then $X = F^{-1}(U)$ has CDF $F$.

### Proof

$$
P(F(X) \leq u) = P(X \leq F^{-1}(u)) = F(F^{-1}(u)) = u
$$

which is the CDF of $\text{Uniform}(0, 1)$.

This theorem is the basis for generating random samples from any distribution using only a uniform random number generator.

---

## Discrete Uniform Distribution

The discrete counterpart assigns equal probability to a finite set of values $\{a, a+1, \ldots, b\}$:

$$
P(X = k) = \frac{1}{b - a + 1}, \quad k = a, a+1, \ldots, b
$$

$$
E[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{(b - a + 1)^2 - 1}{12}
$$

---

## Worked Example

**Problem:** Daily returns of a certain asset are modeled as uniformly distributed between $-2\%$ and $+3\%$. What is the probability the return exceeds $1\%$? What is the expected return?

**Solution:**

$$
P(X > 1) = \frac{3 - 1}{3 - (-2)} = \frac{2}{5} = 0.40
$$

$$
E[X] = \frac{-2 + 3}{2} = 0.5\%
$$

---

## Python: PDF, CDF, and Sampling

### PDF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

a, b = 2, 8
x = np.linspace(a - 1, b + 1, 300)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), label='PDF', lw=2)
ax.plot(x, stats.uniform(loc=a, scale=b-a).cdf(x), label='CDF', lw=2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Sampling with Histogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
a, b = 2, 8
samples = stats.uniform(loc=a, scale=b-a).rvs(50_000)

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(samples, bins=60, density=True, alpha=0.7, label='Samples')
x = np.linspace(a - 1, b + 1, 300)
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), 'r-', lw=2, label='PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Inverse Transform Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate exponential samples via inverse transform
u = np.random.uniform(0, 1, 50_000)
lam = 2.0
x_exp = -np.log(1 - u) / lam  # F_inv(u) for Exponential(lambda)

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(x_exp, bins=100, density=True, alpha=0.7, label='Inverse transform samples')
t = np.linspace(0, 4, 200)
ax.plot(t, stats.expon(scale=1/lam).pdf(t), 'r-', lw=2, label='Exponential PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

---

## Key Takeaways

- The uniform distribution assigns equal probability to all values in an interval, making it the "maximally uninformative" distribution over a bounded range.
- The standard uniform $U(0,1)$ is the building block for random number generation via the inverse transform method.
- The probability integral transform establishes that applying the CDF to any continuous random variable yields a uniform result.
- Despite its simplicity, the uniform distribution is foundational to Monte Carlo simulation and computational statistics.
