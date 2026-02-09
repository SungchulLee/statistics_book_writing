# Poisson Distribution

## Overview

The **Poisson distribution** models the number of events occurring in a fixed interval of time or space, given a known average rate. It is widely used in finance (trade arrivals, default counts), insurance (claim frequency), and queueing theory.

---

## Definition

A random variable $X$ follows a Poisson distribution with rate parameter $\lambda > 0$:

$$
X \sim \text{Poisson}(\lambda), \qquad P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots
$$

The parameter $\lambda$ represents both the mean and the variance of the distribution.

### Verifying the PMF Sums to 1

$$
\sum_{k=0}^{\infty} \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{-\lambda} \cdot e^{\lambda} = 1
$$

using the Taylor expansion of $e^{\lambda}$.

---

## Properties

$$
\begin{aligned}
E[X] &= \lambda \\
\text{Var}(X) &= \lambda \\
\text{SD}(X) &= \sqrt{\lambda}
\end{aligned}
$$

The equality of mean and variance is a defining characteristic of the Poisson distribution and is often used as a diagnostic check.

### Derivation of Mean

$$
E[X] = \sum_{k=0}^{\infty} k \cdot \frac{e^{-\lambda}\lambda^k}{k!} = \lambda e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!} = \lambda e^{-\lambda} \cdot e^{\lambda} = \lambda
$$

### Derivation of Variance

First compute $E[X(X-1)]$:

$$
E[X(X-1)] = \sum_{k=2}^{\infty} k(k-1) \frac{e^{-\lambda}\lambda^k}{k!} = \lambda^2 e^{-\lambda} \sum_{k=2}^{\infty} \frac{\lambda^{k-2}}{(k-2)!} = \lambda^2
$$

Then:

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = E[X(X-1)] + E[X] - (E[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda
$$

---

## Poisson as a Limit of the Binomial

The Poisson distribution arises as a limit of the binomial when $n$ is large, $p$ is small, and $\lambda = np$ remains constant:

$$
\lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k} = \frac{e^{-\lambda}\lambda^k}{k!} \qquad \text{where } p = \frac{\lambda}{n}
$$

### Proof Sketch

With $p = \lambda/n$:

$$
\binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1 - \frac{\lambda}{n}\right)^{n-k}
= \frac{n!}{k!(n-k)!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1 - \frac{\lambda}{n}\right)^n \cdot \left(1 - \frac{\lambda}{n}\right)^{-k}
$$

As $n \to \infty$: $\frac{n!}{(n-k)! \, n^k} \to 1$, $\left(1 - \frac{\lambda}{n}\right)^n \to e^{-\lambda}$, and $\left(1 - \frac{\lambda}{n}\right)^{-k} \to 1$.

**Rule of thumb:** Use Poisson when $n \geq 20$ and $p \leq 0.05$ (or more conservatively, $n \geq 100$ and $np \leq 10$).

---

## Additive Property

If $X_1 \sim \text{Poisson}(\lambda_1)$ and $X_2 \sim \text{Poisson}(\lambda_2)$ are independent, then:

$$
X_1 + X_2 \sim \text{Poisson}(\lambda_1 + \lambda_2)
$$

This extends to any finite sum of independent Poisson random variables.

---

## Poisson Process Connection

The Poisson distribution is intimately connected to the **Poisson process**. If events arrive at a constant rate $\lambda$ per unit time, and arrivals are independent, then the number of events in an interval of length $t$ follows $\text{Poisson}(\lambda t)$, and the time between consecutive events follows $\text{Exponential}(\lambda)$.

---

## Worked Example

**Problem:** A stock exchange processes an average of 3 large block trades per hour. What is the probability of observing exactly 5 block trades in a given hour? What is the probability of observing at most 2?

**Solution:**

$$
P(X = 5) = \frac{e^{-3} \cdot 3^5}{5!} = \frac{0.0498 \cdot 243}{120} = 0.1008
$$

$$
P(X \leq 2) = \sum_{k=0}^{2} \frac{e^{-3} \cdot 3^k}{k!} = e^{-3}(1 + 3 + 4.5) = 0.0498 \cdot 8.5 = 0.4232
$$

---

## Python: PMF, CDF, and Sampling

### PMF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lam = 5
x = np.arange(0, 20)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x - 0.15, stats.poisson(lam).pmf(x), width=0.3, label='PMF', alpha=0.7)
ax.bar(x + 0.15, stats.poisson(lam).cdf(x), width=0.3, label='CDF', alpha=0.7)
ax.set_xlabel('k')
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Comparing Different Rates

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
for lam in [1, 4, 10]:
    x = np.arange(0, 25)
    ax.plot(x, stats.poisson(lam).pmf(x), 'o-', label=f'λ={lam}', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('k')
ax.legend()
plt.show()
```

### Poisson as Binomial Limit

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lam = 5
x = np.arange(0, 20)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x, stats.poisson(lam).pmf(x), alpha=0.5, label='Poisson(λ=5)')
for n in [20, 50, 200]:
    ax.plot(x, stats.binom(n, lam/n).pmf(x), 'o-', label=f'Binom(n={n}, p={lam/n:.3f})', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Sampling and Mean-Variance Check

```python
import numpy as np
from scipy import stats

np.random.seed(42)
lam = 7
samples = stats.poisson(lam).rvs(100_000)

print(f"Theoretical mean: {lam},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {lam},  Sample var:  {samples.var():.4f}")
print(f"Mean ≈ Var: {np.isclose(samples.mean(), samples.var(), atol=0.1)}")
```

---

## Key Takeaways

- The Poisson distribution models rare event counts with rate parameter $\lambda$ that equals both the mean and variance.
- It arises as the limit of the binomial distribution when $n$ is large and $p$ is small.
- The additive property makes it natural for aggregating independent event counts.
- The connection to the Poisson process links discrete event counts to continuous inter-arrival times (exponential distribution).
- The mean-equals-variance property is a useful diagnostic: if sample variance greatly exceeds the mean, the data may be **overdispersed** relative to the Poisson model.
