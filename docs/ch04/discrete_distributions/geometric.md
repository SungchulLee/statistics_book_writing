# Geometric and Negative Binomial Distributions

## Overview

The **geometric distribution** models the number of trials until the first success, while the **negative binomial distribution** generalizes this to the number of trials until the $r$-th success. Both arise naturally in sequential experiments with independent Bernoulli trials.

---

## Geometric Distribution

### Definition

If independent Bernoulli trials with success probability $p$ are performed until the first success, then the number of trials $X$ follows a geometric distribution:

$$
X \sim \text{Geometric}(p), \qquad P(X = k) = (1 - p)^{k-1} p, \quad k = 1, 2, 3, \ldots
$$

The PMF captures that the first $k-1$ trials must be failures and the $k$-th trial must be a success.

**Alternative parameterization:** Some texts define $Y$ as the number of *failures* before the first success, so $Y = X - 1$ and $P(Y = k) = (1-p)^k p$ for $k = 0, 1, 2, \ldots$

### Verifying the PMF Sums to 1

$$
\sum_{k=1}^{\infty} (1-p)^{k-1} p = p \sum_{j=0}^{\infty} (1-p)^j = p \cdot \frac{1}{1 - (1-p)} = 1
$$

using the geometric series formula with ratio $|1-p| < 1$.

### Properties

$$
\begin{aligned}
E[X] &= \frac{1}{p} \\[4pt]
\text{Var}(X) &= \frac{1 - p}{p^2}
\end{aligned}
$$

### Derivation of Mean

$$
E[X] = \sum_{k=1}^{\infty} k(1-p)^{k-1} p = p \cdot \frac{d}{dq}\left[\sum_{k=0}^{\infty} q^k \right]_{q=1-p} \!\!\!\!= p \cdot \frac{1}{(1-q)^2}\bigg|_{q=1-p} = \frac{1}{p}
$$

### Derivation of Variance

Using $E[X(X-1)] = \sum_{k=2}^{\infty} k(k-1)(1-p)^{k-1}p = \frac{2(1-p)}{p^2}$:

$$
E[X^2] = E[X(X-1)] + E[X] = \frac{2(1-p)}{p^2} + \frac{1}{p}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2(1-p)}{p^2} + \frac{1}{p} - \frac{1}{p^2} = \frac{1-p}{p^2}
$$

---

## Memoryless Property

The geometric distribution is the **only** discrete distribution with the memoryless property:

$$
P(X > s + t \mid X > s) = P(X > t) \quad \text{for all } s, t \geq 0
$$

### Proof

$$
P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X > t)
$$

**Interpretation:** Given that you have already waited $s$ trials without success, the probability of waiting at least $t$ more trials is the same as starting fresh. Past failures carry no information about future success.

---

## Negative Binomial Distribution

### Definition

The number of trials $Y$ needed to achieve $r$ successes in independent Bernoulli trials follows a **negative binomial distribution**:

$$
Y \sim \text{NegBin}(r, p), \qquad P(Y = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots
$$

The binomial coefficient $\binom{k-1}{r-1}$ counts the ways to place $r-1$ successes among the first $k-1$ trials (the $k$-th trial is necessarily a success).

**Note:** When $r = 1$, the negative binomial reduces to the geometric distribution.

### Properties

$$
\begin{aligned}
E[Y] &= \frac{r}{p} \\[4pt]
\text{Var}(Y) &= \frac{r(1-p)}{p^2}
\end{aligned}
$$

### Derivation via Sum of Geometrics

If $X_1, X_2, \ldots, X_r$ are independent $\text{Geometric}(p)$ random variables, then $Y = \sum_{i=1}^r X_i \sim \text{NegBin}(r, p)$. Therefore:

$$
E[Y] = \sum_{i=1}^r E[X_i] = \frac{r}{p}, \qquad \text{Var}(Y) = \sum_{i=1}^r \text{Var}(X_i) = \frac{r(1-p)}{p^2}
$$

---

## Worked Example

**Problem:** A trader's strategy has a 30% win rate on each independent trade. What is the expected number of trades to achieve the first win? What is the probability that the first win occurs on the 5th trade?

**Solution:**

$$
E[X] = \frac{1}{0.3} \approx 3.33 \text{ trades}
$$

$$
P(X = 5) = (1 - 0.3)^{5-1} \cdot 0.3 = (0.7)^4 \cdot 0.3 = 0.2401 \cdot 0.3 = 0.0720
$$

---

## Python: PMF, CDF, and Sampling

### Geometric Distribution

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

p = 0.3
x = np.arange(1, 20)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x - 0.15, stats.geom(p).pmf(x), width=0.3, label='PMF', alpha=0.7)
ax.bar(x + 0.15, stats.geom(p).cdf(x), width=0.3, label='CDF', alpha=0.7)
ax.set_xlabel('k (number of trials)')
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Negative Binomial Distribution

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# scipy parameterizes by number of failures: nbinom(r, p) gives P(Y=k) for k failures
r, p = 5, 0.4
x = np.arange(0, 30)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x, stats.nbinom(r, p).pmf(x), alpha=0.7, label=f'NegBin(r={r}, p={p})')
ax.set_xlabel('k (number of failures before r-th success)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Verifying the Memoryless Property

```python
import numpy as np
from scipy import stats

np.random.seed(42)
p = 0.3
samples = stats.geom(p).rvs(1_000_000)

s = 3
# P(X > s + t | X > s) vs P(X > t)
for t in [1, 3, 5]:
    conditional = np.mean(samples[samples > s] > s + t)
    unconditional = np.mean(samples > t)
    print(f"P(X>{s}+{t}|X>{s}) = {conditional:.4f},  P(X>{t}) = {unconditional:.4f}")
```

### Comparing Parameters

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
for p in [0.2, 0.4, 0.6]:
    x = np.arange(1, 25)
    ax.plot(x, stats.geom(p).pmf(x), 'o-', label=f'Geometric(p={p})', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('k')
ax.legend()
plt.show()
```

---

## Relationship to Other Distributions

$$
\begin{aligned}
\text{Geometric}(p) &= \text{NegBin}(1, p) \\[4pt]
\text{NegBin}(r, p) &= \sum_{i=1}^r \text{Geometric}_i(p) \quad \text{(independent sum)} \\[4pt]
\text{Geometric} &\leftrightarrow \text{Exponential} \quad \text{(discrete vs continuous memoryless)}
\end{aligned}
$$

---

## Key Takeaways

- The geometric distribution models waiting time to the first success and is the only discrete memoryless distribution.
- The negative binomial generalizes the geometric to count trials until the $r$-th success.
- Both distributions arise from sequences of independent Bernoulli trials.
- The mean $1/p$ of the geometric distribution has an intuitive interpretation: lower success probability means longer expected wait.
- The geometric distribution is the discrete analogue of the exponential distribution, sharing the memoryless property.
