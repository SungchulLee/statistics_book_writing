# Bernoulli and Binomial Distributions

## Overview

The **Bernoulli distribution** models a single trial with two outcomes (success/failure), while the **binomial distribution** extends this to count the number of successes in $n$ independent trials. Together, they form the foundation of discrete probability modeling.

---

## Bernoulli Distribution

### Definition

A random variable $X$ follows a Bernoulli distribution if it takes value 1 (success) with probability $p$ and value 0 (failure) with probability $1 - p$:

$$
X \sim \text{Bernoulli}(p), \qquad P(X = x) = p^x (1 - p)^{1-x}, \quad x \in \{0, 1\}
$$

### Properties

$$
\begin{aligned}
E[X] &= p \\
\text{Var}(X) &= p(1 - p) \\
\text{SD}(X) &= \sqrt{p(1 - p)}
\end{aligned}
$$

### Derivation of Variance

$$
E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1 - p)
$$

---

## Binomial Distribution

### Definition

If $X_1, X_2, \ldots, X_n$ are independent $\text{Bernoulli}(p)$ random variables, then $Y = \sum_{i=1}^n X_i$ follows a **binomial distribution**:

$$
Y \sim \text{Binomial}(n, p), \qquad P(Y = k) = \binom{n}{k} p^k (1 - p)^{n-k}, \quad k = 0, 1, \ldots, n
$$

The binomial coefficient $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ counts the number of ways to choose $k$ successes from $n$ trials.

### Properties

$$
\begin{aligned}
E[Y] &= np \\
\text{Var}(Y) &= np(1 - p) \\
\text{SD}(Y) &= \sqrt{np(1 - p)}
\end{aligned}
$$

### Derivation of Mean and Variance

Since $Y = \sum_{i=1}^n X_i$ where $X_i \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$:

$$
E[Y] = \sum_{i=1}^n E[X_i] = np
$$

By independence:

$$
\text{Var}(Y) = \sum_{i=1}^n \text{Var}(X_i) = np(1 - p)
$$

### Verifying the PMF Sums to 1

By the binomial theorem:

$$
\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1^n = 1
$$

---

## Binomial Coefficient Identities

Several identities are useful for working with binomial distributions:

$$
\begin{aligned}
(1) &\quad \binom{n}{k} = \binom{n}{n-k} \quad \text{(symmetry)} \\[4pt]
(2) &\quad \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{(Pascal's rule)} \\[4pt]
(3) &\quad k\binom{n}{k} = n\binom{n-1}{k-1} \quad \text{(absorption identity)}
\end{aligned}
$$

The absorption identity is particularly useful for computing $E[Y]$ directly from the PMF:

$$
E[Y] = \sum_{k=0}^n k \binom{n}{k} p^k (1-p)^{n-k} = np \sum_{k=1}^n \binom{n-1}{k-1} p^{k-1} (1-p)^{n-k} = np
$$

---

## Worked Example

**Problem:** A stock has a 60% chance of rising on any given day (independent across days). Over 10 trading days, what is the probability it rises on exactly 7 days?

**Solution:**

$$
P(Y = 7) = \binom{10}{7} (0.6)^7 (0.4)^3 = 120 \cdot 0.0280 \cdot 0.064 = 0.2150
$$

Expected number of up days: $E[Y] = 10 \times 0.6 = 6$.

---

## Python: PMF, CDF, and Sampling

### PMF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 10, 0.6
x = np.arange(0, n + 1)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x - 0.15, stats.binom(n, p).pmf(x), width=0.3, label='PMF', alpha=0.7)
ax.bar(x + 0.15, stats.binom(n, p).cdf(x), width=0.3, label='CDF', alpha=0.7)
ax.set_xlabel('k')
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Comparing Different Parameters

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
for n, p in [(10, 0.5), (20, 0.5), (20, 0.7)]:
    x = np.arange(0, n + 1)
    ax.plot(x, stats.binom(n, p).pmf(x), 'o-', label=f'n={n}, p={p}', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('k')
ax.legend()
plt.show()
```

### Sampling and Verification

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n, p = 10, 0.6
samples = stats.binom(n, p).rvs(100_000)

print(f"Theoretical mean: {n*p:.4f},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {n*p*(1-p):.4f},  Sample var:  {samples.var():.4f}")
```

---

## Normal Approximation to the Binomial

For large $n$, the binomial distribution is well approximated by a normal distribution:

$$
Y \sim \text{Binomial}(n, p) \approx N(np, \, np(1-p)) \quad \text{when } np \geq 5 \text{ and } n(1-p) \geq 5
$$

With continuity correction, $P(Y \leq k) \approx \Phi\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)$.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 50, 0.4
x_disc = np.arange(0, n + 1)
x_cont = np.linspace(0, n, 200)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x_disc, stats.binom(n, p).pmf(x_disc), alpha=0.5, label='Binomial PMF')
ax.plot(x_cont, stats.norm(n*p, np.sqrt(n*p*(1-p))).pdf(x_cont),
        'r-', lw=2, label='Normal approx.')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

---

## Key Takeaways

- The Bernoulli distribution models a single binary trial; the binomial counts successes over $n$ independent trials.
- The binomial PMF uses the binomial coefficient to account for all possible orderings of successes.
- Mean $np$ and variance $np(1-p)$ follow directly from the sum-of-independent-Bernoullis representation.
- For large $n$, the binomial is well approximated by the normal distribution, connecting discrete and continuous probability.
