# Law of Large Numbers

## Overview

The **Law of Large Numbers (LLN)** is a fundamental theorem in probability theory describing how the average of a large number of independent and identically distributed (i.i.d.) random variables converges to the expected value as the sample size increases. It provides the theoretical basis for why sample averages are reliable estimators of population means.

---

## The Weak Law of Large Numbers (WLLN)

For a sequence of i.i.d. random variables $X_1, X_2, \ldots, X_n$ with finite mean $\mu$, the sample mean converges to the population mean **in probability**:

$$
\bar{X} = \frac{S_n}{n} \xrightarrow{P} \mu \quad \text{as } n \to \infty
$$

where $S_n = X_1 + X_2 + \cdots + X_n$.

More precisely, for any fixed $\varepsilon > 0$:

$$
P\left(\left|\frac{S_n}{n} - \mu\right| > \varepsilon\right) \to 0 \quad \text{as } n \to \infty
$$

**Interpretation:** As we collect more data, the probability that the sample mean deviates from $\mu$ by more than any fixed amount $\varepsilon$ goes to zero. For example, flipping a fair coin many times will result in the proportion of heads converging to 0.5.

---

## The Strong Law of Large Numbers (SLLN)

The SLLN provides a stronger form of convergence known as **almost sure convergence**:

$$
\bar{X} = \frac{S_n}{n} \xrightarrow{\text{a.s.}} \mu \quad \text{as } n \to \infty
$$

Equivalently, for any fixed $\varepsilon > 0$:

$$
P\left(\omega \in \Omega : \frac{S_n(\omega)}{n} \to \mu\right) = 1
$$

**Difference from WLLN:** The WLLN says the probability of a large deviation goes to zero for each $n$, but individual sample paths might still occasionally deviate. The SLLN guarantees that **every sample path** (except a set of probability zero) converges to $\mu$.

---

## Python Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate die rolls and track the running average
n_rolls = 10_000
rolls = np.random.randint(1, 7, size=n_rolls)
running_avg = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(running_avg, alpha=0.8, label='Running Average')
ax.axhline(y=3.5, color='r', linestyle='--', label='E[X] = 3.5')
ax.set_xlabel('Number of Rolls')
ax.set_ylabel('Sample Mean')
ax.set_title('Law of Large Numbers: Fair Die')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Key Takeaways

- The **WLLN** says the sample mean converges to $\mu$ in probability.
- The **SLLN** says the sample mean converges to $\mu$ almost surely (a stronger guarantee).
- Both require i.i.d. samples with finite mean.
- The LLN is the theoretical justification for using sample averages to estimate population parameters.
