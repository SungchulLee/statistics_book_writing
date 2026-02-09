# Exponential Distribution

## Overview

The **exponential distribution** models the time between events in a Poisson process. It is the continuous analogue of the geometric distribution and the only continuous distribution with the memoryless property. Common applications include modeling inter-arrival times, waiting times, and component lifetimes.

---

## Definition

A random variable $X$ follows an exponential distribution with rate parameter $\lambda > 0$:

$$
X \sim \text{Exponential}(\lambda), \qquad f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**Alternative parameterization:** Some texts use the scale parameter $\beta = 1/\lambda$, writing $f(x) = \frac{1}{\beta}e^{-x/\beta}$. SciPy uses the scale parameterization.

### CDF

$$
F(x) = 1 - e^{-\lambda x}, \quad x \geq 0
$$

### Survival Function

$$
S(x) = P(X > x) = e^{-\lambda x}
$$

---

## Properties

$$
\begin{aligned}
E[X] &= \frac{1}{\lambda} \\[4pt]
\text{Var}(X) &= \frac{1}{\lambda^2} \\[4pt]
\text{SD}(X) &= \frac{1}{\lambda} \\[4pt]
\text{Median} &= \frac{\ln 2}{\lambda}
\end{aligned}
$$

Note that $\text{Mean} = \text{SD} = 1/\lambda$, a distinctive feature of the exponential distribution.

### Derivation of Mean

$$
E[X] = \int_0^{\infty} x \lambda e^{-\lambda x}\,dx = \left[-x e^{-\lambda x}\right]_0^{\infty} + \int_0^{\infty} e^{-\lambda x}\,dx = \frac{1}{\lambda}
$$

### Derivation of Variance

$$
E[X^2] = \int_0^{\infty} x^2 \lambda e^{-\lambda x}\,dx = \frac{2}{\lambda^2}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
$$

---

## Memoryless Property

The exponential distribution is the **only** continuous distribution with the memoryless property:

$$
P(X > s + t \mid X > s) = P(X > t) \quad \text{for all } s, t \geq 0
$$

### Proof

$$
P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
$$

**Interpretation:** If you have already waited $s$ units of time, the remaining wait time has the same distribution as if you had just started. The process "forgets" its history.

---

## Connection to the Poisson Process

If events arrive according to a Poisson process with rate $\lambda$, then:

$$
\begin{aligned}
\text{Number of events in } [0, t] &\sim \text{Poisson}(\lambda t) \\
\text{Time between consecutive events} &\sim \text{Exponential}(\lambda) \\
\text{Time to the } n\text{-th event} &\sim \text{Gamma}(n, \lambda)
\end{aligned}
$$

---

## Minimum of Exponentials

If $X_1 \sim \text{Exp}(\lambda_1)$ and $X_2 \sim \text{Exp}(\lambda_2)$ are independent, then:

$$
\min(X_1, X_2) \sim \text{Exp}(\lambda_1 + \lambda_2)
$$

### Proof

$$
P(\min(X_1, X_2) > t) = P(X_1 > t) \cdot P(X_2 > t) = e^{-\lambda_1 t} \cdot e^{-\lambda_2 t} = e^{-(\lambda_1 + \lambda_2)t}
$$

This generalizes to $n$ independent exponentials: $\min(X_1, \ldots, X_n) \sim \text{Exp}\left(\sum_{i=1}^n \lambda_i\right)$.

---

## Worked Example

**Problem:** Orders arrive at a trading desk at an average rate of 12 per hour. What is the probability that the time between consecutive orders exceeds 10 minutes?

**Solution:** The rate is $\lambda = 12$ per hour $= 0.2$ per minute.

$$
P(X > 10) = e^{-0.2 \times 10} = e^{-2} \approx 0.1353
$$

Expected time between orders: $E[X] = 1/0.2 = 5$ minutes.

---

## Python: PDF, CDF, and Sampling

### PDF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lam = 2.0
x = np.linspace(0, 4, 200)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, stats.expon(scale=1/lam).pdf(x), label='PDF')
ax.plot(x, stats.expon(scale=1/lam).cdf(x), label='CDF')
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
for lam in [0.5, 1.0, 2.0, 5.0]:
    x = np.linspace(0, 6, 200)
    ax.plot(x, stats.expon(scale=1/lam).pdf(x), label=f'λ={lam}')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('x')
ax.legend()
plt.show()
```

### Sampling and Verification

```python
import numpy as np
from scipy import stats

np.random.seed(42)
lam = 3.0
samples = stats.expon(scale=1/lam).rvs(100_000)

print(f"Theoretical mean: {1/lam:.4f},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {1/lam**2:.4f},  Sample var:  {samples.var():.4f}")
print(f"Mean ≈ SD: {np.isclose(samples.mean(), samples.std(), atol=0.01)}")
```

### Verifying the Memoryless Property

```python
import numpy as np
from scipy import stats

np.random.seed(42)
lam = 2.0
samples = stats.expon(scale=1/lam).rvs(1_000_000)

s = 0.5
for t in [0.25, 0.5, 1.0]:
    conditional = np.mean(samples[samples > s] > s + t)
    unconditional = np.mean(samples > t)
    print(f"P(X>{s}+{t}|X>{s}) = {conditional:.4f},  P(X>{t}) = {unconditional:.4f}")
```

### Poisson Process Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
lam = 3.0  # events per unit time
n_events = 50

# Generate inter-arrival times
inter_arrivals = stats.expon(scale=1/lam).rvs(n_events)
arrival_times = np.cumsum(inter_arrivals)

fig, ax = plt.subplots(figsize=(12, 3))
ax.step(arrival_times, range(1, n_events + 1), where='post', lw=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative events')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Key Takeaways

- The exponential distribution models waiting times between events and is parameterized by rate $\lambda$ (or scale $1/\lambda$).
- It is the only continuous memoryless distribution: the remaining wait time is independent of how long you have already waited.
- It connects directly to the Poisson process: Poisson counts and exponential inter-arrival times are two views of the same phenomenon.
- The minimum of independent exponentials is again exponential, with rates summing.
- In SciPy, use `stats.expon(scale=1/lambda)` to work with rate parameterization.
