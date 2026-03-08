# Continuous Random Variables

## Overview

A **continuous random variable** can take on any value within a continuous range (an interval or union of intervals on the real line). Unlike discrete random variables, the probability of any single specific value is zero—instead, probabilities are defined over intervals.

---

## Definition

A **continuous random variable** $X$ can take on infinitely many possible values within a given range. Examples include heights, weights, temperatures, and waiting times.

For continuous random variables, the weight (probability) is **spread continuously** along the real line rather than concentrated at specific points.

---

## Probability Density Function (PDF)

For a continuous random variable $X$, the **PDF** $f(x)$ describes the density of probability at each point:

$$
f(x)\,dx = \text{Weight of the bricks within the interval } [x, x + dx]
$$

Key properties of the PDF:

1. $f(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f(x)\,dx = 1$
3. $P(a \leq X \leq b) = \int_a^b f(x)\,dx$

Note that $f(x)$ itself is **not** a probability—it can exceed 1. Only the **area** under the curve gives probabilities.

---

## Key Difference from Discrete Variables

For a **discrete** random variable, we can ask $P(X = a)$ and get a positive answer. For a **continuous** random variable:

$$
P(X = a) = 0 \quad \text{for any specific value } a
$$

This is because there are infinitely many possible values, and the "weight" at any single point is zero. We can only meaningfully ask about the probability that $X$ falls within a range.

---

## Example: Normal PDF

The most important continuous distribution is the **normal distribution** with mean $\mu$ and variance $\sigma^2$. Its PDF is:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

The area under this curve between any two values gives the probability that $X$ falls within that range.

---

## Example: Amelia's Maximum Average Wait Time

The distribution of average wait times at drive-through restaurants is approximately normal with mean $\mu = 185$ seconds and standard deviation $\sigma = 11$ seconds. Amelia only uses restaurants in the bottom 10% of wait times. What is her maximum acceptable wait time?

**Solution:**

We need the 10th percentile (PPF at 0.1):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mu = 185
sigma = 11

max_wait = stats.norm(loc=mu, scale=sigma).ppf(0.1)
print(f"Maximum average wait time: {max_wait:.2f} seconds")

# Visualization
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
pdf = stats.norm(loc=mu, scale=sigma).pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf)
x_fill = np.linspace(mu - 3*sigma, max_wait, 100)
ax.fill_between(x_fill, stats.norm(loc=mu, scale=sigma).pdf(x_fill),
                alpha=0.3, color='r', label=f'Bottom 10% (≤ {max_wait:.1f}s)')
ax.spines[['right', 'top']].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.legend()
plt.show()
```

---

## Comparing Discrete and Continuous Distributions

| Feature | Discrete | Continuous |
|:---|:---|:---|
| **Values** | Countable set | Uncountable (interval) |
| **Probability at a point** | $P(X = a) > 0$ possible | $P(X = a) = 0$ always |
| **Probability function** | PMF: $p_{x_i}$ | PDF: $f(x)$ |
| **Probability of a range** | $\sum_{x_i \in [a,b]} p_{x_i}$ | $\int_a^b f(x)\,dx$ |
| **Total probability** | $\sum_i p_{x_i} = 1$ | $\int_{-\infty}^{\infty} f(x)\,dx = 1$ |

---

## Key Takeaways

- Continuous random variables take values in an interval; the probability of any single point is zero.
- The PDF describes the "density" of probability—areas under the PDF curve give probabilities.
- The PDF can exceed 1 at specific points, but the total area under the curve is always 1.
- The normal distribution is the most widely used continuous distribution.
