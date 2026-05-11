# Exponential Density Function

## Overview

The **exponential distribution** models the waiting time until the first event in a Poisson process. It is the continuous analogue of the geometric distribution and the only continuous distribution with the **memoryless property**.

The PDF with rate parameter $\lambda > 0$ is:

$$
f(x) = \lambda\, e^{-\lambda x}, \qquad x \ge 0
$$

| Property | Value |
|---|---|
| Mean | $1/\lambda$ |
| Variance | $1/\lambda^2$ |
| Median | $\ln(2)/\lambda$ |
| Mode | 0 |

---

## SciPy Parameterization

SciPy uses the **scale** parameterization: `stats.expon(scale=1/lambda)`.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

lambdas = [0.5, 1.0, 2.0]
x = np.linspace(0, 6, 300)

fig, ax = plt.subplots(figsize=(12, 4))
for lam in lambdas:
    rv = stats.expon(scale=1 / lam)
    ax.plot(x, rv.pdf(x), label=rf'$\lambda={lam}$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Exponential Distribution — PDF')
ax.legend()
plt.tight_layout()
plt.show()
```

Higher $\lambda$ means events occur more frequently, so the distribution is more concentrated near zero.

---

## The Memoryless Property

The exponential distribution satisfies:

$$
P(X > s + t \mid X > s) = P(X > t) \qquad \text{for all } s, t \ge 0
$$

This means the remaining waiting time is independent of how long you have already waited—the process has no memory.

---

## Exercises

**Exercise 1.**
Prove the memoryless property of the exponential distribution.

??? success "Solution to Exercise 1"
    The survival function is $S(x) = e^{-\lambda x}$. Then:

    $$
    P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
    $$

    $\square$

---

**Exercise 2.**
If customer arrivals follow a Poisson process with rate $\lambda = 3$ per hour, what is the probability of waiting more than 30 minutes for the next customer?

??? success "Solution to Exercise 2"
    The inter-arrival time is $X \sim \text{Exp}(\lambda = 3)$ (in hours). Thirty minutes is $t = 0.5$ hours.

    $$
    P(X > 0.5) = e^{-3 \times 0.5} = e^{-1.5} \approx 0.2231
    $$

---

**Exercise 3.**
Show that the exponential distribution is the only continuous memoryless distribution.

??? success "Solution to Exercise 3"
    Suppose $P(X > s + t) = P(X > s) \cdot P(X > t)$ for all $s, t \ge 0$. Let $g(t) = P(X > t)$. Then $g(s+t) = g(s)g(t)$ with $g(0) = 1$ and $g$ decreasing.

    The only continuous solution to the functional equation $g(s+t) = g(s)g(t)$ with $g(0) = 1$ is $g(t) = e^{-\lambda t}$ for some $\lambda > 0$. This is the survival function of the exponential distribution. $\square$

---

**Exercise 4.**
Derive the CDF, median, and mean of the exponential distribution from the PDF.

??? success "Solution to Exercise 4"
    **CDF:**

    $$
    F(x) = \int_0^x \lambda e^{-\lambda t}\,dt = 1 - e^{-\lambda x}
    $$

    **Median:** Solve $F(m) = 0.5$:

    $$
    1 - e^{-\lambda m} = 0.5 \implies m = \frac{\ln 2}{\lambda}
    $$

    **Mean:**

    $$
    E[X] = \int_0^{\infty} x\lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}
    $$

    (via integration by parts).
