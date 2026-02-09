# Expectation and Linearity

## Overview

The **expected value** (or **expectation**) of a random variable is its long-run average value over many repetitions of an experiment. It provides a single number summarizing the "center" of a distribution. The **linearity of expectation** is one of the most powerful and widely used properties in all of probability.

---

## Definition

### Discrete Random Variables

For a discrete random variable $X$ with PMF $p_{x_i}$:

$$
E[X] = \sum_i x_i \cdot P(X = x_i) = \sum_i x_i \cdot p_{x_i}
$$

In the brick metaphor: $E[X]$ is the **center of mass** of the bricks placed along the real line.

### Continuous Random Variables

For a continuous random variable $X$ with PDF $f(x)$:

$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

---

## The Law of the Unconscious Statistician (LOTUS)

To compute the expected value of a function $g(X)$ without first finding the distribution of $g(X)$:

$$
E[g(X)] =
\begin{cases}
\displaystyle\sum_i g(x_i) \cdot P(X = x_i), & \text{discrete} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx, & \text{continuous}
\end{cases}
$$

This avoids the often tedious step of deriving the distribution of $g(X)$.

---

## Linearity of Expectation

For any random variables $X$ and $Y$ (not necessarily independent) and constants $a, b, c$:

$$
E[aX + bY + c] = aE[X] + bE[Y] + c
$$

This extends to any finite sum:

$$
E\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

**Key insight:** Linearity holds **regardless of whether the random variables are independent or dependent**. This makes it an exceptionally powerful tool.

---

## Properties of Expectation

1. **Constant:** $E[c] = c$
2. **Scaling:** $E[aX] = aE[X]$
3. **Additivity:** $E[X + Y] = E[X] + E[Y]$
4. **Monotonicity:** If $X \leq Y$ always, then $E[X] \leq E[Y]$
5. **Product (independent only):** If $X \perp\!\!\!\perp Y$, then $E[XY] = E[X] \cdot E[Y]$

Note that property 5 requires independence; properties 1–4 do not.

---

## Examples

### Example: Expected Value of a Fair Die

$$
E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = 3.5
$$

### Example: Expected Number of Heads in $n$ Coin Flips

Let $X_i = 1$ if flip $i$ is heads, 0 otherwise. Then $X = \sum_{i=1}^n X_i$ counts the total heads. By linearity:

$$
E[X] = \sum_{i=1}^n E[X_i] = \sum_{i=1}^n p = np
$$

For a fair coin with $n = 100$: $E[X] = 50$.

### Example: Coupon Collector Problem

There are $n$ distinct coupons. Each purchase gives a uniformly random coupon. Let $T$ be the total purchases needed to collect all $n$ coupons.

Divide the process into phases: phase $i$ begins when you have $i-1$ distinct coupons and ends when you get the $i$-th new one. In phase $i$, each purchase has probability $\frac{n - i + 1}{n}$ of being new, so the number of purchases in phase $i$ is geometric with mean $\frac{n}{n - i + 1}$.

By linearity:

$$
E[T] = \sum_{i=1}^{n} \frac{n}{n - i + 1} = n \sum_{k=1}^{n} \frac{1}{k} = nH_n \approx n \ln n
$$

For $n = 50$ types: $E[T] \approx 50 \times \ln(50) \approx 225$ purchases.

### Example: Continuous — Exponential Distribution

For $X \sim \text{Exponential}(\lambda)$ with PDF $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$:

$$
E[X] = \int_0^{\infty} x \cdot \lambda e^{-\lambda x} \, dx = \frac{1}{\lambda}
$$

---

## Python Exploration

```python
import numpy as np

# Expected value of a fair die
values = np.arange(1, 7)
probs = np.ones(6) / 6
expected = np.sum(values * probs)
print(f"E[fair die] = {expected:.4f}")

# Simulation
np.random.seed(42)
rolls = np.random.randint(1, 7, size=100_000)
print(f"Simulated mean = {rolls.mean():.4f}")
```

```python
import numpy as np

def coupon_collector_simulation(n_coupons, n_trials=10_000):
    """Simulate the coupon collector problem."""
    np.random.seed(42)
    totals = []
    for _ in range(n_trials):
        collected = set()
        count = 0
        while len(collected) < n_coupons:
            collected.add(np.random.randint(0, n_coupons))
            count += 1
        totals.append(count)

    simulated = np.mean(totals)
    H_n = sum(1/k for k in range(1, n_coupons + 1))
    theoretical = n_coupons * H_n

    print(f"n = {n_coupons}")
    print(f"Simulated E[T] = {simulated:.1f}")
    print(f"Theoretical E[T] = n·Hₙ = {theoretical:.1f}")

coupon_collector_simulation(50)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def linearity_demonstration():
    """Demonstrate linearity of expectation with dependent variables."""
    np.random.seed(42)
    n_sim = 100_000

    # X ~ Uniform(0,1), Y = X^2 (clearly dependent on X)
    X = np.random.rand(n_sim)
    Y = X ** 2

    print("X and Y = X² are dependent, but linearity still holds:")
    print(f"E[X] = {X.mean():.4f} (theoretical: 0.5)")
    print(f"E[Y] = {Y.mean():.4f} (theoretical: 0.3333)")
    print(f"E[X + Y] = {(X + Y).mean():.4f}")
    print(f"E[X] + E[Y] = {X.mean() + Y.mean():.4f}")

linearity_demonstration()
```

---

## Key Takeaways

- The expected value $E[X]$ is the probability-weighted average of all possible values.
- **LOTUS** lets us compute $E[g(X)]$ directly from the distribution of $X$.
- **Linearity of expectation** always holds, even for dependent variables—it is one of the most useful tools in probability.
- The product rule $E[XY] = E[X]E[Y]$ requires independence; linearity does not.
