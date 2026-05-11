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

### Example: Expected Number of Heads in n Coin Flips
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

## Exercises

**Exercise 1.**
A discrete random variable $X$ has the distribution: $P(X=-1) = 0.3$, $P(X=0) = 0.4$, $P(X=2) = 0.3$. Compute $E[X]$ and $E[X^2]$.

??? success "Solution to Exercise 1"
    $$
    E[X] = (-1)(0.3) + (0)(0.4) + (2)(0.3) = -0.3 + 0 + 0.6 = 0.3
    $$

    Using LOTUS for $g(X) = X^2$:

    $$
    E[X^2] = (-1)^2(0.3) + (0)^2(0.4) + (2)^2(0.3) = 0.3 + 0 + 1.2 = 1.5
    $$

---

**Exercise 2.**
Let $X_1, X_2, \ldots, X_{100}$ be the indicator variables for 100 independent coin flips, where $X_i = 1$ if the $i$-th flip is heads (probability 0.5) and $X_i = 0$ otherwise. Using linearity of expectation, find $E\!\left[\sum_{i=1}^{100} X_i\right]$.

??? success "Solution to Exercise 2"
    By linearity of expectation:

    $$
    E\!\left[\sum_{i=1}^{100} X_i\right] = \sum_{i=1}^{100} E[X_i]
    $$

    Each $X_i$ is a Bernoulli random variable with $E[X_i] = P(X_i = 1) = 0.5$. Therefore:

    $$
    E\!\left[\sum_{i=1}^{100} X_i\right] = 100 \times 0.5 = 50
    $$

    We expect 50 heads in 100 flips. Importantly, linearity holds regardless of whether the flips are independent — the same answer would apply even if the flips were dependent.

---

**Exercise 3.**
A fair six-sided die is rolled. Let $Y = (X - 3.5)^2$ where $X$ is the number showing. Compute $E[Y]$ using LOTUS.

??? success "Solution to Exercise 3"
    By LOTUS, $E[Y] = E[(X-3.5)^2] = \sum_{x=1}^{6} (x - 3.5)^2 \cdot P(X=x)$. Since $P(X=x) = 1/6$ for each value:

    $$
    E[Y] = \frac{1}{6}\left[(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2\right]
    $$

    $$
    = \frac{1}{6}\left[6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25\right] = \frac{17.5}{6} \approx 2.917
    $$

    Note: this is precisely $\text{Var}(X)$ for a fair die, since $E[X] = 3.5$.

---

**Exercise 4.**
Let $X$ and $Y$ be independent random variables with $E[X] = 2$, $E[Y] = 3$, $E[X^2] = 5$, and $E[Y^2] = 11$. Compute $E[XY]$ and $E[(X+Y)^2]$.

??? success "Solution to Exercise 4"
    Since $X$ and $Y$ are independent:

    $$
    E[XY] = E[X] \cdot E[Y] = 2 \times 3 = 6
    $$

    For $E[(X+Y)^2]$, expand the square:

    $$
    E[(X+Y)^2] = E[X^2 + 2XY + Y^2] = E[X^2] + 2E[XY] + E[Y^2]
    $$

    $$
    = 5 + 2(6) + 11 = 5 + 12 + 11 = 28
    $$

---

**Exercise 5.**
**Tail-sum formula for expectation.** Prove that for a non-negative random variable $X$, $\mathbb{E}[X] = \int_0^\infty P(X > t) dt$ (continuous) or $\sum_{n=0}^\infty P(X > n)$ (integer-valued).

??? success "Solution to Exercise 5"
    **Continuous case:** by Fubini's theorem:

    $$
    \int_0^\infty P(X > t) dt = \int_0^\infty \int_t^\infty f(x) dx \, dt = \int_0^\infty f(x) \int_0^x dt \, dx = \int_0^\infty x f(x) dx = \mathbb{E}[X]
    $$

    where the interchange is justified by non-negativity.

    **Integer-valued case:**

    $$
    \sum_{n=0}^\infty P(X > n) = \sum_{n=0}^\infty \sum_{k=n+1}^\infty P(X = k) = \sum_{k=1}^\infty P(X = k) \sum_{n=0}^{k-1} 1 = \sum_{k=1}^\infty k P(X = k) = \mathbb{E}[X]
    $$

    **Use:** the tail-sum formula lets you compute expectations from survival probabilities (sometimes easier than the standard PDF integral). Example: for a geometric random variable counting trials until first success, $P(X > n) = (1 - p)^n$, so $\mathbb{E}[X] = \sum_{n=0}^\infty (1 - p)^n = 1/p$.

---

**Exercise 6.**
**Conditional expectation as a random variable.** Let $X, Y$ be jointly distributed. Define $g(y) = \mathbb{E}[X \mid Y = y]$ and the random variable $\mathbb{E}[X \mid Y] = g(Y)$. Prove the **law of total expectation**: $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$.

??? success "Solution to Exercise 6"
    By definition, $g(y) = \mathbb{E}[X \mid Y = y] = \int x f_{X \mid Y}(x \mid y) dx$.

    $\mathbb{E}[g(Y)] = \int g(y) f_Y(y) dy = \int \int x f_{X \mid Y}(x \mid y) f_Y(y) dx \, dy = \int \int x f_{X, Y}(x, y) dx \, dy = \int x f_X(x) dx = \mathbb{E}[X]$.

    $\square$

    Used everywhere in probability, statistics, and dynamic-programming approaches to expectation:

    - **Conditional expectation as optimal predictor:** $\mathbb{E}[X \mid Y]$ minimizes $\mathbb{E}[(X - g(Y))^2]$ over all functions $g$.
    - **Towering** (iterated expectation): $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[\mathbb{E}[\mathbb{E}[X \mid Y, Z] \mid Y]]$, etc.
    - **MCMC / variance reduction**: replacing $X$ with $\mathbb{E}[X \mid Y]$ (when possible) reduces estimator variance via the Rao-Blackwell theorem.
    - **Reinforcement learning**: Bellman equation $V(s) = \mathbb{E}[R + \gamma V(s') \mid s]$ is iterated conditional expectation.
