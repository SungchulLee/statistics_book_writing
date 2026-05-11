# Discrete Distributions Suite

## Overview

This page demonstrates four key discrete distributions with practical examples: **Binomial**, **Poisson**, **Geometric**, and **Hypergeometric**. Each models a different type of counting problem, and understanding when to use which is essential for applied statistics.

---

## 1. Binomial Distribution

The number of successes in $n$ independent Bernoulli trials, each with success probability $p$:

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, \ldots, n
$$

**Example:** A banker meets 50 loan applicants per month. 30% have bad credit.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

n, p = 50, 0.3
k_vals = np.arange(0, n + 1)
pmf = stats.binom.pmf(k_vals, n, p)

print(f"P(X = 14) = {stats.binom.pmf(14, n, p):.4f}")
print(f"P(X <= 12) = {stats.binom.cdf(12, n, p):.4f}")
print(f"E[X] = {n*p:.1f}, Var(X) = {n*p*(1-p):.2f}")
```

---

## 2. Poisson Distribution

The count of rare events in a fixed interval, with mean rate $\lambda$:

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \ldots
$$

**Example:** A trader makes 1200 trades over 5 years. Wipe-out probability per trade is 1/1000, giving $\lambda = 1.2$.

```python
lam = 1.2
print(f"P(X = 2) = {stats.poisson.pmf(2, lam):.4f}")
print(f"P(X > 2) = {1 - stats.poisson.cdf(2, lam):.4f}")
```

---

## 3. Geometric Distribution

The number of failures before the first success in independent Bernoulli trials:

$$
P(X = k) = (1-p)^k \, p, \qquad k = 0, 1, 2, \ldots
$$

**Example:** Success probability $p = 0.3$. Expected failures before first success: $(1-p)/p \approx 2.33$.

```python
p = 0.3
print(f"P(5 failures before 1st success) = {(1-p)**5 * p:.4f}")
print(f"E[failures] = {(1-p)/p:.2f}")
```

---

## 4. Hypergeometric Distribution

The number of successes when drawing $n$ items **without replacement** from a population of $N$ with $K$ successes:

$$
P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
$$

**Example:** Population $N = 100$, defectives $K = 20$, draw $n = 5$.

```python
from scipy import special

N, K, n = 100, 20, 5
p2 = stats.hypergeom.pmf(2, N, K, n)
p2_manual = (special.comb(K, 2) * special.comb(N-K, n-2)) / special.comb(N, n)
print(f"P(X = 2) = {p2:.4f}  (manual = {p2_manual:.4f})")
print(f"E[X] = {n*K/N:.2f}")
```

---

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(k_vals, pmf, color="steelblue", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(n * p, color="red", linestyle="--", label=f"E[X] = {n*p:.0f}")
axes[0, 0].set_title("Binomial(n=50, p=0.3)")
axes[0, 0].set_xlabel("k")
axes[0, 0].legend()

pk = np.arange(0, 10)
axes[0, 1].bar(pk, stats.poisson.pmf(pk, lam), color="seagreen",
               edgecolor="white", alpha=0.8)
axes[0, 1].set_title(f"Poisson(λ={lam})")
axes[0, 1].set_xlabel("k")

gk = np.arange(0, 20)
axes[1, 0].bar(gk, stats.geom.pmf(gk + 1, 0.3), color="coral",
               edgecolor="white", alpha=0.8)
axes[1, 0].set_title("Geometric(p=0.3)")
axes[1, 0].set_xlabel("k (failures)")

hk = np.arange(0, n + 1)
axes[1, 1].bar(hk, stats.hypergeom.pmf(hk, N, K, n), color="mediumpurple",
               edgecolor="white", alpha=0.8)
axes[1, 1].set_title("Hypergeometric(N=100, K=20, n=5)")
axes[1, 1].set_xlabel("k (defectives)")

plt.tight_layout()
plt.show()
```

---

## Exercises

**Exercise 1.**
Show that the Poisson distribution is the limit of the Binomial as $n \to \infty$ and $p \to 0$ with $np = \lambda$ fixed.

??? success "Solution to Exercise 1"
    Start with $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$ where $p = \lambda/n$:

    $$
    \binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1 - \frac{\lambda}{n}\right)^{n-k}
    $$

    As $n \to \infty$: $\binom{n}{k}/n^k \to 1/k!$ and $(1-\lambda/n)^n \to e^{-\lambda}$ and $(1-\lambda/n)^{-k} \to 1$. So:

    $$
    P(X = k) \to \frac{\lambda^k}{k!} e^{-\lambda}
    $$

    which is the Poisson PMF. $\square$

---

**Exercise 2.**
A quality inspector draws 5 items from a batch of 100 (20 defective). Compare $P(X = 2)$ using the hypergeometric (exact) and binomial (approximate) distributions. When is the binomial a good approximation?

??? success "Solution to Exercise 2"
    **Hypergeometric:** $P(X=2) = \binom{20}{2}\binom{80}{3}/\binom{100}{5} \approx 0.2075$.

    **Binomial** ($n=5$, $p=0.2$): $P(X=2) = \binom{5}{2}(0.2)^2(0.8)^3 = 10 \times 0.04 \times 0.512 = 0.2048$.

    The approximation is close because $n/N = 5/100 = 5\%$ is small. Rule of thumb: the binomial approximates the hypergeometric well when the sample is less than 5–10% of the population.

---

**Exercise 3.**
Prove that the geometric distribution is memoryless: $P(X > s + t \mid X > s) = P(X > t)$.

??? success "Solution to Exercise 3"
    Here $X$ counts the number of failures before the first success. $P(X > k) = (1-p)^{k+1}$ (no success in the first $k+1$ trials). Then:

    $$
    P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{(1-p)^{s+t+1}}{(1-p)^{s+1}} = (1-p)^t = \frac{P(X > t)}{1}
    $$

    Wait — more carefully: if $X$ is the number of failures (starting from 0), then $P(X \ge k) = (1-p)^k$. So:

    $$
    P(X \ge s+t \mid X \ge s) = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X \ge t)
    $$

    $\square$

---

**Exercise 4.**
The expected value of the hypergeometric is $E[X] = nK/N$. Derive this result.

??? success "Solution to Exercise 4"
    Write $X = \sum_{i=1}^n X_i$ where $X_i = 1$ if the $i$-th drawn item is defective. By symmetry, $P(X_i = 1) = K/N$ for each $i$ (each item is equally likely to be any of the $N$ items). By linearity of expectation:

    $$
    E[X] = \sum_{i=1}^n E[X_i] = n \cdot \frac{K}{N}
    $$

    Note: the $X_i$ are not independent (sampling without replacement), but linearity of expectation does not require independence. $\square$
