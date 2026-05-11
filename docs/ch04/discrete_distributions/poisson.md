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

## Exercises

**Exercise 1.**
Call center receives 4 calls/min. (a) Distribution of $N$? (b) $P(N = 0)$, $P(N \ge 6)$. (c) $P(>10 \text{ calls in 2 min})$. (d) Normal approximation to $P(N \ge 8)$.

??? success "Solution to Exercise 1"
    (a) $N \sim \mathrm{Poisson}(4)$.

    (b) $P(N = 0) = e^{-4} \approx 0.018$. Computing partial sum $P(N \le 5) \approx 0.785$, so $P(N \ge 6) \approx 0.215$.

    (c) In 2 min, $N_2 \sim \mathrm{Poisson}(8)$ (additivity). $P(N_2 > 10) = 1 - P(N_2 \le 10) \approx 0.184$.

    (d) Normal approximation: $\mu = 4$, $\sigma = 2$. With continuity correction: $P(N \ge 8) \approx P(Z \ge (7.5 - 4)/2) = P(Z \ge 1.75) = 0.040$. Exact: $0.051$. The approximation is rough at $\lambda = 4$ (still right-skewed); accuracy improves substantially for $\lambda \ge 30$.

---

**Exercise 2.**
Poisson approximation: 500 pages, each independently has misprint with $p = 0.004$. (a) Exact distribution? (b) Poisson approximation parameter? (c) $P(X = 0)$, $P(X = 1)$, $P(X \ge 4)$ via Poisson.

??? success "Solution to Exercise 2"
    (a) $X \sim \mathrm{Binomial}(500, 0.004)$.

    (b) $\lambda = np = 2$. Poisson approximation valid since $n$ large, $p$ small.

    (c) $P(X = 0) \approx e^{-2} = 0.135$. $P(X = 1) \approx 2 e^{-2} = 0.271$. $P(X \ge 4) = 1 - e^{-2}(1 + 2 + 2 + 4/3) \approx 0.143$.

    Exact binomial gives $P(X = 2) \approx 0.272$ vs Poisson $0.271$ — agreement to 3 decimal places, demonstrating Poisson approximation works very well in this regime.

---

**Exercise 3.**
**Memoryless property of inter-arrival times.** If events occur according to a Poisson process with rate $\lambda$, prove inter-arrival times are exponential with rate $\lambda$.

??? success "Solution to Exercise 3"
    Let $T_1$ be the time of the first event. The event $\{T_1 > t\}$ is equivalent to "no events in $[0, t]$" — which has probability $P(N(t) = 0) = e^{-\lambda t}$.

    So $P(T_1 > t) = e^{-\lambda t}$, the survival function of $\mathrm{Exp}(\lambda)$. Hence $T_1 \sim \mathrm{Exp}(\lambda)$.

    By the Poisson process's stationary increments property, the time between event $k$ and event $k+1$ has the same distribution, independent of past. All inter-arrival times are i.i.d. $\mathrm{Exp}(\lambda)$. $\square$

    This connection makes Poisson processes the canonical model for "completely random" event occurrence — events that arrive at a constant rate without memory of history.

---

**Exercise 4.**
**Sum and superposition.** Independent Poisson processes with rates $\lambda_1, \lambda_2$ are superposed. Show the combined process is Poisson with rate $\lambda_1 + \lambda_2$.

??? success "Solution to Exercise 4"
    Let $N_1(t), N_2(t)$ be independent Poisson processes. The combined count is $N(t) = N_1(t) + N_2(t)$.

    Marginal: $N(t) = N_1(t) + N_2(t) \sim \mathrm{Poisson}(\lambda_1 t) + \mathrm{Poisson}(\lambda_2 t) \sim \mathrm{Poisson}((\lambda_1 + \lambda_2) t)$ by the Poisson sum property.

    The combined inter-arrival times are exponential with rate $\lambda_1 + \lambda_2$ (the minimum of two independent exponentials is exponential at the sum of rates), and they remain independent across arrivals.

    Together, these properties confirm $N$ is a Poisson process with rate $\lambda_1 + \lambda_2$. $\square$

    **Application:** if customer arrivals at a store split into two types (online vs. in-person) and each is Poisson, total arrivals form a Poisson process with combined rate. This justifies aggregating Poisson sources into a single model.

---

**Exercise 5.**
**Variance equals mean** for Poisson. Compute $\mathbb{E}[X], \mathbb{E}[X^2]$ for $X \sim \mathrm{Poisson}(\lambda)$ directly from the PMF.

??? success "Solution to Exercise 5"
    **Mean:**

    $$
    \mathbb{E}[X] = \sum_{k=0}^\infty k \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \lambda \sum_{k=1}^\infty \frac{\lambda^{k-1}}{(k-1)!} = \lambda
    $$

    using $\sum_{j=0}^\infty \lambda^j/j! = e^\lambda$.

    **Second factorial moment** $\mathbb{E}[X(X-1)] = \sum k(k-1) P(X=k) = e^{-\lambda} \lambda^2 \sum_{k=2}^\infty \lambda^{k-2}/(k-2)! = \lambda^2$.

    So $\mathbb{E}[X^2] = \mathbb{E}[X(X-1)] + \mathbb{E}[X] = \lambda^2 + \lambda$.

    **Variance:** $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda$. $\square$

    **Distinguishing characteristic:** mean = variance is a Poisson signature. When real count data shows variance > mean (**overdispersion**), the Poisson model is inadequate and a negative binomial model is typically used instead.

---

**Exercise 6.**
**Test for Poisson assumption.** Given a sample of counts $X_1, \ldots, X_n$, propose a simple **dispersion test** for the Poisson assumption based on the ratio of sample variance to sample mean.

??? success "Solution to Exercise 6"
    Under $H_0$: $X_i$ are i.i.d. $\mathrm{Poisson}(\lambda)$, $\mathrm{Var}(X) = \lambda = \mathbb{E}[X]$, so the **dispersion ratio** $D = s^2/\bar X$ should be near 1.

    Test statistic: $(n - 1) D = (n - 1) s^2 / \bar X$. Under $H_0$, this is approximately $\chi^2_{n-1}$ (the **Poisson dispersion test**, derived under the assumption that the sample mean is approximately the rate).

    **Decision rule:** reject Poisson if $(n - 1)D$ falls outside the $\alpha/2$ and $1 - \alpha/2$ quantiles of $\chi^2_{n-1}$. Specifically:

    - $D \gg 1$ (overdispersion): variance exceeds mean. Suggests negative binomial or quasi-Poisson alternative.
    - $D \ll 1$ (underdispersion): variance below mean. Suggests Conway-Maxwell-Poisson or truncated distributions.

    Modern count-data analysis often skips Poisson entirely in favor of more flexible models (negative binomial, zero-inflated Poisson, hurdle models). The Poisson is more useful as a *building block* (in Poisson processes) than as a flexible fitting tool.
