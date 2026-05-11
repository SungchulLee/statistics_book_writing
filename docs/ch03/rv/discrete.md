# Discrete Random Variables

## Overview

A **random variable** is a function that maps outcomes from a sample space to real numbers. A **discrete random variable** takes on a countable number of distinct values.

---

## Definition

A **random variable** $X$ is formally defined as a function:

$$
X : \Omega \longrightarrow \mathbb{R}
$$

where $\Omega$ is the sample space and $\mathbb{R}$ is the set of real numbers.

A **discrete random variable** takes on a countable set of distinct values $\{x_1, x_2, x_3, \ldots\}$. Examples include the result of rolling a die or the number of heads in a series of coin flips.

---

## Distribution of a Discrete Random Variable

Imagine each outcome $\omega \in \Omega$ as having a "brick" of a certain weight attached to it, representing the probability of that outcome. When we apply the random variable $X$, we move the brick from $\omega$ to the position $X(\omega)$ on the real line.

After transferring all bricks, the arrangement of weights along $\mathbb{R}$ defines the **distribution of $X$**:

$$
\begin{aligned}
\mathbb{P}(X = a) &= \text{Weight of the bricks at } a \\
\mathbb{P}(X \in A) &= \text{Weight of the bricks in the set } A
\end{aligned}
$$

---

## Probability Mass Function (PMF)

For a discrete random variable $X$, the **PMF** assigns a probability to each specific value:

$$
p_{x_i} = P(X = x_i) = \text{Weight of the brick at } x_i
$$

The PMF must satisfy:

1. $p_{x_i} \geq 0$ for all $i$
2. $\sum_i p_{x_i} = 1$

---

## Examples

### Example: PMF of a Fair Die

Let $X$ represent the outcome of rolling a fair six-sided die:

$$
P(X = x) = \frac{1}{6}, \quad \text{for } x = 1, 2, 3, 4, 5, 6
$$

### Example: Number of Heads in 3 Coin Flips

Let $X$ represent the number of heads when flipping a fair coin 3 times. The possible values are $\{0, 1, 2, 3\}$:

$$
\begin{aligned}
P(X = 0) &= \frac{1}{8} \\
P(X = 1) &= \frac{3}{8} \\
P(X = 2) &= \frac{3}{8} \\
P(X = 3) &= \frac{1}{8}
\end{aligned}
$$

### Example: Baseball Cards

Hugo plans to purchase packs of baseball cards until he obtains his favorite player's card. He can afford at most four packs, and each pack has a 0.2 probability of containing the card. Let $X$ be the number of packs Hugo buys.

**Solution:**

$$
\begin{aligned}
P(X=1) &= 0.2 \\
P(X=2) &= 0.8 \times 0.2 = 0.16 \\
P(X=3) &= 0.8^2 \times 0.2 = 0.128 \\
P(X=4) &= 1 - P(X=1) - P(X=2) - P(X=3) = 0.512
\end{aligned}
$$

Therefore:

$$
\begin{aligned}
P(X \geq 2) &= 1 - P(X=1) = 0.8 \\
P(X = 4) &= 0.512
\end{aligned}
$$

Note that $P(X=4) = 0.512$ includes both the probability of finding the card on the 4th pack and the probability of never finding it—Hugo stops at 4 regardless.

### Example: Difference on Two 3-Sided Dice

Let $D = |D_1 - D_2|$ where $D_1, D_2$ are rolls of 3-sided dice. The nine equally likely outcomes yield:

| $D_1 \backslash D_2$ | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| **1** | 0 | 1 | 2 |
| **2** | 1 | 0 | 1 |
| **3** | 2 | 1 | 0 |

$$
P(D=0) = \frac{3}{9}, \quad P(D=1) = \frac{4}{9}, \quad P(D=2) = \frac{2}{9}
$$

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_pmf(values, probabilities, title="PMF"):
    """Plot the probability mass function."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(values, probabilities, width=0.4, alpha=0.7, edgecolor='black')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X = x)')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Fair die PMF
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
plot_pmf(values, probs, "PMF of a Fair Die")

# Coin flip PMF (3 flips, counting heads)
from math import comb
n = 3
values = list(range(n + 1))
probs = [comb(n, k) * (0.5**k) * (0.5**(n-k)) for k in values]
plot_pmf(values, probs, "PMF: Number of Heads in 3 Coin Flips")

# Baseball cards PMF
values = [1, 2, 3, 4]
probs = [0.2, 0.16, 0.128, 0.512]
plot_pmf(values, probs, "PMF: Baseball Card Packs Purchased")
```

---

## Key Takeaways

- A discrete random variable maps outcomes to a countable set of real numbers.
- The PMF gives the probability of each possible value and must sum to 1.
- The "brick" metaphor provides intuition: each outcome carries a weight (probability), and the random variable relocates these weights to the real line.

## Exercises

**Exercise 1.**
A discrete random variable $X$ has the PMF: $P(X=0) = 0.1$, $P(X=1) = 0.3$, $P(X=2) = c$, $P(X=3) = 0.2$. Find the value of $c$ and compute $P(X \geq 2)$.

??? success "Solution to Exercise 1"
    Since the PMF must sum to 1:

    $$
    0.1 + 0.3 + c + 0.2 = 1 \implies c = 0.4
    $$

    Therefore:

    $$
    P(X \geq 2) = P(X=2) + P(X=3) = 0.4 + 0.2 = 0.6
    $$

---

**Exercise 2.**
Two fair four-sided dice (with faces 1, 2, 3, 4) are rolled. Let $S$ be the sum of the two dice. Write out the PMF of $S$ and verify that the probabilities sum to 1.

??? success "Solution to Exercise 2"
    There are $4 \times 4 = 16$ equally likely outcomes. The possible sums range from 2 to 8:

    | $s$ | Outcomes | $P(S = s)$ |
    |:---:|:---|:---:|
    | 2 | $(1,1)$ | $1/16$ |
    | 3 | $(1,2),(2,1)$ | $2/16$ |
    | 4 | $(1,3),(2,2),(3,1)$ | $3/16$ |
    | 5 | $(1,4),(2,3),(3,2),(4,1)$ | $4/16$ |
    | 6 | $(2,4),(3,3),(4,2)$ | $3/16$ |
    | 7 | $(3,4),(4,3)$ | $2/16$ |
    | 8 | $(4,4)$ | $1/16$ |

    Verification: $1 + 2 + 3 + 4 + 3 + 2 + 1 = 16$, so $\sum P(S=s) = 16/16 = 1$. $\square$

---

**Exercise 3.**
A loaded coin has $P(\text{Heads}) = 0.7$. The coin is flipped 3 times. Let $X$ be the number of heads. Write the PMF of $X$.

??? success "Solution to Exercise 3"
    Each flip is independent with $p = 0.7$ (heads) and $q = 0.3$ (tails). The number of heads in 3 flips follows a Binomial distribution:

    $$
    P(X = k) = \binom{3}{k} (0.7)^k (0.3)^{3-k}
    $$

    Computing each value:

    $$
    P(X=0) = \binom{3}{0}(0.7)^0(0.3)^3 = 0.027
    $$

    $$
    P(X=1) = \binom{3}{1}(0.7)^1(0.3)^2 = 3 \times 0.063 = 0.189
    $$

    $$
    P(X=2) = \binom{3}{2}(0.7)^2(0.3)^1 = 3 \times 0.147 = 0.441
    $$

    $$
    P(X=3) = \binom{3}{3}(0.7)^3(0.3)^0 = 0.343
    $$

    Check: $0.027 + 0.189 + 0.441 + 0.343 = 1.000$. $\square$

---

**Exercise 4.**
Explain why a continuous random variable (e.g., the exact height of a randomly selected person) cannot be described by a PMF. What replaces the PMF in the continuous case?

??? success "Solution to Exercise 4"
    A PMF assigns positive probability to individual values: $P(X = x) > 0$ for each value in the support. For a continuous random variable, the support is an uncountable interval (e.g., all real numbers in $[150, 200]$ cm). If every individual value had positive probability, the sum (or integral) over uncountably many values would diverge to infinity, violating the normalization axiom $P(\Omega) = 1$.

    Instead, a continuous random variable is described by a **probability density function (PDF)** $f(x)$, where $f(x) \geq 0$ and $\int_{-\infty}^{\infty} f(x)\,dx = 1$. The PDF gives density, not probability: $P(X = x) = 0$ for any single value, but $P(a \leq X \leq b) = \int_a^b f(x)\,dx$ gives the probability over an interval.

---

**Exercise 5.**
**Geometric distribution.** $X$ = number of trials until first success in i.i.d. Bernoulli($p$). Derive PMF, $\mathbb{E}[X]$, and $\mathrm{Var}(X)$.

??? success "Solution to Exercise 5"
    **PMF:** $X = k$ requires $k - 1$ failures then a success: $P(X = k) = (1 - p)^{k-1} p$ for $k = 1, 2, \ldots$.

    Normalization: $\sum_{k=1}^\infty (1-p)^{k-1} p = p/p = 1$. ✓

    **Expectation:** by tail-sum formula, $\mathbb{E}[X] = \sum_{n=0}^\infty P(X > n) = \sum_{n=0}^\infty (1-p)^n = 1/p$.

    **Variance:** $\mathrm{Var}(X) = (1-p)/p^2$ (derivation by similar calculation involving $\mathbb{E}[X(X-1)]$).

    For $p = 0.5$: mean 2 trials, variance 2, SD $\sqrt 2$. Geometric is the discrete analog of the exponential and inherits the memoryless property: $P(X > m + n \mid X > m) = P(X > n)$.

---

**Exercise 6.**
**Poisson approximation to the binomial.** Show that as $n \to \infty$ with $np \to \lambda$ constant, Binomial$(n, p)$ → Poisson$(\lambda)$.

??? success "Solution to Exercise 6"
    Substitute $p = \lambda/n$ in the binomial PMF:

    $$
    P(X = k) = \binom{n}{k}\left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}
    $$

    Rearrange:

    $$
    = \frac{\lambda^k}{k!} \cdot \underbrace{\frac{n!}{(n-k)! n^k}}_{\to 1} \cdot \underbrace{\left(1 - \frac{\lambda}{n}\right)^n}_{\to e^{-\lambda}} \cdot \underbrace{\left(1 - \frac{\lambda}{n}\right)^{-k}}_{\to 1}
    $$

    As $n \to \infty$: $P(X = k) \to \frac{\lambda^k e^{-\lambda}}{k!}$ — the Poisson PMF.

    **Use:** for rare events ($p$ small, $n$ large), Poisson is much simpler than binomial. Applications: number of car accidents per day in a city, defects per chip, calls per minute, mutations per genome.

    Rule of thumb: Poisson works well when $n \ge 20$, $p \le 0.05$, and $np \le 10$. Otherwise stick with the binomial or use the normal approximation if $np \ge 10$.
