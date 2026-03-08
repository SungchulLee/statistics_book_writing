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
