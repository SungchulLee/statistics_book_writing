# Sample Spaces and Events

## Overview

Probability theory is built upon a set of fundamental rules known as **Kolmogorov's Axioms**, named after the Russian mathematician Andrey Kolmogorov. These axioms provide a formal foundation for reasoning about probability and ensure consistency when calculating the likelihood of events.

---

## Basic Definitions

### Sample

A possible outcome $\omega$ of an experiment is called a **sample**.

### Sample Space

The **sample space** $\Omega$ is the set of all possible outcomes (samples) of an experiment:

$$
\Omega = \{\omega_1, \omega_2, \omega_3, \ldots\}
$$

### Event

An **event** $A$ is any subset of $\Omega$. It represents a collection of outcomes of interest.

$$
A \subseteq \Omega
$$

---

## Intuitive Picture: Bricks and Weights

For each outcome $\omega \in \Omega$, we attach a "brick" with a certain weight representing its probability. Different bricks may have different weights, but the total weight of all bricks across the sample space is 1. This weight distribution over $\Omega$ defines a **probability measure**:

$$
\begin{aligned}
P(\omega) &= \text{Weight of the brick attached to } \omega \\
P(A) &= \sum_{\omega \in A} P(\omega) = \text{Total weight of the bricks attached to } A
\end{aligned}
$$

---

## Examples

### Example: Rolling a Six-Sided Die

When rolling a fair six-sided die:

- **Sample space:** $\Omega = \{1, 2, 3, 4, 5, 6\}$
- **Event $A$ (rolling an even number):** $A = \{2, 4, 6\}$
- **Event $B$ (rolling an odd number):** $B = \{1, 3, 5\}$

Since the die is fair, each outcome has equal probability $P(\omega) = \frac{1}{6}$.

### Example: Flipping Three Coins

When flipping three coins:

- **Sample space:** $\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}$
- **Event (exactly 2 heads):** $A = \{HHT, HTH, THH\}$
- $P(A) = \frac{3}{8}$

---

## Python Exploration

```python
from itertools import product

# Sample space for three coin flips
sample_space = list(product(['H', 'T'], repeat=3))
print(f"Sample space size: {len(sample_space)}")
print(f"Sample space: {sample_space}")

# Event: exactly 2 heads
event_2_heads = [s for s in sample_space if s.count('H') == 2]
print(f"\nEvent (2 heads): {event_2_heads}")
print(f"P(2 heads) = {len(event_2_heads)}/{len(sample_space)} = {len(event_2_heads)/len(sample_space):.4f}")
```

---

## Key Takeaways

- The **sample space** $\Omega$ captures every possible outcome of an experiment.
- An **event** is any subset of $\Omega$.
- Probability assigns a non-negative "weight" to each outcome such that the total weight is 1.
- The probability of an event is the sum of weights of all outcomes in that event.
