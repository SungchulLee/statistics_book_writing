# Axioms of Probability

## Overview

The axioms of probability formalize the intuitive idea of assigning "weights" (probabilities) to outcomes and events. We present three equivalent formulations—from the most intuitive to the most rigorous.

---

## Naive Axioms of Probability

These axioms capture the essential rules in an accessible form:

1. **Non-negativity:** For any event $A$,

$$
P(A) \geq 0
$$

2. **Normalization:** The probability of the entire sample space is 1:

$$
P(\Omega) = 1
$$

3. **Additivity:** For any two mutually exclusive events $A$ and $B$ (i.e., $A \cap B = \emptyset$),

$$
P(A \cup B) = P(A) + P(B)
$$

---

## Kolmogorov's Axioms of Probability

A **probability measure** $P$ is a real-valued function defined on events that satisfies:

$$
\begin{aligned}
(1) &\quad P(\Omega) = 1, \quad P(\emptyset) = 0 \\[6pt]
(2) &\quad 0 \leq P(A) \leq 1 \quad \text{for any event } A \\[6pt]
(3) &\quad P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \quad \text{for any sequence of disjoint events } A_i
\end{aligned}
$$

The key difference from the naive axioms is axiom (3): **countable additivity** extends the finite additivity rule to an infinite (countable) collection of disjoint events.

---

## Examples

### Example: Rolling an Even or Odd Number

Let $A = \{2, 4, 6\}$ (even) and $B = \{1, 3, 5\}$ (odd) when rolling a fair six-sided die. Since $A \cap B = \emptyset$:

$$
P(A \cup B) = P(A) + P(B) = \frac{3}{6} + \frac{3}{6} = 1
$$

This satisfies the normalization axiom since $A \cup B = \Omega$.

---

## Interpretation of Probability

### Probability of 0.7

A 0.7 probability of rain tomorrow means there is a 70% chance of rain. Out of 10 similar days with the same weather conditions, we would expect rain on about 7 of those days.

### Probability of 0.05

A 0.05 probability of drawing two aces in a row from a shuffled deck (without replacement) means a 5% chance—out of 100 repeated attempts, we would expect success about 5 times.

### Probability of 0

A probability of 0 means the event is impossible. For example, rolling a 7 on a standard six-sided die has probability 0 because that outcome is not in the sample space.

---

## Python Exploration

```python
import numpy as np

def verify_axioms(probabilities):
    """Verify Kolmogorov's axioms for a discrete probability distribution."""
    # Axiom 1: Non-negativity
    assert all(p >= 0 for p in probabilities), "Non-negativity violated"

    # Axiom 2: Normalization
    total = sum(probabilities)
    assert np.isclose(total, 1.0), f"Normalization violated: total = {total}"

    # Axiom 3: Additivity (verified by construction for disjoint events)
    print("All axioms satisfied!")
    print(f"  Total probability: {total:.4f}")
    print(f"  Min probability:   {min(probabilities):.4f}")
    print(f"  Max probability:   {max(probabilities):.4f}")

# Fair die
fair_die = [1/6] * 6
verify_axioms(fair_die)

# Loaded die
loaded_die = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
verify_axioms(loaded_die)
```

---

## Key Takeaways

- Kolmogorov's axioms provide the rigorous mathematical foundation for all of probability theory.
- The three axioms (normalization, non-negativity, countable additivity) are sufficient to derive all probability rules.
- Probability can be interpreted as long-run frequency (frequentist) or as a degree of belief (Bayesian).
