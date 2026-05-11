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

## Exercises

**Exercise 1.**
A bag contains 3 red balls and 2 blue balls. Two balls are drawn without replacement. Write out the sample space $\Omega$ using ordered pairs (e.g., $(R_1, B_1)$) and compute the probability of drawing two red balls.

??? success "Solution to Exercise 1"
    Label the balls $R_1, R_2, R_3, B_1, B_2$. The sample space of ordered draws is:

    $$
    \Omega = \{(R_1,R_2),(R_1,R_3),(R_1,B_1),(R_1,B_2),(R_2,R_1),(R_2,R_3),(R_2,B_1),(R_2,B_2),
    $$

    $$
    (R_3,R_1),(R_3,R_2),(R_3,B_1),(R_3,B_2),(B_1,R_1),(B_1,R_2),(B_1,R_3),(B_1,B_2),
    $$

    $$
    (B_2,R_1),(B_2,R_2),(B_2,R_3),(B_2,B_1)\}
    $$

    There are $5 \times 4 = 20$ equally likely ordered outcomes. The event "two red balls" is $A = \{(R_1,R_2),(R_1,R_3),(R_2,R_1),(R_2,R_3),(R_3,R_1),(R_3,R_2)\}$, which has 6 outcomes. Therefore:

    $$
    P(A) = \frac{6}{20} = \frac{3}{10}
    $$

---

**Exercise 2.**
An experiment consists of rolling a fair die and flipping a fair coin. Write out the sample space. Define the event $A$ = "the die shows an even number and the coin shows heads." Compute $P(A)$.

??? success "Solution to Exercise 2"
    The sample space is the Cartesian product of die outcomes and coin outcomes:

    $$
    \Omega = \{(1,H),(1,T),(2,H),(2,T),(3,H),(3,T),(4,H),(4,T),(5,H),(5,T),(6,H),(6,T)\}
    $$

    There are $6 \times 2 = 12$ equally likely outcomes.

    The event $A$ = {even die and heads} = $\{(2,H),(4,H),(6,H)\}$, which has 3 outcomes.

    $$
    P(A) = \frac{3}{12} = \frac{1}{4}
    $$

---

**Exercise 3.**
Let $\Omega = \{a, b, c\}$ with $P(a) = 0.5$, $P(b) = 0.3$, $P(c) = 0.2$. List all possible events (subsets of $\Omega$) and compute the probability of each.

??? success "Solution to Exercise 3"
    A set with 3 elements has $2^3 = 8$ subsets:

    | Event | Probability |
    |:---|:---|
    | $\emptyset$ | $0$ |
    | $\{a\}$ | $0.5$ |
    | $\{b\}$ | $0.3$ |
    | $\{c\}$ | $0.2$ |
    | $\{a, b\}$ | $0.5 + 0.3 = 0.8$ |
    | $\{a, c\}$ | $0.5 + 0.2 = 0.7$ |
    | $\{b, c\}$ | $0.3 + 0.2 = 0.5$ |
    | $\{a, b, c\} = \Omega$ | $0.5 + 0.3 + 0.2 = 1.0$ |

---

**Exercise 4.**
Explain the difference between a sample space that is finite, countably infinite, and uncountable. Give one example of an experiment for each type.

??? success "Solution to Exercise 4"
    **Finite sample space:** The set of outcomes is finite. Example: rolling a die gives $\Omega = \{1,2,3,4,5,6\}$ with $|\Omega| = 6$.

    **Countably infinite sample space:** The set of outcomes can be put in one-to-one correspondence with the natural numbers. Example: flipping a coin until the first heads appears gives $\Omega = \{H, TH, TTH, TTTH, \ldots\}$. There are infinitely many outcomes (one for each possible number of tails before the first heads), but they can be enumerated.

    **Uncountable sample space:** The set of outcomes has the cardinality of the real numbers and cannot be enumerated. Example: spinning a perfectly balanced spinner and recording the angle gives $\Omega = [0, 360)$, which is an uncountable set. Any single angle has probability zero, and probabilities are assigned to intervals.

---

**Exercise 5.**
For a sample space $\Omega$ with $n$ elements, the **power set** $2^\Omega$ has $2^n$ elements. Prove this by induction. Why is the power set the natural "event space" for discrete probability?

??? success "Solution to Exercise 5"
    **Base case:** $n = 0$, $\Omega = \emptyset$ has only the empty set as subset, so $|2^\Omega| = 1 = 2^0$. ✓

    **Inductive step:** assume $|2^A| = 2^k$ for any set $A$ with $k$ elements. Let $\Omega$ have $k + 1$ elements, picking one element $x$. Every subset of $\Omega$ either contains $x$ (there are as many such subsets as subsets of $\Omega \setminus \{x\}$, namely $2^k$) or does not contain $x$ (also $2^k$). Total: $2 \cdot 2^k = 2^{k+1}$. $\square$

    **Why power set:** for discrete $\Omega$, we want every subset to be a measurable event — assigning probability to any combination of outcomes should be allowed. The power set is the unique $\sigma$-algebra containing all singletons in a discrete setting.

    For continuous $\Omega$ (uncountable), the power set is too large to admit a $\sigma$-additive measure. We use the **Borel $\sigma$-algebra** instead — generated by open intervals — which excludes pathological non-measurable sets (Vitali sets, Banach-Tarski) while including all reasonable events.

---

**Exercise 6.**
**Continuous sample space and "probability zero" events.** For $\Omega = [0, 1]$ with uniform probability, give two examples of events with probability zero. Are they impossible?

??? success "Solution to Exercise 6"
    Two examples:

    1. **Singleton:** $\{0.5\}$ has probability 0 — the uniform distribution assigns zero mass to any single point.
    2. **Rational numbers:** $\mathbb{Q} \cap [0, 1]$ has probability 0 — the rationals are countable, and any countable union of singletons has probability $\sum 0 = 0$.

    **Are they impossible?** No. "Impossible" means $A = \emptyset$ — the event cannot occur. "Probability zero" means $P(A) = 0$, but $A$ may still be non-empty.

    The distinction matters in continuous probability: when you "draw a uniform random number in $[0, 1]$", the outcome *is* some specific real number. That number has probability zero of being any particular value (including itself!), yet it occurred. The countable-additivity axiom ensures only that countable unions of probability-zero events still have probability zero — uncountable unions can have positive probability (e.g., the uncountable union of all singletons in $[0, 1]$ is $[0, 1]$ with probability 1).

    This is why mathematicians distinguish "almost sure" ($P = 1$, the complement has $P = 0$, but the complement need not be empty) from "certain" ($A = \Omega$, the complement is empty). The two coincide for finite $\Omega$ but diverge for continuous $\Omega$.
