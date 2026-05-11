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

## Exercises

**Exercise 1.**
Using only the three axioms of probability, prove that $P(A^c) = 1 - P(A)$ for any event $A$.

??? success "Solution to Exercise 1"
    Since $A$ and $A^c$ are disjoint (mutually exclusive) and $A \cup A^c = \Omega$, the additivity axiom gives:

    $$
    P(A \cup A^c) = P(A) + P(A^c)
    $$

    By the normalization axiom, $P(\Omega) = 1$, so:

    $$
    1 = P(A) + P(A^c)
    $$

    Rearranging:

    $$
    P(A^c) = 1 - P(A)
    $$

    $\square$

---

**Exercise 2.**
Prove from the axioms that for any two events $A$ and $B$:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

??? success "Solution to Exercise 2"
    Write $A \cup B$ as a disjoint union. Note that $A \cup B = A \cup (B \cap A^c)$, where $A$ and $B \cap A^c$ are disjoint. By the additivity axiom:

    $$
    P(A \cup B) = P(A) + P(B \cap A^c)
    $$

    Similarly, $B = (B \cap A) \cup (B \cap A^c)$ is a disjoint union, so:

    $$
    P(B) = P(B \cap A) + P(B \cap A^c)
    $$

    Solving for $P(B \cap A^c)$:

    $$
    P(B \cap A^c) = P(B) - P(A \cap B)
    $$

    Substituting back:

    $$
    P(A \cup B) = P(A) + P(B) - P(A \cap B)
    $$

    $\square$

---

**Exercise 3.**
A student claims that $P(A) = 0.4$, $P(B) = 0.5$, and $P(A \cup B) = 0.8$. Another student claims that $P(A) = 0.7$, $P(B) = 0.6$, and $P(A \cap B) = 0.1$. Determine whether each assignment is consistent with the axioms.

??? success "Solution to Exercise 3"
    **First student:** Using inclusion-exclusion, $P(A \cap B) = P(A) + P(B) - P(A \cup B) = 0.4 + 0.5 - 0.8 = 0.1$. Since $0 \leq 0.1 \leq \min(0.4, 0.5)$ and all probabilities are in $[0,1]$, this assignment is **consistent** with the axioms.

    **Second student:** We need $P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.7 + 0.6 - 0.1 = 1.2$. But the normalization axiom requires $P(A \cup B) \leq P(\Omega) = 1$. Since $1.2 > 1$, this assignment **violates** the axioms and is therefore impossible.

---

**Exercise 4.**
Using the axioms, prove that if $A \subseteq B$, then $P(A) \leq P(B)$ (monotonicity of probability).

??? success "Solution to Exercise 4"
    Since $A \subseteq B$, we can write $B = A \cup (B \cap A^c)$, where $A$ and $B \cap A^c$ are disjoint. By the additivity axiom:

    $$
    P(B) = P(A) + P(B \cap A^c)
    $$

    By the non-negativity axiom, $P(B \cap A^c) \geq 0$. Therefore:

    $$
    P(B) = P(A) + P(B \cap A^c) \geq P(A)
    $$

    $\square$

---

**Exercise 5.**
**Bonferroni's inequality.** Prove that for any events $A_1, \ldots, A_n$, $P(\bigcup_i A_i) \le \sum_i P(A_i)$. When is this useful?

??? success "Solution to Exercise 5"
    By induction on $n$. Base case $n = 1$: trivially $P(A_1) \le P(A_1)$.

    Inductive step: assume $P(\bigcup_{i=1}^k A_i) \le \sum_{i=1}^k P(A_i)$. By inclusion-exclusion,

    $$
    P(A_1 \cup \cdots \cup A_{k+1}) = P(\bigcup_{i=1}^k A_i) + P(A_{k+1}) - P((\bigcup_{i=1}^k A_i) \cap A_{k+1})
    $$

    The last term is non-negative, so

    $$
    P(\bigcup_{i=1}^{k+1} A_i) \le P(\bigcup_{i=1}^k A_i) + P(A_{k+1}) \le \sum_{i=1}^k P(A_i) + P(A_{k+1}) = \sum_{i=1}^{k+1} P(A_i)
    $$

    $\square$

    **Use:** in **multiple comparisons**, suppose we conduct $n$ hypothesis tests each at level $\alpha$. The probability of at least one false rejection is $P(\bigcup_i \text{Reject}_i \mid H_0) \le n\alpha$ by Bonferroni. So testing at level $\alpha/n$ ensures family-wise error rate $\le \alpha$. The Bonferroni correction is conservative but uniformly valid regardless of dependence structure.

---

**Exercise 6.**
**$\sigma$-additivity vs. finite additivity.** State the difference and explain why measure-theoretic probability requires the stronger property.

??? success "Solution to Exercise 6"
    **Finite additivity:** for disjoint $A_1, \ldots, A_n$ (finite collection), $P(\bigcup_{i=1}^n A_i) = \sum_{i=1}^n P(A_i)$.

    **$\sigma$-additivity** (countable additivity, Kolmogorov's axiom 3): for *countably infinite* sequences of disjoint events $A_1, A_2, \ldots$, $P(\bigcup_{i=1}^\infty A_i) = \sum_{i=1}^\infty P(A_i)$.

    **Why $\sigma$-additivity?** Many useful results require it:

    - **Continuity of measure**: if $A_1 \subseteq A_2 \subseteq \ldots$ and $A = \bigcup A_n$, then $P(A) = \lim P(A_n)$. Crucial for limit theorems.
    - **Existence of probability density**: defining a continuous distribution requires assigning probabilities to arbitrarily fine subdivisions, which is a countable operation.
    - **Strong law of large numbers**: requires almost-sure convergence, defined via countable unions.

    Finite additivity is consistent with paradoxical assignments (e.g., the "uniform" distribution on the integers that's intuitive but not $\sigma$-additive). Kolmogorov adopted $\sigma$-additivity to rule these out and connect probability to measure theory.
