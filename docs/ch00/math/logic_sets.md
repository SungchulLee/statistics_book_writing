# Sets, Functions, and Logic

This section reviews the foundational mathematical language used throughout the book. Precise definitions of logic, sets, and functions prevent ambiguity in probability, random variables, and statistical inference.

## Definition

A **proposition** is a declarative sentence that is either true or false. The fundamental connectives are:

| Symbol | Name | Read as |
|---|---|---|
| $\neg P$ | Negation | "not $P$" |
| $P \land Q$ | Conjunction | "$P$ and $Q$" |
| $P \lor Q$ | Disjunction | "$P$ or $Q$" (inclusive) |
| $P \Rightarrow Q$ | Implication | "if $P$ then $Q$" |
| $P \Leftrightarrow Q$ | Biconditional | "$P$ if and only if $Q$" |

A **set** is an unordered collection of distinct objects. Standard operations include union ($A \cup B$), intersection ($A \cap B$), difference ($A \setminus B$), and complement ($A^c$).

A **function** $f: A \to B$ assigns to each $x \in A$ exactly one $f(x) \in B$.

## Explanation

**Quantifiers** formalize "for all" ($\forall$) and "there exists" ($\exists$). Negation swaps them:

$$
\neg(\forall\, x,\; P(x)) \;\Leftrightarrow\; \exists\, x \text{ s.t. } \neg P(x)
$$

**De Morgan's Laws** for sets generalize to arbitrary collections and are used constantly in probability:

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

**Countability** distinguishes discrete from continuous settings. A set is countably infinite if it bijects onto $\mathbb{N}$ (e.g., $\mathbb{Z}$, $\mathbb{Q}$). Uncountable sets like $\mathbb{R}$ require integration rather than summation for probability measures.

**Key function types** in statistics: indicator functions $\mathbf{1}_A(x)$ for Bernoulli variables, exponentials $e^x$ for MGFs, logarithms $\ln x$ for log-likelihoods, and the logistic sigmoid $\sigma(x) = 1/(1 + e^{-x})$ for classification.

!!! note "Vacuous Truth"
    When $P$ is false, $P \Rightarrow Q$ is true regardless of $Q$. This matters when conditioning on probability-zero events.

## Examples

```python
import numpy as np

# De Morgan's law with finite sets
omega = set(range(1, 11))
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7}

A_c = omega - A
B_c = omega - B

# (A ∪ B)^c = A^c ∩ B^c
lhs = omega - (A | B)
rhs = A_c & B_c
print(f"(A ∪ B)^c = {lhs}")
print(f"A^c ∩ B^c = {rhs}")
print(f"Equal: {lhs == rhs}")

# Indicator function
def indicator(x, S):
    return 1 if x in S else 0

vals = [indicator(x, A) for x in range(1, 11)]
print(f"1_A for x=1..10: {vals}")
print(f"|A| via indicator sum: {sum(vals)}")
```
