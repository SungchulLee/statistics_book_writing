# Independence of Events

## Overview

Two events are **independent** if knowing that one has occurred provides no information about whether the other has occurred. Independence is a fundamental concept that simplifies probability calculations and underpins key results like the Law of Large Numbers and the Central Limit Theorem.

---

## Definition

Events $A$ and $B$ are **independent** if and only if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

Equivalently, if $P(B) > 0$:

$$
P(A \mid B) = P(A)
$$

**Interpretation:** Conditioning on $B$ does not change the probability of $A$—learning that $B$ occurred gives no new information about $A$.

---

## Independence vs. Mutual Exclusivity

These two concepts are frequently confused but are fundamentally different:

| Property | Independent | Mutually Exclusive |
|:---|:---|:---|
| **Condition** | $P(A \cap B) = P(A) \cdot P(B)$ | $P(A \cap B) = 0$ |
| **Can both occur?** | Yes | No |
| **Knowing one affects the other?** | No | Yes (the other cannot occur) |

If $A$ and $B$ are mutually exclusive with $P(A) > 0$ and $P(B) > 0$, then they are **not** independent:

$$
P(A \cap B) = 0 \neq P(A) \cdot P(B) > 0
$$

---

## Independence of Multiple Events

Events $A_1, A_2, \ldots, A_n$ are **mutually independent** if for every subset $S \subseteq \{1, 2, \ldots, n\}$:

$$
P\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)
$$

**Pairwise independence** alone is not sufficient for mutual independence. Pairwise independence requires only that every pair satisfies the product rule, but mutual independence requires the product rule for all subsets of any size.

---

## Examples

### Example: Coin Flips

Flip a fair coin twice. Let $A$ = "first flip is heads" and $B$ = "second flip is heads."

$$
P(A) = \frac{1}{2}, \quad P(B) = \frac{1}{2}, \quad P(A \cap B) = \frac{1}{4} = P(A) \cdot P(B)
$$

The flips are independent—the outcome of the first flip has no effect on the second.

### Example: Rolling a Die

Roll a fair die. Let $A$ = "result is even" = $\{2, 4, 6\}$ and $B$ = "result is $\leq 3$" = $\{1, 2, 3\}$.

$$
P(A) = \frac{1}{2}, \quad P(B) = \frac{1}{2}, \quad P(A \cap B) = P(\{2\}) = \frac{1}{6}
$$

Since $\frac{1}{6} \neq \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}$, events $A$ and $B$ are **not** independent.

### Example: Pairwise but Not Mutually Independent

Flip two fair coins. Define:

- $A$ = "first coin is heads"
- $B$ = "second coin is heads"
- $C$ = "both coins show the same face"

Every pair is independent: $P(A \cap B) = P(A)P(B)$, $P(A \cap C) = P(A)P(C)$, $P(B \cap C) = P(B)P(C)$, each equaling $\frac{1}{4}$. However:

$$
P(A \cap B \cap C) = P(\{HH\}) = \frac{1}{4} \neq P(A) \cdot P(B) \cdot P(C) = \frac{1}{8}
$$

The events are pairwise independent but not mutually independent.

---

## Python Exploration

```python
import numpy as np
from itertools import product

def check_independence(sample_space, prob, event_A, event_B, labels=("A", "B")):
    """Check whether two events are independent."""
    p_A = sum(prob[s] for s in event_A)
    p_B = sum(prob[s] for s in event_B)
    p_AB = sum(prob[s] for s in event_A & event_B)

    independent = np.isclose(p_AB, p_A * p_B)

    print(f"P({labels[0]}) = {p_A:.4f}")
    print(f"P({labels[1]}) = {p_B:.4f}")
    print(f"P({labels[0]} ∩ {labels[1]}) = {p_AB:.4f}")
    print(f"P({labels[0]}) × P({labels[1]}) = {p_A * p_B:.4f}")
    print(f"Independent: {independent}\n")

# Die roll example
outcomes = {i: 1/6 for i in range(1, 7)}
A = {2, 4, 6}       # even
B = {1, 2, 3}       # ≤ 3
check_independence(outcomes, outcomes, A, B, ("Even", "≤3"))

# Two coin flips
flips = list(product(['H', 'T'], repeat=2))
prob = {f: 0.25 for f in flips}
A = {f for f in flips if f[0] == 'H'}
B = {f for f in flips if f[1] == 'H'}
check_independence(flips, prob, A, B, ("1st=H", "2nd=H"))
```

```python
import numpy as np

def simulate_independence(n_simulations=200_000):
    """Verify independence of coin flips by simulation."""
    np.random.seed(42)
    flip1 = np.random.randint(0, 2, size=n_simulations)  # 0=T, 1=H
    flip2 = np.random.randint(0, 2, size=n_simulations)

    p_A = flip1.mean()
    p_B = flip2.mean()
    p_AB = ((flip1 == 1) & (flip2 == 1)).mean()

    print(f"P(H₁) = {p_A:.4f},  P(H₂) = {p_B:.4f}")
    print(f"P(H₁ ∩ H₂) = {p_AB:.4f}")
    print(f"P(H₁) × P(H₂) = {p_A * p_B:.4f}")

simulate_independence()
```

---

## Key Takeaways

- Independence means $P(A \cap B) = P(A) \cdot P(B)$: knowing one event tells you nothing about the other.
- Independence and mutual exclusivity are **opposite** in spirit—mutually exclusive events are maximally dependent.
- Pairwise independence does not imply mutual independence; the product rule must hold for **all** subsets.
- Independence of random variables (discussed later) extends this concept: $X$ and $Y$ are independent if their joint distribution factors into the product of marginals.
