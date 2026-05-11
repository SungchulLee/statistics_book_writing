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

## Exercises

**Exercise 1.**
A fair die is rolled twice. Let $A$ = "sum is 7" and $B$ = "first roll is 3". (a) Sample space size? (b) List $A$, $B$, $A \cap B$. (c) Compute $P(A), P(B), P(A \cap B)$. (d) Are $A$ and $B$ independent?

??? success "Solution to Exercise 1"
    (a) $\Omega = \{(i, j) : i, j \in \{1,\ldots,6\}\}$, $|\Omega| = 36$.

    (b) $A = \{(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)\}$. $B = \{(3,1),(3,2),(3,3),(3,4),(3,5),(3,6)\}$. $A \cap B = \{(3,4)\}$.

    (c) $P(A) = 6/36 = 1/6$, $P(B) = 1/6$, $P(A \cap B) = 1/36$.

    (d) $P(A) \cdot P(B) = 1/36 = P(A \cap B)$, so independent. Intuition: knowing first roll = 3 doesn't change the probability that the sum is 7, because exactly one of six second-roll values (namely 4) produces sum 7.

---

**Exercise 2.**
Prove: if $A$ and $B$ are independent, then $A^c$ and $B$ are also independent. Likewise $A$ and $B^c$, and $A^c$ and $B^c$.

??? success "Solution to Exercise 2"
    By independence: $P(A \cap B) = P(A) P(B)$.

    $P(A^c \cap B) = P(B) - P(A \cap B) = P(B) - P(A) P(B) = P(B)[1 - P(A)] = P(B) P(A^c)$.

    So $A^c$ and $B$ are independent. By symmetric reasoning, $A$ and $B^c$ are independent. Applying the same argument to $A^c$ and $B^c$ — start from $A$ and $B^c$ independent, deduce $A^c$ and $B^c$ independent.

    Lesson: independence is preserved under complementation. Sometimes it is easier to verify independence using complements (e.g., $P(\text{no failures}) = \prod P(\text{component } i \text{ works})$ in reliability).

---

**Exercise 3.**
**Pairwise independence does not imply mutual independence.** Flip two fair coins. Let $A$ = "1st is H", $B$ = "2nd is H", $C$ = "both coins match". Show that $\{A, B, C\}$ are pairwise independent but **not** mutually independent.

??? success "Solution to Exercise 3"
    Sample space $\{HH, HT, TH, TT\}$, each with probability 1/4.

    $P(A) = P(B) = P(C) = 1/2$ (two outcomes each).

    Pairwise checks:
    - $A \cap B = \{HH\}$, $P = 1/4 = P(A) P(B)$. ✓
    - $A \cap C = \{HH\}$, $P = 1/4 = P(A) P(C)$. ✓
    - $B \cap C = \{HH\}$, $P = 1/4 = P(B) P(C)$. ✓

    Mutual check: $A \cap B \cap C = \{HH\}$, $P = 1/4 \ne P(A) P(B) P(C) = 1/8$.

    So the three events are pairwise but not mutually independent. Intuitively, $A$ and $B$ together determine $C$ — knowing both eliminates the randomness in the third. This is why mutual independence requires the product rule for *every* subset, not just pairs.

---

**Exercise 4.**
**Independence is not transitive.** Construct three events $A$, $B$, $C$ such that $A$ is independent of $B$ and $B$ is independent of $C$, but $A$ is **not** independent of $C$.

??? success "Solution to Exercise 4"
    Roll a fair die. Let:

    - $A$ = "result is even" = $\{2, 4, 6\}$
    - $B$ = "result is $\le 3$" = $\{1, 2, 3\}$
    - $C$ = "result is even" = $\{2, 4, 6\}$ (same as $A$ — we'll fix this)

    Better: take $A = \{1, 2, 3, 4\}$, $B = \{2, 3, 5, 6\}$, $C = \{1, 2, 3, 4\}$. Then $C = A$, so trivially not independent. Let's redo.

    Construction: roll two dice, consider results $X_1, X_2$.

    - $A$ = "$X_1$ is even" — $P(A) = 1/2$.
    - $B$ = "$X_1 + X_2$ is even" — $P(B) = 1/2$.
    - $C$ = "$X_2$ is even" — $P(C) = 1/2$.

    Check $A$ ⊥ $B$: $P(A \cap B)$ = "first die even AND sum even" = "first even AND second even" = $1/4$. $P(A) P(B) = 1/4$. ✓

    Check $B$ ⊥ $C$: by symmetry, same value, ✓.

    Check $A$ ⊥ $C$: $P(A \cap C)$ = "both dice even" = $1/4$. $P(A) P(C) = 1/4$. ✓

    Oops — this example has all three pairwise independent. Let me try another.

    Better: roll one die.
    - $A = \{1, 2, 3\}$, $P(A) = 1/2$.
    - $B = \{1, 4\}$, $P(B) = 1/3$.
    - $C = \{4, 5, 6\}$, $P(C) = 1/2$.

    $A \cap B = \{1\}$, $P = 1/6 = P(A) P(B)$ ✓.
    $B \cap C = \{4\}$, $P = 1/6 = P(B) P(C)$ ✓.
    $A \cap C = \emptyset$, $P = 0 \ne P(A) P(C) = 1/4$. ✗

    So $A$ ⊥ $B$, $B$ ⊥ $C$, but $A$ and $C$ are not independent. Independence is not transitive.

---

**Exercise 5.**
**Independence of complementary information.** If $X$ is uniform on $\{1, 2, 3, 4\}$, consider $A$ = "$X$ is even" and $B$ = "$X \le 2$". Show these are independent. Generalize: under what symmetry condition on a uniform distribution can two events partition into independent pieces?

??? success "Solution to Exercise 5"
    $A = \{2, 4\}$, $P(A) = 1/2$. $B = \{1, 2\}$, $P(B) = 1/2$. $A \cap B = \{2\}$, $P = 1/4 = P(A) P(B)$. Independent. ✓

    **Generalization:** independence here arises because the four outcomes $\{1, 2, 3, 4\}$ can be encoded as bits $(b_1, b_2)$ where $b_1$ = "even" indicator and $b_2$ = "$\le 2$" indicator. The four outcomes are $(\bar b_1, \bar b_2), (b_1, \bar b_2), (\bar b_1, b_2), (b_1, b_2)$ in some order, each with probability 1/4 — making the two bits uniform i.i.d. and hence independent.

    More generally: events $A$ and $B$ are independent under a uniform distribution on a finite sample space $|\Omega| = n$ if and only if the four cell counts $|A \cap B|, |A \cap B^c|, |A^c \cap B|, |A^c \cap B^c|$ form a $2 \times 2$ matrix with rank 1 — proportional rows and columns. This is the discrete analog of the rank-1 factorization property of independent joint distributions.

---

**Exercise 6.**
**Independence is a structural property, not a frequency property.** A sample of size $n$ can show $P(\hat A \cap \hat B) = \hat P(\hat A) \hat P(\hat B)$ by chance even when $A$ and $B$ are dependent in the population. Derive the expected value of $|\hat P(A \cap B) - \hat P(A) \hat P(B)|$ under the null of independence with sample size $n$.

??? success "Solution to Exercise 6"
    Define $T = \hat P(A \cap B) - \hat P(A) \hat P(B)$. Under independence, the population value of $T$ is $0$. The sample $T$ has approximate distribution

    $$
    T \approx \frac{1}{n} \sum_i \mathbf{1}(X_i \in A) \mathbf{1}(X_i \in B) - \bar A_n \bar B_n
    $$

    Asymptotically, $\sqrt n \cdot T \xrightarrow{d} N(0, \sigma^2)$ where $\sigma^2 = P(A) P(B)(1 - P(A))(1 - P(B))$ (a Cramér–Wold / delta-method calculation).

    Therefore $\mathbb{E}|T| \approx \sqrt{2/\pi} \cdot \sigma/\sqrt{n} = O(n^{-1/2})$.

    **Consequence:** a non-zero $T$ in a finite sample does *not* prove dependence; it has standard error of order $n^{-1/2}$. The **chi-squared test of independence** formalizes this: under $H_0$ the test statistic has a chi-squared distribution, and only large values are evidence of dependence.

    This is the same lesson as elsewhere: structural population properties (independence, no causation) cannot be conclusively *demonstrated* from sample data — only the *failure* of independence (significant departure from $T = 0$) provides evidence against the null.
